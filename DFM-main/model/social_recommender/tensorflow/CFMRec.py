import math
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sp_linalg

from model.base.abstract_recommender import SocialAbstractRecommender
from data.data_iterator import DataIterator
from reckit import timer
from util.common.tool import csr_to_user_dict_bytime

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from util.common.tool import batch_randint_choice
from reckit import pad_sequences
from util.tensorflow import get_session

epsilon = 1e-9


def mean_flat_tf(tensor):
    """
    TensorFlow版本的mean_flat。
    计算一个张量除了第一个维度（批次维度）外所有其他维度的均值。
    """
    return tf.reduce_mean(
        tensor,
        axis=list(range(1, len(tensor.shape.as_list())))
    )


class FlowMatchingProcessTF:
    """
    TensorFlow 1.x版本的Flow Matching过程。
    支持：
    1. 高斯噪声作为源状态；
    2. 指定表征作为源状态；
    3. 条件流匹配；
    4. ODE数值积分推理。
    """

    def __init__(self, steps: int, t_dist: str = "uniform",
                 eps: float = 1e-4):

        self.steps = steps
        self.t_dist = t_dist
        self.eps = eps
        self.default_steps_fallback = 80
        self.default_integrator = "euler"

    def sample_times_tf(self, batch_size_tensor, method: str = None):
        """
        时间采样。
        """
        method = method or self.t_dist

        if method == "uniform":
            t = tf.random.uniform(
                [batch_size_tensor],
                minval=0.0,
                maxval=1.0,
                dtype=tf.float32
            )

        elif method == "cosine":
            u = tf.random.uniform(
                [batch_size_tensor],
                minval=self.eps,
                maxval=1.0 - self.eps,
                dtype=tf.float32
            )
            t = tf.acos(1.0 - 2.0 * u) / math.pi

        elif method == "beta22":
            u1 = tf.random.uniform(
                [batch_size_tensor], minval=0.0,
                maxval=1.0, dtype=tf.float32
            )
            u2 = tf.random.uniform(
                [batch_size_tensor], minval=0.0,
                maxval=1.0, dtype=tf.float32
            )
            t = tf.clip_by_value(
                tf.minimum(u1, u2),
                self.eps,
                1.0 - self.eps
            )

        else:
            raise ValueError("Unknown t sampling: {}".format(method))

        return t

    def _expand_t_tf(self, t, like_tensor):
        """
        将时间步t扩展为适合数据张量的形状。
        """
        like_dims = like_tensor.shape.as_list()

        while len(t.shape.as_list()) < len(like_dims):
            t = tf.expand_dims(t, -1)

        return t

    def path_xt_tf(self, x0, x1, t):
        """
        线性概率路径：
        x_t = (1-t)x_0 + tx_1
        """
        t_ = self._expand_t_tf(t, x1)
        return (1.0 - t_) * x0 + t_ * x1

    def target_velocity_tf(self, x0, x1):
        """
        目标向量场：
        v = x_1 - x_0
        """
        return x1 - x0

    def caculate_losses_tf(self, model_fn, emb_s,
                           reweight=True, condition=None, x0=None):
        """
        条件流匹配损失。

        x0:
            源状态。
            如果为None，则从标准高斯分布采样。

        emb_s:
            目标状态。

        condition:
            条件信息。

        x_t:
            (1-t)*x0 + t*emb_s
        """

        batch_size = tf.shape(emb_s)[0]

        t = self.sample_times_tf(batch_size)

        # 源状态：
        # 长期流：高斯噪声
        # 短期流：显式传入的序列/社交表示
        if x0 is None:
            source = tf.random.normal(
                tf.shape(emb_s),
                dtype=emb_s.dtype
            )
        else:
            source = x0

        # 线性插值
        x_t = self.path_xt_tf(source, emb_s, t)

        # 目标速度
        target_velocity = self.target_velocity_tf(
            source, emb_s
        )

        # 条件向量场预测
        predicted_velocity = model_fn(
            x_t, t, condition
        )

        # Flow Matching MSE
        mse = mean_flat_tf(
            tf.square(predicted_velocity - target_velocity)
        )

        if reweight:
            weight = tf.clip_by_value(
                t * (1.0 - t),
                self.eps,
                1.0
            )
        else:
            weight = tf.ones_like(t)

        loss = weight * mse

        return {
            "loss": loss,
            "pred_xstart": source + predicted_velocity
        }

    def p_sample_tf(self, model_fn, shape_ref_tensor, steps,
                    integrator=None, condition=None, x_start=None):
        """
        ODE数值积分推理。

        x_start=None：
            从高斯噪声开始，适用于长期流匹配。

        x_start不为None：
            从指定源状态开始，适用于短期两个流匹配分支。
        """

        integrator = integrator or self.default_integrator

        if steps is None or steps <= 0:
            steps = self.steps

        if steps <= 0:
            raise ValueError("flow steps must be positive")

        if x_start is None:
            x = tf.random.normal(
                tf.shape(shape_ref_tensor),
                dtype=shape_ref_tensor.dtype
            )
        else:
            x = x_start

        batch_size = tf.shape(x)[0]

        ts = tf.linspace(
            0.0, 1.0, steps + 1
        )

        for i in range(steps):

            t_i_scalar = ts[i]
            t_next_scalar = ts[i + 1]

            dt = t_next_scalar - t_i_scalar

            t_i = tf.fill(
                [batch_size], t_i_scalar
            )

            t_next = tf.fill(
                [batch_size], t_next_scalar
            )

            if integrator == "euler":

                v = model_fn(
                    x, t_i, condition
                )

                x = x + dt * v

            elif integrator == "heun":

                v_i = model_fn(
                    x, t_i, condition
                )

                x_euler = x + dt * v_i

                v_next = model_fn(
                    x_euler, t_next, condition
                )

                x = x + 0.5 * dt * (
                    v_i + v_next
                )

            elif integrator == "rk4":

                t_mid = tf.fill(
                    [batch_size],
                    t_i_scalar + 0.5 * dt
                )

                k1 = model_fn(
                    x, t_i, condition
                )

                k2 = model_fn(
                    x + 0.5 * dt * k1,
                    t_mid, condition
                )

                k3 = model_fn(
                    x + 0.5 * dt * k2,
                    t_mid, condition
                )

                k4 = model_fn(
                    x + dt * k3,
                    t_next, condition
                )

                x = x + (dt / 6.0) * (
                    k1 + 2.0 * k2 +
                    2.0 * k3 + k4
                )

            else:
                raise ValueError(
                    "Unknown integrator: {}".format(integrator)
                )
        return x

    def forward_process_tf(self, emb_s, t, noise=None):
        if noise is None:
            noise = tf.random.normal(
                tf.shape(emb_s)
            )

        return self.path_xt_tf(
            noise, emb_s, t
        )


def timestep_embedding_tf(timesteps, dim, max_period=10000):
    """
    正弦时间步嵌入。
    """

    half = dim // 2

    freqs = tf.exp(
        -math.log(max_period) *
        tf.range(
            start=0,
            limit=half,
            dtype=tf.float32
        ) / half
    )

    args = (
        tf.expand_dims(
            tf.cast(timesteps, tf.float32), 1
        ) *
        tf.expand_dims(freqs, 0)
    )

    embedding = tf.concat(
        [tf.cos(args), tf.sin(args)],
        axis=-1
    )

    if dim % 2:
        embedding = tf.concat(
            [
                embedding,
                tf.zeros_like(embedding[:, :1])
            ],
            axis=-1
        )

    return embedding


class SDNetTF:
    """
    条件向量场预测网络。

    通过condition进行AdaLN调制。
    两个短期流匹配分支共享同一个SDNetTF实例，
    但由于输入的condition不同，会产生不同的条件向量场。
    """

    def __init__(self, name, hidden_dims, emb_size,
                 norm=False, dropout_rate=0.5):

        self.name = name
        self.hidden_dims = hidden_dims
        self.time_emb_dim = emb_size
        self.norm = norm
        self.dropout_rate = dropout_rate

        with tf.variable_scope(
            self.name, reuse=tf.AUTO_REUSE
        ):

            self.emb_layer = tf.keras.layers.Dense(
                units=self.time_emb_dim,
                name="time_emb_layer"
            )

            self.dropout_layer = tf.keras.layers.Dropout(
                rate=self.dropout_rate
            )

            self.hidden_layers = []

            for i, h_dim in enumerate(hidden_dims[:-1]):

                self.hidden_layers.append(
                    tf.keras.layers.Dense(
                        units=h_dim,
                        activation='tanh',
                        name="hidden_layer_{}".format(i)
                    )
                )

            self.output_layer = tf.keras.layers.Dense(
                units=hidden_dims[-1],
                activation=None,
                name="output_layer"
            )

            self.adaLN_generators = []

            for i, h_dim in enumerate(hidden_dims[:-1]):

                self.adaLN_generators.append(
                    tf.keras.layers.Dense(
                        units=h_dim * 2,
                        name="adaln_gen_{}".format(i)
                    )
                )

            self.adaln_layer_norms = []

            for i, _ in enumerate(hidden_dims[:-1]):

                self.adaln_layer_norms.append(
                    tf.keras.layers.LayerNormalization(
                        axis=-1,
                        center=False,
                        scale=False,
                        name="adaln_ln_{}".format(i)
                    )
                )

    def __call__(self, x, timesteps,
                 is_training=True, condition=None):

        with tf.variable_scope(
            self.name, reuse=tf.AUTO_REUSE
        ):

            time_emb = timestep_embedding_tf(
                timesteps, self.time_emb_dim
            )

            emb = self.emb_layer(time_emb)

            h = tf.concat(
                [x, emb],
                axis=-1
            )

            if condition is not None:

                # 条件信息与时间嵌入共同调制隐藏层
                merged_condition = emb + condition

                for i, layer in enumerate(
                    self.hidden_layers
                ):

                    style_params = self.adaLN_generators[i](
                        merged_condition
                    )

                    gamma, beta = tf.split(
                        style_params,
                        2,
                        axis=-1
                    )

                    h_normalized = self.adaln_layer_norms[i](
                        h
                    )

                    h = (
                        gamma * h_normalized + beta
                    )

                    h = layer(h)

                    h = self.dropout_layer(
                        h,
                        training=is_training
                    )

            else:

                for layer in self.hidden_layers:

                    h = layer(h)

                    h = self.dropout_layer(
                        h,
                        training=is_training
                    )

            output = self.output_layer(h)

            return output


class CFMRec(SocialAbstractRecommender):

    def __init__(self, conf):

        super(CFMRec, self).__init__(conf)

        self.num_users, self.num_items = (
            self.dataset.train_matrix.shape
        )

        self.lr = conf["lr"]
        self.l2_reg = conf["l2_reg"]
        self.embedding_size = conf["embedding_size"]
        self.seq_l = conf["seq_l"]
        self.batch_size = conf["batch_size"]
        self.epochs = conf["epochs"]

        self.SocialtrainDict = self._get_SocialDict()

        self.n_layers = conf['n_layers']

        self.norm_adj = self.create_adj_mat()
        self.social_norm_adj = self._create_social_adj_mat()

        self.long_term_flow_loss_weight = (
            conf["long_term_flow_loss_weight"]
        )

        self.flow_loss_weight = (
            conf["flow_loss_weight"]
        )

        self.social_flow_loss_weight = (
            conf["social_flow_loss_weight"]
        )

        self.flow_steps = conf["flow_steps"]
        self.dropout_rate = conf["dropout_rate"]

        self.reweight = conf["reweight"]

        self.ema_decay = conf["ema_decay"]

        self.sess = get_session(
            conf["gpu_mem"]
        )

        self.build_graph()

        self.saver = tf.train.Saver(
            max_to_keep=5
        )

        self.sess.run(
            tf.global_variables_initializer()
        )

    def _get_SocialDict(self):

        SocialDict = {}

        for u in range(self.num_users):

            trustors = self.social_matrix[u].indices

            SocialDict[u] = (
                trustors.tolist()
                if len(trustors) > 0
                else []
            )

        return SocialDict

    @timer
    def create_adj_mat(self):

        n_nodes = self.num_users + self.num_items

        user_list, item_list, _ = (
            self.dataset.get_train_interactionssecond()
        )

        user_np = np.array(
            user_list,
            dtype=np.int32
        )

        item_np = np.array(
            item_list,
            dtype=np.int32
        )

        ratings = np.ones_like(
            user_np,
            dtype=np.float32
        )

        tmp_adj = sp.csr_matrix(
            (
                ratings,
                (user_np, item_np + self.num_users)
            ),
            shape=(n_nodes, n_nodes)
        )

        adj_mat = tmp_adj + tmp_adj.T

        rowsum = np.array(
            adj_mat.sum(1)
        )

        d_inv = np.power(
            rowsum, -0.5
        ).flatten()

        d_inv[np.isinf(d_inv)] = 0.
        d_inv[np.isnan(d_inv)] = 0.

        d_mat_inv = sp.diags(d_inv)

        norm_adj_tmp = d_mat_inv.dot(
            adj_mat
        )

        adj_matrix = norm_adj_tmp.dot(
            d_mat_inv
        )

        print(
            'use the pre adjacency matrix for user-item graph'
        )

        return adj_matrix

    def _create_social_adj_mat(self):

        uu_idx = [
            [ui, uj]
            for (ui, uj), r
            in self.social_matrix.todok().items()
        ]

        u1_idx, u2_idx = list(
            zip(*uu_idx)
        )

        self.u1_idx = tf.constant(
            u1_idx,
            dtype=tf.int32,
            shape=None,
            name="u1_idx"
        )

        self.u2_idx = tf.constant(
            u2_idx,
            dtype=tf.int32,
            shape=None,
            name="u2_idx"
        )

        u1_idx = np.array(
            u1_idx,
            dtype=np.int32
        )

        u2_idx = np.array(
            u2_idx,
            dtype=np.int32
        )

        ratings = np.ones_like(
            u1_idx,
            dtype=np.float32
        )

        tmp_adj = sp.csr_matrix(
            (ratings, (u1_idx, u2_idx)),
            shape=(self.num_users, self.num_users)
        )

        adj_mat = tmp_adj + tmp_adj.T

        return self._normalize_spmat(
            adj_mat
        )

    def _normalize_spmat(self, adj_mat):

        rowsum = np.array(
            adj_mat.sum(1)
        )

        d_inv = np.power(
            rowsum, -0.5
        ).flatten()

        d_inv[np.isinf(d_inv)] = 0.
        d_inv[np.isnan(d_inv)] = 0.

        d_mat_inv = sp.diags(d_inv)

        norm_adj_tmp = d_mat_inv.dot(
            adj_mat
        )

        adj_matrix = norm_adj_tmp.dot(
            d_mat_inv
        )

        print(
            'use the pre adjacency matrix for social graph'
        )

        return adj_matrix

    def _convert_sp_mat_to_sp_tensor(self, X):

        coo = X.tocoo().astype(
            np.float32
        )

        indices = np.mat(
            [coo.row, coo.col]
        ).transpose()

        return tf.SparseTensor(
            indices,
            coo.data,
            coo.shape
        )

    def _lightgcn_propagate(self, norm_adj, embeddings):
        """
        通用LightGCN传播函数。
        """

        all_embeddings = [embeddings]

        for k in range(self.n_layers):

            embeddings = tf.sparse.sparse_dense_matmul(
                norm_adj,
                embeddings
            )

            all_embeddings.append(
                embeddings
            )

        return tf.reduce_mean(
            tf.stack(all_embeddings, axis=1),
            axis=1
        )

    def _create_placeholder(self):

        self.user_ph = tf.placeholder(
            tf.int32,
            [None],
            name="user"
        )

        self.item_seq_ph = tf.placeholder(
            tf.int32,
            [None, self.seq_l],
            name="item_seq"
        )

        self.item_pos_ph = tf.placeholder(
            tf.int32,
            [None],
            name="item_pos"
        )

        self.item_neg_ph = tf.placeholder(
            tf.int32,
            [None],
            name="item_neg"
        )

        self.is_training_ph = tf.placeholder_with_default(
            True,
            shape=(),
            name="is_training"
        )

        self.restore_vars_ph = tf.placeholder(
            tf.float32,
            shape=[None, None]
        )

    def _create_variable(self):

        self.embeddings = dict()

        initializer = tf.glorot_uniform_initializer()

        self.embeddings["user_embeddings"] = tf.Variable(
            initializer(
                [self.num_users, self.embedding_size]
            )
        )

        self.embeddings["item_embeddings"] = tf.Variable(
            initializer(
                [self.num_items, self.embedding_size]
            )
        )

        self.embeddings["social_init_embeddings"] = tf.Variable(
            initializer(
                [self.num_users, self.embedding_size]
            )
        )

        self.embeddings["seq_item_embeddings"] = tf.Variable(
            initializer(
                [self.num_items + 1, self.embedding_size]
            )
        )

        self.embeddings["position_embeddings"] = tf.Variable(
            initializer(
                [self.seq_l, self.embedding_size]
            )
        )

        self.global_embedding = tf.Variable(
            initializer(
                [1, self.embedding_size]
            )
        )

        self.item_biases = tf.Variable(
            tf.zeros([self.num_items])
        )

        # =====================================================
        # 1. 长期流匹配预测网络
        # =====================================================

        self.flow_process = FlowMatchingProcessTF(
            steps=self.flow_steps,
            t_dist="cosine"
        )

        long_term_hidden_dims = [
            self.embedding_size * 2,
            self.embedding_size
        ]

        self.long_term_predictor = SDNetTF(
            name="long_term_predictor",
            hidden_dims=long_term_hidden_dims,
            emb_size=self.embedding_size,
            dropout_rate=self.dropout_rate
        )

        # =====================================================
        # 2. 短期流匹配预测网络
        # =====================================================

        short_term_hidden_dims = [
            self.embedding_size * 2,
            self.embedding_size
        ]

        self.short_term_predictor = SDNetTF(
            name="short_term_predictor",
            hidden_dims=short_term_hidden_dims,
            emb_size=self.embedding_size,
            dropout_rate=self.dropout_rate
        )

    def _gated_fusion(self, vec_a, vec_b, name_scope):

        with tf.variable_scope(
            name_scope,
            reuse=tf.AUTO_REUSE
        ):

            combined = tf.concat(
                [vec_a, vec_b],
                axis=-1
            )

            gate = tf.layers.dense(
                combined,
                units=self.embedding_size,
                activation=tf.nn.sigmoid,
                name="gate_weight",
                kernel_initializer=tf.glorot_uniform_initializer()
            )

            return (
                gate * vec_a +
                (1.0 - gate) * vec_b
            )

    def _create_inference(self):

        batch_size = tf.shape(
            self.user_ph
        )[0]

        # =====================================================
        # Step 0: User-Item Graph和Social Graph表征学习
        # =====================================================

        ui_adj_tensor = self._convert_sp_mat_to_sp_tensor(
            self.norm_adj
        )

        social_adj_tensor = self._convert_sp_mat_to_sp_tensor(
            self.social_norm_adj
        )

        # User-Item Graph
        ui_embs_initial = tf.concat(
            [
                self.embeddings["user_embeddings"],
                self.embeddings["item_embeddings"]
            ],
            axis=0
        )

        gcn_ui_embs = self._lightgcn_propagate(
            ui_adj_tensor,
            ui_embs_initial
        )

        user_gcn_embs, item_gcn_embs = tf.split(
            gcn_ui_embs,
            [self.num_users, self.num_items],
            axis=0
        )

        # 用户协同嵌入
        batch_user_gcn_embs = tf.nn.embedding_lookup(
            user_gcn_embs,
            self.user_ph
        )

        # Social Graph
        social_gcn_embs = self._lightgcn_propagate(
            social_adj_tensor,
            self.embeddings["social_init_embeddings"]
        )

        # 用户社交嵌入
        batch_social_gcn_embs = tf.nn.embedding_lookup(
            social_gcn_embs,
            self.user_ph
        )

        # =====================================================
        # A. Long-term Socially-guided Flow Matching
        #
        # Source: Gaussian Noise
        # Target: Collaborative Embedding
        # Condition: Social Embedding
        # =====================================================

        long_term_model_fn = lambda x_t, t, cond: (
            self.long_term_predictor(
                x_t,
                t,
                self.is_training_ph,
                condition=cond
            )
        )

        long_term_losses = self.flow_process.caculate_losses_tf(
            long_term_model_fn,
            emb_s=batch_user_gcn_embs,
            reweight=self.reweight,
            condition=batch_social_gcn_embs,
            x0=None
        )

        self.long_term_flow_loss = tf.reduce_mean(
            long_term_losses["loss"]
        )

        # 长期流匹配推理：从高斯噪声出发
        long_term_output = self.flow_process.p_sample_tf(
            long_term_model_fn,
            shape_ref_tensor=batch_user_gcn_embs,
            steps=self.flow_steps,
            condition=batch_social_gcn_embs,
            x_start=None
        )

        # 长期偏好门控融合
        self.final_long_term_user_embs = self._gated_fusion(
            batch_user_gcn_embs,
            long_term_output,
            "long_term_fusion"
        )

        # =====================================================
        # B. Short-term Sequential-wise Flow Matching
        # =====================================================

        # 1. 编码用户近期行为序列
        seq_item_embs = tf.nn.embedding_lookup(
            self.embeddings["seq_item_embeddings"],
            self.item_seq_ph
        )

        seq_item_embs_with_pos = (
            seq_item_embs +
            self.embeddings["position_embeddings"]
        )

        # 取序列最后一个物品作为Query
        last_item_emb = seq_item_embs[:, -1, :]

        query = tf.expand_dims(
            last_item_emb,
            axis=1
        )

        valid_mask = tf.not_equal(
            self.item_seq_ph,
            self.num_items
        )

        scores = tf.matmul(
            query,
            seq_item_embs_with_pos,
            transpose_b=True
        )

        scores = scores / tf.sqrt(
            tf.cast(
                self.embedding_size,
                tf.float32
            )
        )

        scores += (
            tf.expand_dims(
                tf.cast(
                    tf.logical_not(valid_mask),
                    tf.float32
                ),
                axis=1
            ) * (-1e9)
        )

        attn_weights = tf.nn.softmax(
            scores,
            axis=-1
        )

        # 潜在行为意图表示
        seq_embs_agg = tf.squeeze(
            tf.matmul(
                attn_weights,
                seq_item_embs_with_pos
            ),
            axis=1
        )

        # 下一次交互物品的协同嵌入
        next_item_embedding = tf.nn.embedding_lookup(
            item_gcn_embs,
            self.item_pos_ph
        )

        def short_term_model_fn(x_t, t, cond):

            return self.short_term_predictor(
                x_t,
                t,
                self.is_training_ph,
                condition=cond
            )

        item_flow_loss_list = []
        social_flow_loss_list = []

        curr_I = seq_embs_agg
        curr_S = batch_social_gcn_embs

        # 三轮交替迭代
        for _ in range(3):

            # -------------------------------------------------
            # 1. Sequential-wise Flow
            # I_(k-1) -> I_k | S_(k-1)
            # -------------------------------------------------

            item_losses = self.flow_process.caculate_losses_tf(
                short_term_model_fn,
                emb_s=next_item_embedding,
                reweight=self.reweight,
                condition=curr_S,
                x0=curr_I
            )

            item_flow_loss_list.append(
                item_losses["loss"]
            )

            # 从当前行为意图开始ODE积分
            next_I = self.flow_process.p_sample_tf(
                short_term_model_fn,
                shape_ref_tensor=curr_I,
                steps=self.flow_steps,
                condition=curr_S,
                x_start=curr_I
            )

            # -------------------------------------------------
            # 2. Social-wise Flow
            # S_(k-1) -> S_k | I_k
            # -------------------------------------------------

            social_losses = self.flow_process.caculate_losses_tf(
                short_term_model_fn,
                emb_s=batch_user_gcn_embs,
                reweight=self.reweight,
                condition=next_I,
                x0=curr_S
            )

            social_flow_loss_list.append(
                social_losses["loss"]
            )

            # 从当前社交状态开始ODE积分
            next_S = self.flow_process.p_sample_tf(
                short_term_model_fn,
                shape_ref_tensor=curr_S,
                steps=self.flow_steps,
                condition=next_I,
                x_start=curr_S
            )

            curr_I = next_I
            curr_S = next_S

        # =====================================================
        # D. Short-term Preference Fusion
        # =====================================================

        self.short_term_vector = self._gated_fusion(
            curr_I,
            curr_S,
            "short_term_fusion"
        )

        self.short_term_item_flow_loss = tf.reduce_mean(
            tf.stack(
                item_flow_loss_list,
                axis=0
            )
        )

        self.short_term_social_flow_loss = tf.reduce_mean(
            tf.stack(
                social_flow_loss_list,
                axis=0
            )
        )

        # =====================================================
        # E. Final User Preference
        # =====================================================

        self.final_user_preference = (
            tf.tile(
                self.global_embedding,
                [batch_size, 1]
            )
            + self.final_long_term_user_embs
            + self.short_term_vector
        )

        # 正负样本物品嵌入
        pos_item_embs = tf.nn.embedding_lookup(
            item_gcn_embs,
            self.item_pos_ph
        )

        neg_item_embs = tf.nn.embedding_lookup(
            item_gcn_embs,
            self.item_neg_ph
        )

        pos_item_bias = tf.nn.embedding_lookup(
            self.item_biases,
            self.item_pos_ph
        )

        neg_item_bias = tf.nn.embedding_lookup(
            self.item_biases,
            self.item_neg_ph
        )

        # 正样本得分
        pos_scores = tf.reduce_sum(
            tf.multiply(
                self.final_user_preference,
                pos_item_embs
            ),
            axis=1
        )

        # 负样本得分
        neg_scores = tf.reduce_sum(
            tf.multiply(
                self.final_user_preference,
                neg_item_embs
            ),
            axis=1
        )

        self.output = (
            pos_scores + pos_item_bias
        )

        self.output_neg = (
            neg_scores + neg_item_bias
        )

        self.pred_item_gcn_embs = item_gcn_embs

    def _create_loss(self):

        self._create_inference()

        # BPR Loss
        self.bpr_loss = -tf.reduce_sum(
            tf.log_sigmoid(
                self.output - self.output_neg
            )
        )

        # L2 Regularization
        self.l2_loss = self.l2_reg * (
            tf.nn.l2_loss(
                self.embeddings["user_embeddings"]
            )
            + tf.nn.l2_loss(
                self.embeddings["item_embeddings"]
            )
            + tf.nn.l2_loss(
                self.embeddings["social_init_embeddings"]
            )
            + tf.nn.l2_loss(
                self.embeddings["seq_item_embeddings"]
            )
        )

        # 总损失
        self.loss = (
            self.bpr_loss
            + self.long_term_flow_loss_weight *
            self.long_term_flow_loss
            + self.flow_loss_weight *
            self.short_term_item_flow_loss
            + self.social_flow_loss_weight *
            self.short_term_social_flow_loss
            + self.l2_loss
        )

    def _create_optimizer(self):

        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.lr
        )

        train_op = optimizer.minimize(
            self.loss
        )

        # EMA
        self.ema = tf.train.ExponentialMovingAverage(
            decay=self.ema_decay
        )

        with tf.control_dependencies([train_op]):

            self.train_opt = self.ema.apply(
                self.ema_vars
            )

    def build_graph(self):

        self._create_placeholder()

        self._create_variable()

        self._create_loss()

        # 在预测网络实际建立后收集EMA变量
        trainable = tf.trainable_variables()

        ema_scopes = (
            "long_term_predictor",
            "short_term_predictor"
        )

        self.ema_vars = [
            var
            for var in trainable
            if any(
                scope in var.name
                for scope in ema_scopes
            )
        ]

        if not self.ema_vars:
            raise RuntimeError(
                "No flow predictor variables found for EMA."
            )

        self._create_optimizer()

        # EMA赋值操作
        self.ema_assign_op = tf.group(
            [
                tf.assign(
                    var,
                    self.ema.average(var)
                )
                for var in self.ema_vars
            ]
        )

        # 备份和恢复原始权重
        self.restore_placeholders = []
        self.restore_ops = []

        for var in self.ema_vars:

            ph = tf.placeholder(
                var.dtype,
                shape=var.get_shape()
            )

            self.restore_placeholders.append(
                ph
            )

            self.restore_ops.append(
                tf.assign(var, ph)
            )

        self.restore_group_op = tf.group(
            self.restore_ops
        )

        # 全物品预测
        all_scores = tf.matmul(
            self.final_user_preference,
            self.pred_item_gcn_embs,
            transpose_b=True
        )

        self.prediction = (
            all_scores + self.item_biases
        )

    def train_model(self):

        self.logger.info(
            self.evaluator.metrics_info()
        )

        self.user_pos_train = csr_to_user_dict_bytime(
            self.dataset.time_matrix,
            self.dataset.train_matrix
        )

        users_list, item_seq_list, item_pos_list = (
            self._generate_sequences()
        )

        for epoch in range(self.epochs):

            item_neg_list = self._sample_negative(
                users_list
            )

            data = DataIterator(
                users_list,
                item_seq_list,
                item_pos_list,
                item_neg_list,
                batch_size=self.batch_size,
                shuffle=True
            )

            for (
                bat_user,
                bat_item_seq,
                bat_item_pos,
                bat_item_neg
            ) in data:

                feed = {
                    self.user_ph: bat_user,
                    self.item_seq_ph: bat_item_seq,
                    self.item_pos_ph: bat_item_pos,
                    self.item_neg_ph: bat_item_neg,
                    self.is_training_ph: True
                }

                self.sess.run(
                    self.train_opt,
                    feed_dict=feed
                )

            result = self.evaluate_model()

            self.logger.info(
                "epoch %d:\t%s" % (epoch, result)
            )

    def _generate_sequences(self):

        self.user_test_seq = {}

        user_list = []
        item_seq_list = []
        item_pos_list = []

        for user_id in range(self.num_users):

            seq_items = self.user_pos_train.get(
                user_id, []
            )

            if len(seq_items) < 2:
                continue

            for index_id in range(
                1, len(seq_items)
            ):

                content_data = seq_items[
                    max(0, index_id - self.seq_l):index_id
                ]

                if len(content_data) < self.seq_l:

                    content_data = (
                        [self.num_items] *
                        (self.seq_l - len(content_data))
                        + content_data
                    )

                user_list.append(user_id)

                item_seq_list.append(
                    content_data
                )

                item_pos_list.append(
                    seq_items[index_id]
                )

            user_id_seq = seq_items[
                -self.seq_l:
            ]

            if len(user_id_seq) < self.seq_l:

                user_id_seq = (
                    [self.num_items] *
                    (self.seq_l - len(user_id_seq))
                    + user_id_seq
                )

            self.user_test_seq[user_id] = (
                user_id_seq
            )

        return (
            user_list,
            item_seq_list,
            item_pos_list
        )

    def _sample_negative(self, users_list):

        neg_items_list = []
        user_neg_items_dict = {}

        all_uni_user, all_counts = np.unique(
            users_list,
            return_counts=True
        )

        user_count = DataIterator(
            all_uni_user,
            all_counts,
            batch_size=1024,
            shuffle=False
        )

        for bat_users, bat_counts in user_count:

            n_neg_items = [
                c for c in bat_counts
            ]

            exclusion = [
                self.user_pos_train[u]
                for u in bat_users
            ]

            bat_neg = batch_randint_choice(
                self.num_items,
                n_neg_items,
                replace=True,
                exclusion=exclusion
            )

            for u, neg in zip(
                bat_users, bat_neg
            ):

                user_neg_items_dict[u] = neg

        for u, c in zip(
            all_uni_user, all_counts
        ):

            neg_items = np.reshape(
                user_neg_items_dict[u],
                newshape=[c, ]
            )

            neg_items_list.extend(
                neg_items
            )

        return neg_items_list

    def evaluate_model(self):

        return self.evaluator.evaluate(
            self
        )

    def predict(self, users):

        last_items = [
            self.user_test_seq.get(
                u,
                [self.num_items] * self.seq_l
            )
            for u in users
        ]

        # 1. 备份原始权重
        backup_vars_values = self.sess.run(
            self.ema_vars
        )

        # 2. 加载EMA权重
        self.sess.run(
            self.ema_assign_op
        )

        # 3. 预测
        feed = {
            self.user_ph: users,
            self.item_seq_ph: last_items,
            self.is_training_ph: False
        }

        bat_ratings = self.sess.run(
            self.prediction,
            feed_dict=feed
        )

        # 4. 恢复原始权重
        restore_feed = {
            ph: val
            for ph, val in zip(
                self.restore_placeholders,
                backup_vars_values
            )
        }

        self.sess.run(
            self.restore_group_op,
            feed_dict=restore_feed
        )

        return bat_ratings
