import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sp_linalg
from model.base.abstract_recommender import SocialAbstractRecommender
from data.data_iterator import DataIterator
from reckit import timer
from util.common.tool import csr_to_user_dict_bytime
import tensorflow as tf
import tensorflow.compat.v1 as tf
from util.common.tool import batch_randint_choice
from reckit import pad_sequences
from util.tensorflow import get_session

epsilon = 1e-9


# def l2_distance(a, b, name="euclidean_distance"):
#     return tf.norm(a - b, ord='euclidean', axis=-1, name=name)


def mean_flat_tf(tensor):
    """
    TensorFlow版本的mean_flat。
    计算一个张量除了第一个维度（批次维度）外所有其他维度的均值。
    """
    return tf.reduce_mean(tensor, axis=list(range(1, len(tensor.shape.as_list()))))


class FlowMatchingProcessTF:
    """
    flow_process.py 的 TensorFlow 1.x 等价实现。
    这个类封装了流匹配的算法逻辑，其方法被设计为在主模型的图构建阶段调用。
    """

    def __init__(self, steps: int, t_dist: str = "cosine", eps: float = 1e-4):
        """
        初始化函数，设置超参数。
        注意：device参数在TF1.x中通常在构建图时通过 tf.device() 控制，这里不需要传入。
        """
        self.steps = steps
        self.t_dist = t_dist
        self.eps = eps
        self.default_steps_fallback = 80
        self.default_anchor_ratio = 0.35
        self.delta_scale_infer = 0.50
        self.norm_out_infer = True
        self.dir_align_infer = 0.30
        self.default_integrator = "heun"
        self.delta_scale_train = 0.50
        self.norm_out_train = True
        # self.dir_align_train = 0.25,0.5(不好），0.1(还可以）
        self.dir_align_train = 0.2


    def sample_times_tf(self, batch_size_tensor, method: str = None):
        """
        TensorFlow版本的时间采样函数。
        batch_size_tensor: 一个形状为()的int32张量, 通常是 tf.shape(input)[0]。
        """
        method = method or self.t_dist
        if method == "uniform":
            t = tf.random.uniform([batch_size_tensor], minval=0.0, maxval=1.0, dtype=tf.float32)
        elif method == "cosine":
            u = tf.random.uniform([batch_size_tensor], minval=self.eps, maxval=1.0 - self.eps, dtype=tf.float32)
            t = tf.acos(1 - 2 * u) / math.pi
        elif method == "beta22":
            u1 = tf.random.uniform([batch_size_tensor], minval=0.0, maxval=1.0, dtype=tf.float32)
            u2 = tf.random.uniform([batch_size_tensor], minval=0.0, maxval=1.0, dtype=tf.float32)
            t = tf.clip_by_value(tf.minimum(u1, u2), self.eps, 1.0 - self.eps)
        else:
            raise ValueError(f"Unknown t sampling: {method}")
        return t

    def _expand_t_tf(self, t, like_tensor):
        """TensorFlow版本的_expand_t，用于调整时间张量t的形状以匹配数据张量。"""
        # 获取目标形状的维度数量
        like_dims = like_tensor.shape.as_list()
        # 将t从 [B] 扩展为 [B, 1, 1, ...] 以便广播
        while len(t.shape.as_list()) < len(like_dims):
            t = tf.expand_dims(t, -1)
        return t

    def path_xt_tf(self, x0, x1, t):
        """TensorFlow版本的路径定义函数。"""
        t_ = self._expand_t_tf(t, x1)
        return (1. - t_) * x0 + t_ * x1

    def target_velocity_tf(self, x0, x1):
        """TensorFlow版本的目标速度计算。"""
        return x1 - x0

    def _project_on_tf(self, u, v):
        """TensorFlow版本的向量投影函数。"""
        # 沿着最后一个维度计算点积和范数
        axis = len(u.shape.as_list()) - 1
        # 计算点积 (u * v)
        uv_dot = tf.reduce_sum(u * v, axis=axis, keepdims=True)
        # 计算v的范数的平方 ||v||^2
        v_norm_sq = tf.reduce_sum(tf.square(v), axis=axis, keepdims=True) + 1e-8
        # 计算投影系数
        coeff = uv_dot / v_norm_sq
        return coeff * v

    def caculate_losses_tf(self, model_fn, emb_s, reweight: bool = True, condition=None):
        """
        【已修改】TensorFlow版本的损失计算函数，新增可选的 condition 参数。
        model_fn: 一个函数，接收(x_t, t, condition)作为输入，返回预测的速度v_pred。
        emb_s: 真实数据的张量 (即 x1)。
        condition: (可选) 条件向量。
        """
        batch_size = tf.shape(emb_s)[0]

        t = self.sample_times_tf(batch_size)
        x0 = tf.random.normal(tf.shape(emb_s), dtype=tf.float32)
        x_t = self.path_xt_tf(x0, emb_s, t)
        v_tgt = self.target_velocity_tf(x0, emb_s)

        # 【修改】调用外部模型函数时，传入 condition
        v_pred = model_fn(x_t, t, condition)

        mse = mean_flat_tf(tf.square(v_pred - v_tgt))
        if reweight:
            w = tf.clip_by_value(t * (1. - t), self.eps, 1.0)
        else:
            w = tf.ones_like(t)
        loss = w * mse

        # 训练时的后处理
        x1_hat = x0 + v_pred
        if self.dir_align_train > 0:
            x1_proj = self._project_on_tf(x1_hat, emb_s)
            x1_hat = self.dir_align_train * x1_proj + (1.0 - self.dir_align_train) * x1_hat
        if self.norm_out_train:
            x1_hat = tf.nn.l2_normalize(x1_hat, axis=-1) * (tf.norm(emb_s, axis=-1, keepdims=True) + 1e-8)
        if self.delta_scale_train != 1.0:
            x1_hat = self.delta_scale_train * x1_hat

        return {"loss": loss, "pred_xstart": x1_hat}

    def p_sample_tf(self, model_fn, emb_s, steps, sampling_noise: bool = False,
                    anchor=None, anchor_ratio: float = None, integrator: str = None, condition=None):
        """
        【已修改】TensorFlow版本的标准生成/采样函数，新增可选的 condition 参数。
        """
        integrator = integrator or self.default_integrator

        if steps is None or steps <= 0:
            steps = max(int(getattr(self, "steps", 20)), int(self.default_steps_fallback))

        if anchor is None:
            anchor = emb_s
        if anchor_ratio is None:
            anchor_ratio = self.default_anchor_ratio

        batch_size = tf.shape(emb_s)[0]
        x0 = tf.random.normal(tf.shape(emb_s))
        alpha = float(anchor_ratio)
        x = (1.0 - alpha) * x0 + alpha * anchor

        ts = tf.linspace(0., 1., steps + 1)

        if integrator == "heun":
            for i in range(steps):
                t_i_scalar = ts[i]
                t_n_scalar = ts[i + 1]
                t_i = tf.fill([batch_size], t_i_scalar)
                t_n = tf.fill([batch_size], t_n_scalar)
                dt = (t_n_scalar - t_i_scalar)

                # 【修改】传入 condition
                v_i = model_fn(x, t_i, condition)
                x_p = x + dt * v_i
                # 【修改】传入 condition
                v_n = model_fn(x_p, t_n, condition)
                x = x + 0.5 * dt * (v_i + v_n)

        elif integrator == "rk4":
            for i in range(steps):
                t_i_scalar = ts[i]
                t_n_scalar = ts[i + 1]
                t_i = tf.fill([batch_size], t_i_scalar)
                t_n = tf.fill([batch_size], t_n_scalar)
                dt = (t_n_scalar - t_i_scalar)

                # 【修改】所有 model_fn 调用都传入 condition
                k1 = model_fn(x, t_i, condition)
                k2 = model_fn(x + 0.5 * dt * k1, t_i + 0.5 * (t_n_scalar - t_i_scalar), condition)
                k3 = model_fn(x + 0.5 * dt * k2, t_i + 0.5 * (t_n_scalar - t_i_scalar), condition)
                k4 = model_fn(x + dt * k3, t_n, condition)
                x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError(f"Unknown integrator: {integrator}")

        # 推理时的后处理
        if self.dir_align_infer > 0:
            x_proj = self._project_on_tf(x, anchor)
            x = self.dir_align_infer * x_proj + (1.0 - self.dir_align_infer) * x
        if self.norm_out_infer:
            x = tf.nn.l2_normalize(x, axis=-1) * (tf.norm(anchor, axis=-1, keepdims=True) + 1e-8)
        if self.delta_scale_infer != 1.0:
            x = self.delta_scale_infer * x
        return x

    def forward_process_tf(self, emb_s, t, noise=None):
        """TensorFlow版本的加噪过程"""
        if noise is None:
            noise = tf.random.normal(tf.shape(emb_s))
        return self.path_xt_tf(noise, emb_s, t)


def timestep_embedding_tf(timesteps, dim, max_period=10000):
    """
    TensorFlow版本的正弦时间步长嵌入函数。
    这是PyTorch版本 timestep_embedding 的直接翻译。

    参数:
    - timesteps: 一个一维张量，包含N个时间步长（可以是浮点数）。
    - dim: 输出嵌入的维度。
    - max_period: 控制嵌入的最小频率。
    返回: 一个形状为 [N, dim] 的位置嵌入张量。
    """
    half = dim // 2
    # 计算频率
    freqs = tf.exp(
        -math.log(max_period) * tf.range(start=0, limit=half, dtype=tf.float32) / half
    )
    # 计算参数
    args = tf.expand_dims(tf.cast(timesteps, tf.float32), 1) * tf.expand_dims(freqs, 0)
    # 拼接sin和cos部分
    embedding = tf.concat([tf.cos(args), tf.sin(args)], axis=-1)
    if dim % 2:
        # 如果维度是奇数，补一个零
        embedding = tf.concat([embedding, tf.zeros_like(embedding[:, :1])], axis=-1)
    return embedding


class SDNetTF:
    """
    通用的SDNet实现，【更新】：可以处理可选的条件输入 (condition)。
    如果提供了 condition，则使用 AdaLN 进行调制。
    如果没有提供 condition，则作为标准的无条件网络运行。
    """

    def __init__(self, name, hidden_dims, emb_size, norm=False, dropout_rate=0.5):
        self.name = name
        self.hidden_dims = hidden_dims
        self.time_emb_dim = emb_size
        self.norm = norm
        self.dropout_rate = dropout_rate

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            # --- 所有带权重的层都在这里预先定义好 ---
            self.emb_layer = tf.keras.layers.Dense(units=self.time_emb_dim, name="time_emb_layer")
            self.dropout_layer = tf.keras.layers.Dropout(rate=self.dropout_rate)

            # 预定义隐藏层
            self.hidden_layers = []
            for i, h_dim in enumerate(hidden_dims[:-1]):
                self.hidden_layers.append(
                    tf.keras.layers.Dense(units=h_dim, activation='tanh', name=f"hidden_layer_{i}")
                )

            # 预定义输出层
            self.output_layer = tf.keras.layers.Dense(units=hidden_dims[-1], activation=None, name="output_layer")

            # 预定义AdaLN的gamma和beta生成器层
            self.adaLN_generators = []
            for i, h_dim in enumerate(hidden_dims[:-1]):
                self.adaLN_generators.append(
                    tf.keras.layers.Dense(units=h_dim * 2, name=f"adaln_gen_{i}")
                )

            # 【关键修复】为每个需要应用AdaLN的隐藏层预定义一个LayerNormalization层
            self.adaln_layer_norms = []
            for i, _ in enumerate(hidden_dims[:-1]):
                self.adaln_layer_norms.append(
                    tf.keras.layers.LayerNormalization(axis=-1, center=False, scale=False, name=f"adaln_ln_{i}")
                )

    def __call__(self, x, timesteps, is_training=True, condition=None):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            time_emb = timestep_embedding_tf(timesteps, self.time_emb_dim)
            emb = self.emb_layer(time_emb)

            h = tf.concat([x, emb], axis=-1)

            if condition is not None:
                # --- 有条件流程 (AdaLN) ---
                merged_condition = emb + condition

                for i, layer in enumerate(self.hidden_layers):
                    style_params = self.adaLN_generators[i](merged_condition)
                    gamma, beta = tf.split(style_params, 2, axis=-1)

                    # 【关键修复】调用在__init__中预定义好的LayerNormalization层
                    h_normalized = self.adaln_layer_norms[i](h)

                    h = gamma * h_normalized + beta
                    h = layer(h)
                    h = self.dropout_layer(h, training=is_training)
            else:
                # --- 无条件流程 (标准MLP) ---
                for layer in self.hidden_layers:
                    h = layer(h)
                    h = self.dropout_layer(h, training=is_training)

            output = self.output_layer(h)
            return output

class DFM(SocialAbstractRecommender):
    def __init__(self, conf):
        super(DFM, self).__init__(conf)
        self.num_users, self.num_items = self.dataset.train_matrix.shape
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

        self.long_term_flow_loss_weight = conf["long_term_flow_loss_weight"]
        self.flow_loss_weight = conf["flow_loss_weight"]
        self.social_flow_loss_weight = conf["social_flow_loss_weight"]

        self.fusion_scale = conf["fusion_scale"]
        # self.sigma_min = conf["sigma_min"]
        self.flow_steps = conf["flow_steps"]
        self.dropout_rate = conf["dropout_rate"]

        self.reweight = conf["reweight"]

        # --- 新增EMA超参数 ---
        # 从配置文件中读取EMA的衰减率，如果未提供，则默认为0.999
        self.ema_decay = conf["ema_decay"]

        self.sess = get_session(conf["gpu_mem"])
        self.build_graph()
        self.sess.run(tf.global_variables_initializer())

    def _get_SocialDict(self):
        SocialDict = {}
        for u in range(self.num_users):
            trustors = self.social_matrix[u].indices
            SocialDict[u] = trustors.tolist() if len(trustors) > 0 else []
        return SocialDict

    @timer
    def create_adj_mat(self):
        n_nodes = self.num_users + self.num_items
        user_list, item_list, _ = self.dataset.get_train_interactionssecond()
        user_np = np.array(user_list, dtype=np.int32)
        item_np = np.array(item_list, dtype=np.int32)
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_inv[np.isnan(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        print('use the pre adjacency matrix for user-item graph')
        return adj_matrix

    def _create_social_adj_mat(self):
        uu_idx = [[ui, uj] for (ui, uj), r in self.social_matrix.todok().items()]
        u1_idx, u2_idx = list(zip(*uu_idx))
        self.u1_idx = tf.constant(u1_idx, dtype=tf.int32, shape=None, name="u1_idx")
        self.u2_idx = tf.constant(u2_idx, dtype=tf.int32, shape=None, name="u2_idx")
        u1_idx = np.array(u1_idx, dtype=np.int32)
        u2_idx = np.array(u2_idx, dtype=np.int32)
        ratings = np.ones_like(u1_idx, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (u1_idx, u2_idx)), shape=(self.num_users, self.num_users))
        adj_mat = tmp_adj + tmp_adj.T
        return self._normalize_spmat(adj_mat)

    def _normalize_spmat(self, adj_mat):
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_inv[np.isnan(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        print('use the pre adjacency matrix for social graph')
        return adj_matrix

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _lightgcn_propagate(self, norm_adj, embeddings):
        """通用的LightGCN传播函数"""
        all_embeddings = [embeddings]
        for k in range(self.n_layers):
            embeddings = tf.sparse.sparse_dense_matmul(norm_adj, embeddings)
            all_embeddings.append(embeddings)
        return tf.reduce_mean(tf.stack(all_embeddings, axis=1), axis=1)

    def _create_placeholder(self):
        self.user_ph = tf.placeholder(tf.int32, [None], name="user")
        self.item_seq_ph = tf.placeholder(tf.int32, [None, self.seq_l], name="item_seq")
        self.social_user_input = tf.placeholder(tf.int32, [None, None], name="social_user_input")
        self.item_pos_ph = tf.placeholder(tf.int32, [None], name="item_pos")
        self.item_neg_ph = tf.placeholder(tf.int32, [None], name="item_neg")
        # 添加一个用于控制dropout的占位符
        self.is_training_ph = tf.placeholder_with_default(True, shape=(), name="is_training")

        # --- 新增EMA占位符 ---
        # 这个占位符用于在预测时，将备份的原始权重加载回模型
        self.restore_vars_ph = tf.placeholder(tf.float32, shape=[None, None])

    def _create_variable(self):
        self.embeddings = dict()
        # initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        initializer = tf.glorot_uniform_initializer()  # 尝试改为这个
        self.embeddings["user_embeddings"] = tf.Variable(initializer([self.num_users, self.embedding_size]))
        self.embeddings["item_embeddings"] = tf.Variable(initializer([self.num_items, self.embedding_size]))
        self.embeddings["social_init_embeddings"] = tf.Variable(initializer([self.num_users, self.embedding_size]))
        self.embeddings["seq_item_embeddings"] = tf.Variable(initializer([self.num_items + 1, self.embedding_size]))
        self.embeddings["position_embeddings"] = tf.Variable(initializer([self.seq_l, self.embedding_size]))
        self.global_embedding = tf.Variable(initializer([1, self.embedding_size]))
        self.item_biases = tf.Variable(tf.zeros([self.num_items]))

        # --- 【核心】在这里创建所有需要的【实例】 ---
        # 1. 长期模块实例
        self.flow_process = FlowMatchingProcessTF(steps=self.flow_steps, t_dist="cosine")
        long_term_hidden_dims = [self.embedding_size * 2, self.embedding_size]
        self.long_term_predictor = SDNetTF(name="long_term_predictor", hidden_dims=long_term_hidden_dims,
                                           emb_size=self.embedding_size, dropout_rate=self.dropout_rate)

        # # 2. 短期模块实例
        # self.short_term_flow_process = ShortTermConditionalFlowProcessTF(flow_steps=self.flow_steps,
        #                                                                  sigma_min=self.sigma_min)
        short_term_hidden_dims = [self.embedding_size * 2, self.embedding_size]
        self.short_term_predictor = SDNetTF(name="short_term_predictor", hidden_dims=short_term_hidden_dims,
                                            emb_size=self.embedding_size, dropout_rate=self.dropout_rate)
        # 【新增】在这里预定义短期偏好模块中用于计算注意力的Dense层
        self.short_term_attention_dense = tf.keras.layers.Dense(1, activation=None, name="short_term_attention")

        # --- 新增EMA相关 ---
        # 识别需要进行EMA平滑的变量：即两个predictor网络中的所有可训练参数
        self.ema_vars = (tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="long_term_predictor") +
                         tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="short_term_predictor"))

    def _create_inference(self):
        batch_size = tf.shape(self.user_ph)[0]

        # --- 步骤 0: 计算所有GCN基础嵌入 ---
        ui_adj_tensor = self._convert_sp_mat_to_sp_tensor(self.norm_adj)
        social_adj_tensor = self._convert_sp_mat_to_sp_tensor(self.social_norm_adj)

        ui_embs_initial = tf.concat([self.embeddings["user_embeddings"], self.embeddings["item_embeddings"]], axis=0)
        gcn_ui_embs = self._lightgcn_propagate(ui_adj_tensor, ui_embs_initial)
        user_gcn_embs, item_gcn_embs = tf.split(gcn_ui_embs, [self.num_users, self.num_items], 0)

        social_gcn_embs = self._lightgcn_propagate(social_adj_tensor, self.embeddings["social_init_embeddings"])  #[n,D]
        # print("social_gcn_embs shape:", social_gcn_embs)  #[B,D]

        social_gcn_embs_padded = tf.concat([social_gcn_embs, tf.zeros([1, self.embedding_size])], 0)  #[n+1,D]
        # print("social_gcn_embs_padded shape:", social_gcn_embs_padded)

        # --- 模块 A：长期偏好 (LightGCN + 无条件Flow Matching增强) ---
        batch_user_gcn_embs = tf.nn.embedding_lookup(user_gcn_embs, self.user_ph) #二维[B,D]
        # print("batch_user_gcn_embs shape:", batch_user_gcn_embs)

        batch_social_friends_gcn = tf.nn.embedding_lookup(social_gcn_embs_padded, self.social_user_input) #[B,S,D]
        # print("batch_social_friends_gcn shape:", batch_social_friends_gcn)


        batch_social_gcn_embs = tf.reduce_mean(batch_social_friends_gcn, axis=1)

        # 新增一个参数 cond 来接收传入的 condition
        long_term_model_fn = lambda x_t, t, cond: self.long_term_predictor(x_t, t, self.is_training_ph, condition=cond)
        long_term_outputs = self.flow_process.caculate_losses_tf(long_term_model_fn, batch_social_gcn_embs,
                                                                           self.reweight)

        self.long_term_flow_loss = tf.reduce_mean(long_term_outputs["loss"])
        self.final_long_term_user_embs = batch_user_gcn_embs + self.fusion_scale * long_term_outputs["pred_xstart"]



        # B.1. 聚合序列信息 (保持不变)
        seq_item_embs = tf.nn.embedding_lookup(self.embeddings["seq_item_embeddings"], self.item_seq_ph)
        seq_item_embs_with_pos = seq_item_embs + self.embeddings["position_embeddings"]
        # 【修改】调用在_create_variable中预定义好的层，而不是在这里重新创建
        attn_scores = self.short_term_attention_dense(seq_item_embs_with_pos)

        attn_weights = tf.nn.softmax(attn_scores, axis=1)
        seq_embs_agg = tf.reduce_sum(seq_item_embs_with_pos * attn_weights, axis=1)

        # B.2. 定义短期的流匹配函数 (保持不变)
        def short_term_model_fn(x_t, t_float, condition):
            return self.short_term_predictor(x_t, t_float, self.is_training_ph, condition=condition)

        item_flow_loss_list = []
        social_flow_loss_list = []
        tar_item_emb_pos_gcn = tf.nn.embedding_lookup(item_gcn_embs, self.item_pos_ph)

        # --- 初始迭代 (严格按照 目标 和 条件 的逻辑) ---

        # 1. 物品层生成与损失计算
        # 生成 item_level
        # emb_s: 在生成时，我们没有真实目标，因此使用条件本身作为“锚点”是合理的做法
        # condition: 使用序列信息作为生成条件
        item_level = self.flow_process.p_sample_tf(
            short_term_model_fn,
            emb_s=seq_embs_agg,
            steps=self.flow_steps,
            condition=seq_embs_agg
        )
        # 计算 item_level 的损失
        # emb_s: 训练时的真实目标，即用户交互过的正样本物品
        # condition: 使用序列信息作为条件
        item_loss_outputs = self.flow_process.caculate_losses_tf(
            short_term_model_fn,
            emb_s=tar_item_emb_pos_gcn,
            reweight=self.reweight,
            condition=seq_embs_agg
        )
        item_flow_loss_list.append(tf.reduce_mean(item_loss_outputs["loss"]))

        # 2. 社交层生成与损失计算
        item_level_condition = tf.stop_gradient(item_level)
        # 生成 social_level
        # emb_s: 使用上一层的输出 item_level 作为“锚点”
        # condition: 使用上一层的输出 item_level 作为生成条件
        social_level = self.flow_process.p_sample_tf(
            short_term_model_fn,
            emb_s=item_level_condition,
            steps=self.flow_steps,
            condition=item_level_condition
        )
        # 计算 social_level 的损失
        # emb_s: 训练时的真实目标，即用户的社交网络表征
        # condition: 使用上一层的输出 item_level 作为条件
        social_loss_outputs = self.flow_process.caculate_losses_tf(
            short_term_model_fn,
            emb_s=batch_social_gcn_embs,
            reweight=self.reweight,
            condition=item_level_condition
        )
        social_flow_loss_list.append(tf.reduce_mean(social_loss_outputs["loss"]))

        # --- for循环迭代 (同样严格遵循 目标 和 条件 的逻辑) ---
        for _ in range(2):
            # 3. 基于社交层结果，再次优化物品层
            social_level_condition = tf.stop_gradient(social_level)
            item_level = self.flow_process.p_sample_tf(
                short_term_model_fn,
                emb_s=social_level_condition,
                steps=self.flow_steps,
                condition=social_level_condition
            )
            item_loss_outputs = self.flow_process.caculate_losses_tf(
                short_term_model_fn,
                emb_s=tar_item_emb_pos_gcn,
                reweight=self.reweight,
                condition=social_level_condition
            )
            item_flow_loss_list.append(tf.reduce_mean(item_loss_outputs["loss"]))

            # 4. 基于新的物品层结果，再次优化社交层
            item_level_condition = tf.stop_gradient(item_level)
            social_level = self.flow_process.p_sample_tf(
                short_term_model_fn,
                emb_s=item_level_condition,
                steps=self.flow_steps,
                condition=item_level_condition
            )
            social_loss_outputs = self.flow_process.caculate_losses_tf(
                short_term_model_fn,
                emb_s=batch_social_gcn_embs,
                reweight=self.reweight,
                condition=item_level_condition
            )
            social_flow_loss_list.append(tf.reduce_mean(social_loss_outputs["loss"]))

        # B.4. 最终的短期向量和平均损失
        self.short_term_vector = item_level + social_level
        self.short_term_item_flow_loss = tf.reduce_mean(item_flow_loss_list)
        self.short_term_social_flow_loss = tf.reduce_mean(social_flow_loss_list)

        # --- 模块 C：融合与预测 (保持不变) ---
        self.final_user_preference = (tf.tile(self.global_embedding, [batch_size, 1]) +
                                      self.final_long_term_user_embs +
                                      self.short_term_vector)

        pos_item_embs = tf.nn.embedding_lookup(item_gcn_embs, self.item_pos_ph)
        neg_item_embs = tf.nn.embedding_lookup(item_gcn_embs, self.item_neg_ph)
        pos_item_bias = tf.nn.embedding_lookup(self.item_biases, self.item_pos_ph)
        neg_item_bias = tf.nn.embedding_lookup(self.item_biases, self.item_neg_ph)

        # self.output = pos_item_bias - tf.reduce_sum(tf.square(self.final_user_preference - pos_item_embs), 1)
        # self.output_neg = neg_item_bias - tf.reduce_sum(tf.square(self.final_user_preference - neg_item_embs), 1)
        # 计算最终用户偏好与正样本物品嵌入的内积
        pos_scores = tf.reduce_sum(tf.multiply(self.final_user_preference, pos_item_embs), axis=1)
        # 计算最终用户偏好与负样本物品嵌入的内积
        neg_scores = tf.reduce_sum(tf.multiply(self.final_user_preference, neg_item_embs), axis=1)

        # 将内积得分与物品偏置项相加，得到最终的正负样本预测得分
        self.output = pos_scores + pos_item_bias
        self.output_neg = neg_scores + neg_item_bias

        self.pred_item_gcn_embs = item_gcn_embs

    def _create_loss(self):
        self._create_inference()
        self.bpr_loss = -tf.reduce_sum(tf.log_sigmoid(self.output - self.output_neg))
        self.l2_loss = self.l2_reg * (tf.nn.l2_loss(self.embeddings["user_embeddings"]) +
                                      tf.nn.l2_loss(self.embeddings["item_embeddings"]) +
                                      tf.nn.l2_loss(self.embeddings["social_init_embeddings"]) +
                                      tf.nn.l2_loss(self.embeddings["seq_item_embeddings"]))

        self.loss = (self.bpr_loss +
                     self.long_term_flow_loss_weight * self.long_term_flow_loss +
                     self.flow_loss_weight * self.short_term_item_flow_loss +
                     self.social_flow_loss_weight * self.short_term_social_flow_loss +
                     self.l2_loss)

    def _create_optimizer(self):
        # --- 新增并修改优化器以包含EMA ---
        # 1. 创建Adam优化器
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        # 2. 创建一个操作，用于计算梯度并更新原始模型变量
        train_op = optimizer.minimize(self.loss)

        # 3. 创建EMA对象
        self.ema = tf.train.ExponentialMovingAverage(decay=self.ema_decay)

        # 4. 创建一个操作，用于在模型变量更新后，更新EMA的影子权重
        #    我们只对 predictor 网络的变量进行平滑
        with tf.control_dependencies([train_op]):
            self.train_opt = self.ema.apply(self.ema_vars)

    def build_graph(self):
        self._create_placeholder()
        self._create_variable()
        self._create_loss()
        self._create_optimizer()

        # --- 新增EMA相关操作 ---
        # 创建一个操作，用于将EMA的影子权重加载（赋值）到原始模型变量中
        # 这个操作将在预测时被调用
        self.ema_assign_op = tf.group([tf.assign(var, self.ema.average(var)) for var in self.ema_vars])

        # 创建一个操作，用于将外部传入的（备份的）权重值加载回模型变量
        # 这个操作用于在预测后恢复原始权重
        self.restore_vars_ops = [tf.assign(var, self.restore_vars_ph) for var in self.ema_vars]

        # 最终结果 shape: [batch_size, num_items]
        all_scores = tf.matmul(self.final_user_preference, self.pred_item_gcn_embs, transpose_b=True)

        # 加上物品的偏置项
        self.prediction = all_scores + self.item_biases

    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        self.user_pos_train = csr_to_user_dict_bytime(self.dataset.time_matrix, self.dataset.train_matrix)
        users_list, item_seq_list, item_pos_list, user_social_list = self._generate_sequences()
        for epoch in range(self.epochs):
            item_neg_list = self._sample_negative(users_list)
            data = DataIterator(users_list, item_seq_list, item_pos_list, item_neg_list, user_social_list,
                                batch_size=self.batch_size, shuffle=True)
            for bat_user, bat_item_seq, bat_item_pos, bat_item_neg, bat_user_social in data:
                bat_user_social = pad_sequences(bat_user_social, value=self.num_users)
                feed = {self.user_ph: bat_user,
                        self.item_seq_ph: bat_item_seq,
                        self.item_pos_ph: bat_item_pos,
                        self.item_neg_ph: bat_item_neg,
                        self.social_user_input: bat_user_social,
                        self.is_training_ph: True}
                self.sess.run(self.train_opt, feed_dict=feed)
            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))

    def _generate_sequences(self):
        self.user_test_seq = {}
        user_list, item_seq_list, item_pos_list, user_social_list = [], [], [], []
        social_counts = [len(self.SocialtrainDict[u]) for u in range(self.num_users)]
        max_social = int(np.percentile(social_counts, 97))
        print(f"Dynamic max_social: {max_social}")
        for user_id in range(self.num_users):
            seq_items = self.user_pos_train.get(user_id, [])
            if len(seq_items) < 2: continue

            social_friends = self.SocialtrainDict.get(user_id, [])[:max_social]
            for index_id in range(1, len(seq_items)):
                content_data = seq_items[max(0, index_id - self.seq_l):index_id]
                if len(content_data) < self.seq_l:
                    content_data = [self.num_items] * (self.seq_l - len(content_data)) + content_data
                user_list.append(user_id)
                item_seq_list.append(content_data)
                item_pos_list.append(seq_items[index_id])
                user_social_list.append(social_friends)
            user_id_seq = seq_items[-self.seq_l:]
            if len(user_id_seq) < self.seq_l:
                user_id_seq = [self.num_items] * (self.seq_l - len(user_id_seq)) + user_id_seq
            self.user_test_seq[user_id] = user_id_seq
        return user_list, item_seq_list, item_pos_list, user_social_list

    def _sample_negative(self, users_list):
        neg_items_list = []
        user_neg_items_dict = {}
        all_uni_user, all_counts = np.unique(users_list, return_counts=True)
        user_count = DataIterator(all_uni_user, all_counts, batch_size=1024, shuffle=False)
        for bat_users, bat_counts in user_count:
            n_neg_items = [c for c in bat_counts]
            exclusion = [self.user_pos_train[u] for u in bat_users]
            bat_neg = batch_randint_choice(self.num_items, n_neg_items, replace=True, exclusion=exclusion)
            for u, neg in zip(bat_users, bat_neg):
                user_neg_items_dict[u] = neg
        for u, c in zip(all_uni_user, all_counts):
            neg_items = np.reshape(user_neg_items_dict[u], newshape=[c, ])
            neg_items_list.extend(neg_items)
        return neg_items_list

    def evaluate_model(self):
        return self.evaluator.evaluate(self)

    def predict(self, users):
        # --- 修改predict函数以实现EMA权重的加载与恢复 ---
        last_items = [self.user_test_seq.get(u, [self.num_items] * self.seq_l) for u in users]
        bat_social = [self.SocialtrainDict.get(u, []) for u in users]
        bat_social = pad_sequences(bat_social, value=self.num_users)

        # 1. 备份原始的模型权重
        #    我们只备份那些受EMA影响的变量
        backup_vars_values = self.sess.run(self.ema_vars)

        # 2. 将EMA的平滑权重加载到模型中
        self.sess.run(self.ema_assign_op)

        # 3. 使用平滑后的模型进行预测
        feed = {
            self.user_ph: users,
            self.item_seq_ph: last_items,
            self.social_user_input: bat_social,
            self.is_training_ph: False
        }
        bat_ratings = self.sess.run(self.prediction, feed_dict=feed)

        # 4. 恢复原始的模型权重，以确保下一次训练不受影响
        for var, value in zip(self.ema_vars, backup_vars_values):
            self.sess.run(tf.assign(var, value))

        return bat_ratings
