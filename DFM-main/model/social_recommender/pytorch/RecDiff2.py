import os
import random
import time

import numpy as np
from torch.utils.data import DataLoader, Dataset
import scipy.sparse as sp
import torch
import dgl
import torch.nn as nn
from torch.nn.init import xavier_uniform_
import dgl.function as fn
import torch.nn.functional as F
import math
from tqdm import tqdm
from model.base.abstract_recommender import SocialAbstractRecommender


# 随机数种子
def fix_random_seed_as(seed):
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class NeuRecInteractionDataset(Dataset):
    def __init__(self, interaction, num_items, num_negatives=1):
        """
        Custom Dataset for NeuRec's Interaction object with negative sampling.

        Args:
            interaction: NeuRec Interaction object containing user-item pairs.
            num_items: Total number of items for negative sampling.
            num_negatives: Number of negative samples per positive sample.
        """
        self.user_item_pairs = interaction.to_user_item_pairs()
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.interaction_matrix = interaction.to_dok_matrix()  # 使用 DOK 矩阵便于快速查找
        self.rows = self.user_item_pairs[:, 0]  # 用户索引
        self.cols = self.user_item_pairs[:, 1]  # 正样本项目索引
        self.negs = np.zeros(len(self.rows), dtype=np.int32)  # 负样本初始化
        self.negSampling()  # 初始化时生成负样本

    def negSampling(self):
        """
        Generate negative samples for each positive sample.
        Ensures negative items are not in the user's interaction history.
        """
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                iNeg = np.random.randint(self.num_items)
                if (u, iNeg) not in self.interaction_matrix:
                    break
            self.negs[i] = iNeg

    def __len__(self):
        return len(self.user_item_pairs)

    def __getitem__(self, idx):
        user = self.rows[idx]
        pos_item = self.cols[idx]
        neg_item = self.negs[idx]
        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(neg_item, dtype=torch.long)
        )


class UUGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, weight=False, bias=False, activation=None):
        super(UUGCNLayer, self).__init__()
        self.bias = bias
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.weight = weight
        if self.weight:
            self.u_w = nn.Parameter(torch.Tensor(in_feats, out_feats))
            xavier_uniform_(self.u_w)
        self._activation = activation

    def forward(self, graph, u_f):
        with graph.local_scope():
            if self.weight:
                u_f = torch.mm(u_f, self.u_w)
            node_f = u_f
            degs = graph.out_degrees().to(u_f.device).float().clamp(min=1)
            norm = torch.pow(degs, -0.5).view(-1, 1)
            node_f = node_f * norm

            graph.ndata['n_f'] = node_f
            graph.update_all(fn.copy_u(u='n_f', out='m'), reduce_func=fn.sum(msg='m', out='n_f'))

            rst = graph.ndata['n_f']

            degs = graph.in_degrees().to(u_f.device).float().clamp(min=1)
            norm = torch.pow(degs, -0.5).view(-1, 1)
            rst = rst * norm

            if self._activation is not None:
                rst = self._activation(rst)

            return rst


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, weight=False, bias=False, activation=None):
        super(GCNLayer, self).__init__()
        self.bias = bias
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.weight = weight
        if self.weight:
            self.u_w = nn.Parameter(torch.Tensor(in_feats, out_feats))
            self.v_w = nn.Parameter(torch.Tensor(in_feats, out_feats))
            xavier_uniform_(self.u_w)
            xavier_uniform_(self.v_w)
        self._activation = activation

    def forward(self, graph, u_f, v_f):
        with graph.local_scope():
            if self.weight:
                u_f = torch.mm(u_f, self.u_w)
                v_f = torch.mm(v_f, self.v_w)
            node_f = torch.cat([u_f, v_f], dim=0)
            degs = graph.out_degrees().to(u_f.device).float().clamp(min=1)
            norm = torch.pow(degs, -0.5).view(-1, 1)
            node_f = node_f * norm

            graph.ndata['n_f'] = node_f
            graph.update_all(fn.copy_u(u='n_f', out='m'), reduce_func=fn.sum(msg='m', out='n_f'))

            rst = graph.ndata['n_f']

            degs = graph.in_degrees().to(u_f.device).float().clamp(min=1)
            norm = torch.pow(degs, -0.5).view(-1, 1)
            rst = rst * norm

            if self._activation is not None:
                rst = self._activation(rst)

            return rst


class GCNModel(nn.Module):
    def __init__(self, config, num_user, num_item):
        super(GCNModel, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.n_hid = config["n_hid"]
        self.n_layers = config["n_layers"]
        self.s_layers = config["s_layers"]
        self.embedding_dict = self.init_weight(num_user, num_item, self.n_hid)
        self.act = nn.LeakyReLU(0.5, inplace=True)
        self.layers = nn.ModuleList()
        self.uu_Layers = nn.ModuleList()
        self.weight = config["weight"]
        for i in range(0, self.n_layers):
            self.layers.append(GCNLayer(self.n_hid, self.n_hid, weight=self.weight, bias=False, activation=self.act))
        for i in range(0, self.s_layers):
            self.uu_Layers.append(
                UUGCNLayer(self.n_hid, self.n_hid, weight=self.weight, bias=False, activation=self.act))

    def init_weight(self, userNum, itemNum, hide_dim):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(userNum, hide_dim))),
            'item_emb': nn.Parameter(initializer(torch.empty(itemNum, hide_dim))),
        })
        return embedding_dict

    def forward(self, uigraph, uugraph, isTrain=True):
        init_embedding = torch.concat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], axis=0)
        init_user_embedding = self.embedding_dict['user_emb']
        all_embeddings = [init_embedding]
        all_uu_embeddings = [init_user_embedding]

        for i, layer in enumerate(self.layers):
            if i == 0:
                embeddings = layer(uigraph, self.embedding_dict['user_emb'], self.embedding_dict['item_emb'])
            else:
                embeddings = layer(uigraph, embeddings[:self.num_user], embeddings[self.num_user:])
            norm_embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings += [norm_embeddings]
        ui_embeddings = sum(all_embeddings)

        for i, layer in enumerate(self.uu_Layers):
            if i == 0:
                embeddings = layer(uugraph, self.embedding_dict['user_emb'])
            else:
                embeddings = layer(uugraph, embeddings)
            norm_embeddings = F.normalize(embeddings, p=2, dim=1)
            all_uu_embeddings += [norm_embeddings]
        uu_embeddings = sum(all_uu_embeddings)

        return ui_embeddings, uu_embeddings


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class SDNet(nn.Module):
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5):
        super(SDNet, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims

        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out)
                                        for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out)
                                         for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for layer in self.in_layers:
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.out_layers:
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x, timesteps):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)
        h = torch.cat([x, emb], dim=-1)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)

        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)

        return h


class DiffusionProcess(nn.Module):
    def __init__(self, noise_schedule, noise_scale, noise_min, noise_max, steps, device, keep_num=10):
        super(DiffusionProcess, self).__init__()
        self.noise_schedule = noise_schedule
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps
        self.device = device

        self.keep_num = keep_num
        self.Lt_record = torch.zeros(steps, keep_num, dtype=torch.float64).to(device)
        self.Lt_count = torch.zeros(steps, dtype=int).to(device)

        self.beta_nums = torch.tensor(self.betas_num(), dtype=torch.float64).to(self.device)
        assert len(self.beta_nums.shape) == 1, "betas must be 1-D"
        assert len(self.beta_nums) == self.steps, "num of betas must equal to diffusion steps"
        assert (self.beta_nums > 0).all() and (self.beta_nums <= 1).all(), "betas out of range"

        self.diffusion_setting()

    def betas_num(self):
        st_bound = self.noise_scale * self.noise_min
        e_bound = self.noise_scale * self.noise_max
        if self.noise_schedule == "linear":
            return np.linspace(st_bound, e_bound, self.steps, dtype=np.float64)
        else:
            return betas_from_linear_variance(self.steps, np.linspace(st_bound, e_bound, self.steps, dtype=np.float64))

    def diffusion_setting(self):
        alphas = 1.0 - self.beta_nums
        self.alphas_cumprod = torch.cumprod(alphas, axis=0).to(self.device)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(self.device), self.alphas_cumprod[:-1]]).to(
            self.device)
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).to(self.device)]).to(
            self.device)
        assert self.alphas_cumprod_prev.shape == (self.steps,)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
                self.beta_nums * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
                self.beta_nums * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * torch.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
        )

    def calculate_losses(self, model, emb_s, reweight=False):
        batch_size, device = emb_s.size(0), emb_s.device
        ts, pt = self.sample_timesteps(batch_size, device, 'uniform')
        noise = torch.randn_like(emb_s)
        emb_t = self.forward_process(emb_s, ts, noise)
        terms = {}
        model_output = model(emb_t, ts)

        assert model_output.shape == emb_s.shape

        mse = mean_flat((emb_s - model_output) ** 2)

        if reweight:
            weight = self.SNR(ts - 1) - self.SNR(ts)
            weight = torch.where((ts == 0), 1.0, weight)
            loss = mse
        else:
            weight = torch.tensor([1.0] * len(model_output)).to(device)

        terms["loss"] = weight * loss
        terms["pred_xstart"] = model_output
        return terms

    def p_sample(self, model, emb_s, steps, sampling_noise=False):
        assert steps <= self.steps, "Too much steps in inference."
        if steps == 0:
            emb_t = emb_s
        else:
            t = torch.tensor([steps - 1] * emb_s.shape[0]).to(emb_s.device)
            emb_t = self.q_sample(emb_s, t)

        indices = list(range(self.steps))[::-1]

        if self.noise_scale == 0.:
            for i in indices:
                t = torch.tensor([i] * emb_t.shape[0]).to(emb_s.device)
                emb_t = model(emb_t, t)
            return emb_t

        for i in indices:
            t = torch.tensor([i] * emb_t.shape[0]).to(emb_s.device)
            out = self.p_mean_variance(model, emb_t, t)
            if sampling_noise:
                noise = torch.randn_like(emb_t)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(emb_t.shape) - 1)))
                )
                emb_t = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
            else:
                emb_t = out["mean"]
        return emb_t

    def sample_timesteps(self, batch_size, device, method='uniform', uniform_prob=0.001):
        if method == 'importance':
            if not (self.Lt_count == self.keep_num).all():
                return self.sample_timesteps(batch_size, device, method='uniform')

            Lt_sqrt = torch.sqrt(torch.mean(self.Lt_record ** 2, axis=-1))
            pt_all = Lt_sqrt / torch.sum(Lt_sqrt)
            pt_all *= 1 - uniform_prob
            pt_all += uniform_prob / len(pt_all)

            assert pt_all.sum(-1) - 1. < 1e-5

            t = torch.multinomial(pt_all, num_samples=batch_size, replacement=True)
            pt = pt_all.gather(dim=0, index=t) * len(pt_all)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.steps, (batch_size,), device=device).long()
            pt = torch.ones_like(t).float()

            return t, pt

        else:
            raise ValueError

    def forward_process(self, emb_s, t, noise=None):
        if noise is None:
            noise = torch.randn_like(emb_s)
        assert noise.shape == emb_s.shape
        return (
                self._extract_into_tensor(self.sqrt_alphas_cumprod, t, emb_s.shape) * emb_s
                + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, emb_s.shape)
                * noise
        )

    def q_posterior_mean_variance(self, emb_s, emb_t, t):
        assert emb_s.shape == emb_t.shape
        posterior_mean = (
                self._extract_into_tensor(self.posterior_mean_coef1, t, emb_t.shape) * emb_s
                + self._extract_into_tensor(self.posterior_mean_coef2, t, emb_t.shape) * emb_t
        )
        posterior_variance = self._extract_into_tensor(self.posterior_variance, t, emb_t.shape)
        posterior_log_variance_clipped = self._extract_into_tensor(
            self.posterior_log_variance_clipped, t, emb_s.shape
        )
        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == emb_s.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t):
        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, t)

        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped

        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)
        pred_xstart = model_output

        model_mean, _, _ = self.q_posterior_mean_variance(emb_s=pred_xstart, emb_t=x, t=t)

        assert (
                model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, emb_t, t, eps):
        assert emb_t.shape == eps.shape
        return (
                self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, emb_t.shape) * emb_t
                - self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, emb_t.shape) * eps
        )

    def SNR(self, t):
        self.alphas_cumprod = self.alphas_cumprod.to(t.device)
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        arr = arr.to(timesteps.device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)


def betas_from_linear_variance(steps, variance, max_beta=0.999):
    alpha_bar = 1 - variance
    betas = []
    betas.append(1 - alpha_bar[0])
    for i in range(1, steps):
        betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
    return np.array(betas)


def innerProduct(usrEmbeds, itmEmbeds):
    return torch.sum(usrEmbeds * itmEmbeds, dim=-1)


def pairPredict(ancEmbeds, posEmbeds, negEmbeds):
    return innerProduct(ancEmbeds, posEmbeds) - innerProduct(ancEmbeds, negEmbeds)


class RecDiff2(SocialAbstractRecommender):
    def __init__(self, config):
        super(RecDiff2, self).__init__(config)
        self.sampling_steps = config["sampling_steps"]
        self.sampling_noise = config["sampling_noise"]
        fix_random_seed_as(config["seed"])
        self.reweight = config["reweight"]
        self.reg = config["reg"]
        self.batch_size = config["batch_size"]
        self.epochs = config["epoch"]
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items

        self.uuGraph = self._create_social_adj_mat()
        self.uiGraph = self._create_adj_mat()

        self.GCNModel = GCNModel(config, self.num_users, self.num_items).to(self.device)

        output_dims = [config["dims"]] + [config["n_hid"]]
        input_dims = output_dims[::-1]
        self.SDNet = SDNet(
            input_dims,
            output_dims,
            config["emb_size"],
            time_type=config["time_type"],
            norm=config["norm"],
            dropout=config["dropout"]
        ).to(self.device)

        self.DiffProcess = DiffusionProcess(config["noise_schedule"], config["noise_scale"], config["noise_min"],
                                            config["noise_max"],
                                            config["steps"], self.device).to(self.device)

        self.optimizer1 = torch.optim.Adam([
            {'params': self.GCNModel.parameters(), 'weight_decay': 0},
        ], lr=config["lr"])
        self.optimizer2 = torch.optim.Adam([
            {'params': self.SDNet.parameters(), 'weight_decay': 0},
        ], lr=config["difflr"])
        self.scheduler1 = torch.optim.lr_scheduler.StepLR(
            self.optimizer1,
            step_size=config["decay_step"],
            gamma=config["decay"]
        )
        self.scheduler2 = torch.optim.lr_scheduler.StepLR(
            self.optimizer2,
            step_size=config["decay_step"],
            gamma=config["decay"]
        )

    def _create_social_adj_mat(self):
        uu_idx = [[ui, uj] for (ui, uj), r in self.social_matrix.todok().items()]
        if not uu_idx:
            # Handle empty social matrix
            u1_idx = torch.tensor([], dtype=torch.int32)
            u2_idx = torch.tensor([], dtype=torch.int32)
        else:
            u1_idx, u2_idx = list(zip(*uu_idx))
            u1_idx = np.array(u1_idx, dtype=np.int32)
            u2_idx = np.array(u2_idx, dtype=np.int32)
            u1_idx = torch.tensor(u1_idx, dtype=torch.int32)
            u2_idx = torch.tensor(u2_idx, dtype=torch.int32)
        graph = dgl.graph((u1_idx, u2_idx), num_nodes=self.num_users)
        graph = dgl.add_self_loop(graph)
        return graph.to(self.device)

    def _create_adj_mat(self):
        users_items = self.dataset.train_data.to_user_item_pairs()
        if users_items is None:
            users_np = torch.tensor([], dtype=torch.int32)
            items_np = torch.tensor([], dtype=torch.int32)
        else:
            users_np, items_np = users_items[:, 0], users_items[:, 1]
            items_np = items_np + self.num_users
            users_np = torch.tensor(users_np, dtype=torch.int32)
            items_np = torch.tensor(items_np, dtype=torch.int32)
        graph = dgl.graph((users_np, items_np), num_nodes=self.num_users + self.num_items)
        graph = dgl.add_reverse_edges(graph)
        graph = dgl.add_self_loop(graph)
        return graph.to(self.device)

    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        training_start_time = time.time()
        train_dataset = NeuRecInteractionDataset(
            interaction=self.dataset.train_data,
            num_items=self.num_items,
            num_negatives=1
        )
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        for epoch in range(self.epochs):
            self.SDNet.train()
            self.GCNModel.train()
            epoch_losses = [0.0, 0.0, 0.0]
            train_dataset.negSampling()  # 在每个 epoch 开始时重新生成负样本
            tqdm_dataloader = tqdm(dataloader, desc=f"Epoch {epoch}")

            for iteration, batch in enumerate(tqdm_dataloader):
                user_idx, pos_idx, neg_idx = batch
                user_idx = user_idx.long().to(self.device)
                pos_idx = pos_idx.long().to(self.device)
                neg_idx = neg_idx.long().to(self.device)

                uiEmbeds, uuEmbeds = self.GCNModel(self.uiGraph, self.uuGraph, True)
                uEmbeds = uiEmbeds[:self.num_users]
                iEmbeds = uiEmbeds[self.num_users:]
                user = uEmbeds[user_idx]
                pos = iEmbeds[pos_idx]
                neg = iEmbeds[neg_idx]

                uu_terms = self.DiffProcess.calculate_losses(self.SDNet, uuEmbeds[user_idx], self.reweight)
                uuelbo = uu_terms["loss"].mean()
                user = user + uu_terms["pred_xstart"]
                diffloss = uuelbo
                scoreDiff = pairPredict(user, pos, neg)
                bprLoss = - (scoreDiff).sigmoid().log().mean()
                regLoss = ((torch.norm(user) ** 2 + torch.norm(pos) ** 2 + torch.norm(neg) ** 2) * self.reg)
                loss = bprLoss + regLoss
                losses = [bprLoss.item(), regLoss.item(), diffloss.item()]

                loss = diffloss + loss
                self.optimizer1.zero_grad()
                self.optimizer2.zero_grad()
                loss.backward()
                self.optimizer1.step()
                self.optimizer2.step()

                epoch_losses = [x + y for x, y in zip(epoch_losses, losses)]

            result = self.evaluate_model()
            # self.logger.info(f"epoch {epoch}:\t{result}")
            self.logger.info("[iter %d:time:%f]" % (epoch, time.time() - training_start_time))

            self.scheduler1.step()
            self.scheduler2.step()

            # total_loss = sum(epoch_losses) / len(dataloader)
            # self.logger.info(f"Epoch {epoch}: Total Loss: {total_loss:.6f}")

    def evaluate_model(self):
        self.GCNModel.eval()
        self.SDNet.eval()
        return self.evaluator.evaluate(self)

    def predict(self, users):
        users = torch.from_numpy(np.asarray(users)).long().to(self.device)
        with torch.no_grad():
            uiEmbeds, uuEmbeds = self.GCNModel(self.uiGraph, self.uuGraph, isTrain=False)
            uEmbeds = uiEmbeds[:self.num_users]
            iEmbeds = uiEmbeds[self.num_users:]
            user_embeddings = uEmbeds[users]
            uuemb = uuEmbeds[users]
            user_predict = self.DiffProcess.p_sample(self.SDNet, uuemb, self.sampling_steps,
                                                     self.sampling_noise)
            user_embeddings = user_embeddings + user_predict
            scores = torch.mm(user_embeddings, iEmbeds.t())
        return scores.cpu().detach().numpy()