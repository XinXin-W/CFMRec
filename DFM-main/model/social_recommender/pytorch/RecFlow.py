import shutil
import sys
import copy
import time

import torch as th
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import math
from torch.nn.init import xavier_normal_, constant_, xavier_uniform_
import dgl
import torch, os, random
import numpy as np
from torch.utils.data import DataLoader, Dataset
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

# 这个类定义了在用户-用户社交图上进行图卷积的单层操作。
class UUGCNLayer(nn.Module):
    '''
    __init__: 初始化函数。
    in_feats, out_feats: 输入和输出特征的维度。
    weight=False: 一个布尔值，决定是否要添加一个可学习的线性变换层（权重矩阵）。
    nn.Parameter(...): 将一个张量（Tensor）注册为模型的可学习参数。
    xavier_uniform_(self.u_w): 使用 Xavier 方法初始化权重矩阵，这是一种常用的权重初始化策略，有助于防止梯度消失或爆炸。'''

    def __init__(self,
                 in_feats,
                 out_feats,
                 weight=False,
                 bias=False,
                 activation=None):
        super(UUGCNLayer, self).__init__()
        self.bias = bias
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.weight = weight
        if self.weight:
            self.u_w = nn.Parameter(torch.Tensor(in_feats, out_feats))
            xavier_uniform_(self.u_w)
        self._activation = activation

    '''
    forward: 定义前向传播的逻辑。
    with graph.local_scope(): DGL 的一个特性，确保在此代码块内对图的所有操作（如添加节点/边特征）都是临时的，不会影响到原始图对象。'''

    # def forward(self, graph, feat):
    def forward(self, graph, u_f):
        with graph.local_scope():
            # 如果设置了 weight=True，就对输入的特征 u_f (user features) 进行一次线性变换 (u_f @ u_w)。
            if self.weight:
                u_f = torch.mm(u_f, self.u_w)
            node_f = u_f
            # D^-1/2
            # degs = graph.out_degrees().to(feat.device).float().clamp(min=1)
            '''
            GCN归一化: 这是标准GCN的核心步骤。
            graph.out_degrees(): 计算图中每个节点的出度（发出的边的数量）。
            .clamp(min=1): 确保度数最小为1，防止除以零。
            torch.pow(degs, -0.5): 计算度数的-0.5次方，即归一化因子 D^{-1/2}。'''
            degs = graph.out_degrees().to(u_f.device).float().clamp(min=1)
            norm = torch.pow(degs, -0.5).view(-1, 1)
            # norm = norm.view(-1,1)
            # shp = norm.shape + (1,) * (feat.dim() - 1)
            # norm = t.reshape(norm, shp)

            # 将归一化因子应用到节点特征上。这是公式 D^{-1/2} * X 的一部分。
            node_f = node_f * norm

            '''
            消息传递与聚合:
            graph.ndata['n_f'] = node_f: 将处理后的节点特征存入图的节点数据存储区，命名为 n_f。
            graph.update_all(...): DGL的核心函数，触发消息传递。
            fn.copy_u(u='n_f', out='m'): 消息函数，将每个源节点的 'n_f' 特征复制到边上，作为消息 'm'。
            reduce_func=fn.sum(msg='m', out='n_f'): 聚合函数，对所有发往目标节点的消息 'm' 进行求和，并将结果更新回目标节点的 'n_f' 特征。
            这一步执行了 GCN 公式中的 A * X 部分，即聚合邻居信息。'''
            graph.ndata['n_f'] = node_f
            # graph.edata['e_f'] = e_f
            graph.update_all(fn.copy_u(u='n_f', out='m'), reduce_func=fn.sum(msg='m', out='n_f'))

            rst = graph.ndata['n_f']

            # 再次使用入度（in-degrees）进行归一化。结合前面的出度归一化，完整实现了对称归一化 D^{-1/2} * A * D^{-1/2} * X。
            degs = graph.in_degrees().to(u_f.device).float().clamp(min=1)
            norm = torch.pow(degs, -0.5).view(-1, 1)
            # shp = norm.shape + (1,) * (feat.dim() - 1)
            # norm = t.reshape(norm, shp)
            rst = rst * norm

            '''
            如果定义了激活函数，则应用它。
            return rst: 返回这一层GCN计算后的节点表示。'''
            if self._activation is not None:
                rst = self._activation(rst)

            return rst


'''
GCNLayer 类 (用户-物品GCN层)
这个类的逻辑与 UUGCNLayer 基本完全相同，但它是为二部图（用户和物品）设计的。

主要区别:
__init__: 它为用户 (u_w) 和物品 (v_w) 定义了两套独立的权重矩阵。
forward: 它接收用户特征 u_f 和物品特征 v_f 作为两个独立的输入。
node_f = torch.cat([u_f, v_f], dim=0): 在进行消息传递之前，它将变换后的用户和物品特征拼接成一个大的张量，以匹配DGL图中节点（用户节点+物品节点）的顺序。
之后的归一化和消息传递步骤与 UUGCNLayer 完全一样。'''


class GCNLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 weight=False,
                 bias=False,
                 activation=None):
        super(GCNLayer, self).__init__()
        self.bias = bias
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.weight = weight
        if self.weight:
            self.u_w = nn.Parameter(torch.Tensor(in_feats, out_feats))
            self.v_w = nn.Parameter(torch.Tensor(in_feats, out_feats))
            # self.e_w = nn.Parameter(t.Tensor(in_feats, out_feats))
            xavier_uniform_(self.u_w)
            xavier_uniform_(self.v_w)
            # init.xavier_uniform_(self.e_w)
        self._activation = activation

    # def forward(self, graph, feat):
    def forward(self, graph, u_f, v_f):
        with graph.local_scope():
            if self.weight:
                u_f = torch.mm(u_f, self.u_w)
                v_f = torch.mm(v_f, self.v_w)
                # e_f = t.mm(e_f, self.e_w)
            node_f = torch.cat([u_f, v_f], dim=0)
            # D^-1/2
            # degs = graph.out_degrees().to(feat.device).float().clamp(min=1)
            degs = graph.out_degrees().to(u_f.device).float().clamp(min=1)
            norm = torch.pow(degs, -0.5).view(-1, 1)
            # norm = norm.view(-1,1)
            # shp = norm.shape + (1,) * (feat.dim() - 1)
            # norm = t.reshape(norm, shp)

            node_f = node_f * norm

            graph.ndata['n_f'] = node_f
            # graph.edata['e_f'] = e_f
            graph.update_all(fn.copy_u(u='n_f', out='m'), reduce_func=fn.sum(msg='m', out='n_f'))

            rst = graph.ndata['n_f']

            degs = graph.in_degrees().to(u_f.device).float().clamp(min=1)
            norm = torch.pow(degs, -0.5).view(-1, 1)
            # shp = norm.shape + (1,) * (feat.dim() - 1)
            # norm = t.reshape(norm, shp)
            rst = rst * norm

            if self._activation is not None:
                rst = self._activation(rst)

            return rst


'''
GCNModel 类 (主GCN模型)
这个类将前面定义的GCN层堆叠起来，构建一个完整的图神经网络。'''


class GCNModel(nn.Module):
    '''
    __init__:
    self.embedding_dict: 调用init_weight方法创建可学习的用户和物品的初始嵌入矩阵。
    self.layers = nn.ModuleList(): 创建一个列表来存放 n_layers 个 GCNLayer（用于用户-物品图）。
    self.uu_Layers = nn.ModuleList(): 创建另一个列表来存放 s_layers 个 UUGCNLayer（用于用户-用户图）。
    '''

    def __init__(self, config, n_user, n_item):
        super(GCNModel, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.n_hid = config["n_hid"]
        self.n_layers = config["n_layers"]
        self.s_layers = config["s_layers"]
        self.embedding_dict = self.init_weight(n_user, n_item, self.n_hid)
        self.act = nn.LeakyReLU(0.5, inplace=True)
        self.layers = nn.ModuleList()
        self.uu_Layers = nn.ModuleList()
        self.weight = config["weight"]
        for i in range(0, self.n_layers):
            self.layers.append(GCNLayer(self.n_hid, self.n_hid, weight=self.weight, bias=False, activation=self.act))
        for i in range(0, self.s_layers):
            self.uu_Layers.append(
                UUGCNLayer(self.n_hid, self.n_hid, weight=self.weight, bias=False, activation=self.act))

    '''
    init_weight:
    nn.ParameterDict: 使用一个字典来管理参数，这样可以通过键名（如 'user_emb'）来方便地访问它们。
    nn.Parameter(initializer(torch.empty(...))): 创建指定大小的空张量，并使用Xavier方法进行初始化，然后将其注册为可学习参数。'''

    def init_weight(self, userNum, itemNum, hide_dim):
        initializer = nn.init.xavier_uniform_

        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(userNum, hide_dim))),
            'item_emb': nn.Parameter(initializer(torch.empty(itemNum, hide_dim))),
        })
        return embedding_dict

    '''
    forward: 定义模型的完整前向计算流程。
    all_embeddings, all_uu_embeddings: 创建列表，用于存储每一层GCN的输出。列表的第一个元素是初始嵌入。'''

    def forward(self, uigraph, uugraph, isTrain=True):

        init_embedding = torch.concat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], axis=0)
        init_user_embedding = self.embedding_dict['user_emb']
        all_embeddings = [init_embedding]
        all_uu_embeddings = [init_user_embedding]

        '''
        用户-物品图处理:
        循环遍历 self.layers 中的每一个GCNLayer。
        将上一层的输出embeddings作为当前层的输入，进行图卷积。
        F.normalize(...): 对每层的输出进行L2归一化。
        all_embeddings += [...]: 将当前层的输出添加到列表中。
        ui_embeddings = sum(all_embeddings): 层聚合。将初始嵌入和所有GCN层的输出相加。这是一种类似于 LightGCN 的残差连接（skip connection）方式，可以融合不同深度的信息并缓解过平滑问题。'''
        for i, layer in enumerate(self.layers):
            if i == 0:
                embeddings = layer(uigraph, self.embedding_dict['user_emb'], self.embedding_dict['item_emb'])
            else:
                embeddings = layer(uigraph, embeddings[:self.n_user], embeddings[self.n_user:])

            norm_embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings += [norm_embeddings]
        ui_embeddings = sum(all_embeddings)

        '''
        用户-用户图处理: 对用户-用户图执行完全相同的堆叠、归一化和层聚合操作。
        return ui_embeddings, uu_embeddings: 返回两种最终的嵌入表示。ui_embeddings包含了用户和物品的表示，而uu_embeddings只包含用户的社交表示。'''
        for i, layer in enumerate(self.uu_Layers):
            if i == 0:
                embeddings = layer(uugraph, self.embedding_dict['user_emb'])
            else:
                embeddings = layer(uugraph, embeddings)
            norm_embeddings = F.normalize(embeddings, p=2, dim=1)
            all_uu_embeddings += [norm_embeddings]
        uu_embeddings = sum(all_uu_embeddings)

        return ui_embeddings, uu_embeddings

'''
SDNet 类 (生成模型的核心网络)
这个网络不是GCN，而是一个普通的多层感知机（MLP），用于流匹配过程。'''
# Social hidden fully-connected architecture
class SDNet(nn.Module):
    """
    A deep neural network for the reverse diffusion preocess.
    """
    '''
    __init__:
    in_dims, out_dims: 定义了网络各层的维度，构成一个类似编码器-解码器的结构。
    time_emb_dim: 时间编码向量的维度。
    self.emb_layer: 一个线性层，用于对时间编码向量进行初步处理。
    time_type="cat": 指定时间编码的处理方式是“拼接”（concatenate）。
    in_dims_temp = [...]: 计算拼接后的第一层输入维度（原始输入维度 + 时间编码维度）。
    self.in_layers, self.out_layers: 定义了网络的主体，由一系列线性层组成。'''

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

        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
                                        for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
                                         for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for layer in self.in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)

        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    '''
    forward:
    x: 输入的特征向量（例如，一个用户的社交嵌入）。
    timesteps: 时间步 t (一个标量)。
    timestep_embedding(...): 调用辅助函数将timesteps转换为高维向量time_emb。
    emb = self.emb_layer(time_emb): 对时间编码向量进行线性变换。'''

    def forward(self, x, timesteps):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)
        h = torch.cat([x, emb], dim=-1)  # 核心步骤: 将输入的特征x和处理后的时间编码emb在最后一个维度上拼接起来。这使得网络在处理x时能够感知到当前的时间t。

        '''
        将拼接后的向量h依次通过in_layers和out_layers，中间使用tanh作为激活函数。
        return h: 返回最终的输出，这个输出就是模型预测的“速度”向量。'''
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)

        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)

        return h


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

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
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

'''
这份代码主要实现了 FlowMatchingProcess 类，包含了三个核心功能：
定义路径和目标：定义了从噪声 x0 到数据 x1 的直线路径 x_t，以及这个路径上的目标速度。
计算训练损失：在训练阶段，随机采样一个时间点 t，计算出 x_t，让一个外部神经网络（代码中的model）预测该点的速度，并与真实速度进行比较，计算出损失。
生成样本（推理）：在推理阶段，从一个随机噪声点开始，使用数值积分方法（如 Heun 或 RK4）沿着神经网络预测的速度场“流动”，最终生成一个新样本。'''
# EMA（Exponential Moving Average，指数移动平均）是一个用于稳定模型训练的技巧。
class EMA:
    def __init__(self, model, decay=0.999):
        # decay: EMA的衰减率，一个接近1的数，控制平均的平滑程度。
        self.decay = decay
        # self.shadow: 创建一个“影子权重”字典。它通过复制model的当前所有参数（state_dict）来初始化。.detach().clone()确保了复制的是参数的值，而不是引用，且不带梯度。
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # @torch.no_grad(): 一个装饰器，表示在这个函数内不计算梯度，以节省计算资源。
    @torch.no_grad()
    def update(self, model):  # 在每次模型参数更新后调用此方法。
        for k, v in model.state_dict().items():  # 遍历模型当前最新的参数。
            self.shadow[k].mul_(self.decay).add_(v.detach(),
                                                 alpha=1 - self.decay)  # 这是EMA的核心更新公式：shadow_param = decay * shadow_param + (1 - decay) * new_param。它将影子权重乘以衰减率，然后加上新参数乘以(1 - decay)。

    @torch.no_grad()
    def apply_to(self, model):  # 在评估或测试时，用这个方法将平滑后的影子权重加载回模型中，以获得更稳定的预测结果。
        model.load_state_dict(self.shadow, strict=True)  # 将self.shadow字典中的参数值加载到model中。

class FlowMatchingProcess(nn.Module):

    def __init__(self, steps: int, device, t_dist: str = "cosine", eps: float = 1e-4):
        super().__init__()
        self.steps = steps
        self.device = device
        self.t_dist = t_dist  # 保存训练时采样时间t所用的分布方法（例如 "uniform", "cosine"）。
        self.eps = eps  # 保存一个非常小的数（epsilon），用于防止在计算中出现除以零或对数(0)等数值不稳定问题。

        self.default_steps_fallback = 80  # 如果推理时未指定步数，则使用这个默认值。
        self.default_anchor_ratio = 0.35  # “锚点”混合比例的默认值。在生成样本时，初始噪声会与一个“锚点”向量混合，这个比例控制了混合程度。

        self.delta_scale_infer = 0.50  # 对最终生成的样本进行缩放的系数。
        self.norm_out_infer = True  # 是否对最终输出进行归一化，使其长度（范数）与锚点向量一致。
        self.dir_align_infer = 0.30  # 方向对齐系数。将生成结果向“锚点”向量方向进行投影和混合，可以理解为一种引导。

        self.default_integrator = "heun"  # 默认的数值积分器，这里是 "heun"。数值积分器用于求解常微分方程（ODE）。

        self.delta_scale_train = 0.50
        self.norm_out_train = True
        self.dir_align_train = 0.25

    def sample_times(self, batch_size: int, device, method: str = None):
        method = method or self.t_dist
        if method == "uniform":
            t = th.rand(batch_size, device=device)
        elif method == "cosine":
            u = th.rand(batch_size, device=device).clamp_(self.eps, 1 - self.eps)
            t = th.arccos(1 - 2 * u) / math.pi
        elif method == "beta22":
            u1, u2 = th.rand(batch_size, device=device), th.rand(batch_size, device=device)
            t = th.minimum(u1, u2).clamp_(self.eps, 1 - self.eps)
        else:
            raise ValueError(f"Unknown t sampling: {method}")
        return t

    # @staticmethod: 表示这是一个静态方法，不依赖于类的实例。
    @staticmethod
    def _expand_t(t: th.Tensor, like: th.Tensor):
        while t.dim() < like.dim():
            t = t.view(-1, *([1] * (like.dim() - 1)))
        return t

    '''
    这是流匹配的核心之一。它定义了从噪声x0到数据x1的路径。
    t_ = self._expand_t(t, x1): 首先将时间张量t的形状调整为与数据x1兼容。
    return (1. - t_) * x0 + t_ * x1: 实现线性插值（直线路径）。当t=0时, x_t = x_0；当t=1时, x_t = x_1。'''

    def path_xt(self, x0: th.Tensor, x1: th.Tensor, t: th.Tensor):
        t_ = self._expand_t(t, x1)
        return (1. - t_) * x0 + t_ * x1

    @staticmethod
    def target_velocity(x0: th.Tensor, x1: th.Tensor):
        return x1 - x0

    '''
    这个静态方法计算向量u在向量v上的投影。
    v.norm(...): 计算向量v的长度（范数）。
    (u * v).sum(...): 计算向量u和v的点积。
    return coeff * v: 返回投影向量。这个函数用于dir_align（方向对齐）步骤。'''

    @staticmethod
    def _project_on(u: th.Tensor, v: th.Tensor):

        dim = u.dim() - 1
        denom = (v.norm(dim=dim, keepdim=True) ** 2 + 1e-8)
        coeff = (u * v).sum(dim=dim, keepdim=True) / denom
        return coeff * v

    # 这是模型训练时调用的核心函数。model是外部传入的神经网络，emb_s是输入的真实数据（也就是x1）。
    def caculate_losses(self, model, emb_s: th.Tensor, reweight: bool = True):
        device = emb_s.device
        B = emb_s.size(0)
        t = self.sample_times(B, device)
        x0 = th.randn_like(emb_s)  # 从标准正态分布中采样噪声x0，其形状与真实数据emb_s相同。
        x_t = self.path_xt(x0, emb_s, t)
        v_tgt = self.target_velocity(x0, emb_s)
        v_pred = model(x_t, t)
        assert v_pred.shape == emb_s.shape, "model must predict velocity with same shape as input"

        mse = mean_flat((v_pred - v_tgt) ** 2)  # 计算预测速度v_pred和目标速度v_tgt之间的均方误差（MSE）。
        w = (t * (1. - t)).clamp_min(self.eps) if reweight else th.ones_like(t)
        loss = w * mse
        '''
        训练时的后处理：
        x1_hat = x0 + v_pred: 根据模型预测的速度v_pred，估算出它认为的最终数据x1应该是什么样子。
        if self.dir_align_train > 0: 如果设置了方向对齐，就将x1_hat向真实数据emb_s的方向进行一定比例的混合。
        if self.norm_out_train: 如果设置了归一化，就将x1_hat的长度调整为与真实数据emb_s的长度一致。
        if self.delta_scale_train != 1.0: 如果设置了缩放，就对x1_hat进行缩放。'''
        x1_hat = x0 + v_pred
        if self.dir_align_train > 0:
            x1_proj = self._project_on(x1_hat, emb_s)
            x1_hat = self.dir_align_train * x1_proj + (1.0 - self.dir_align_train) * x1_hat
        if self.norm_out_train:
            x1_hat = F.normalize(x1_hat, dim=1) * (th.norm(emb_s, dim=1, keepdim=True) + 1e-8)
        if self.delta_scale_train != 1.0:
            x1_hat = self.delta_scale_train * x1_hat

        return {"loss": loss, "pred_xstart": x1_hat}

    '''
    @th.no_grad(): 装饰器，表示在推理过程中不计算梯度。
    x1_like: 提供一个形状参考张量。
    K: 评估点的数量。
    anchor: 锚点向量，用于引导生成。
    这是一个特殊的、非迭代的采样方法。它在K个不同的时间点t上进行多次评估，并引入了正负噪声对称性来做一种“集成”或“去噪”操作，最后将结果平均。'''

    @th.no_grad()
    def direct_estimate(self, model, x1_like, K: int = 8, anchor=None, anchor_ratio: float = None):
        device = x1_like.device
        if anchor is None:
            anchor = x1_like
        if anchor_ratio is None:
            anchor_ratio = self.default_anchor_ratio
        alpha = float(anchor_ratio)
        B = x1_like.size(0)

        # t 网格（避开0/1）
        K = int(abs(K))
        ts = th.linspace(self.eps, 1.0 - self.eps, K, device=device)

        acc = th.zeros_like(x1_like)
        for t in ts:
            t_vec = t.expand(B)
            x0 = th.randn_like(x1_like)
            x0a_pos = (1.0 - alpha) * x0 + alpha * anchor
            x0a_neg = (1.0 - alpha) * (-x0) + alpha * anchor

            x_t_pos = self.path_xt(x0a_pos, anchor, t_vec)
            x_t_neg = self.path_xt(x0a_neg, anchor, t_vec)

            v_pos = model(x_t_pos, t_vec)
            v_neg = model(x_t_neg, t_vec)

            x1_hat_pos = x0a_pos + v_pos
            x1_hat_neg = x0a_neg + v_neg

            x1_hat = 0.5 * (x1_hat_pos + x1_hat_neg)

            if self.dir_align_infer > 0:
                x1_proj = self._project_on(x1_hat, anchor)
                x1_hat = self.dir_align_infer * x1_proj + (1.0 - self.dir_align_infer) * x1_hat

            acc += x1_hat

        x = acc / float(K)

        if self.norm_out_infer:
            x = F.normalize(x, dim=1) * (th.norm(anchor, dim=1, keepdim=True) + 1e-8)
        if self.delta_scale_infer != 1.0:
            x = self.delta_scale_infer * x
        return x

    @th.no_grad()
    # 这是标准的生成样本函数。
    def p_sample(self, model, emb_s, steps, sampling_noise: bool = False,
                 anchor=None, anchor_ratio: float = None, integrator: str = None):
        if integrator is None:
            integrator = self.default_integrator

        # 一个巧妙的设计：如果传入的steps是负数，则调用direct_estimate方法。
        if steps is not None and steps < 0:
            return self.direct_estimate(model, emb_s, K=abs(int(steps)), anchor=anchor, anchor_ratio=anchor_ratio)

        steps = int(steps) if steps is not None else 0
        if steps <= 0:
            steps = max(int(getattr(self, "steps", 20)), int(self.default_steps_fallback))

        device = emb_s.device
        if anchor is None:
            anchor = emb_s
        if anchor_ratio is None:
            anchor_ratio = self.default_anchor_ratio

        x0 = th.randn_like(emb_s)
        alpha = float(anchor_ratio)
        x = (1.0 - alpha) * x0 + alpha * anchor

        ts = th.linspace(0., 1., steps + 1, device=device)

        '''
        Heun法积分器：这是一种二阶的数值积分方法（也叫改进欧拉法）。
        v_i = model(x, t_i): 在当前时间t_i用模型预测速度。
        x_p = x + dt * v_i: 用这个速度做一个初步的“预测”。
        v_n = model(x_p, t_n): 在下一个时间点t_n和预测位置x_p再次预测速度。
        x = x + 0.5 * dt * (v_i + v_n): 用两个速度的平均值来做最终的“校正”更新。这比只用一个速度更精确。'''
        if integrator == "heun":
            for i in range(steps):
                t_i = ts[i].expand(emb_s.size(0))
                t_n = ts[i + 1].expand(emb_s.size(0))
                dt = (t_n[0] - t_i[0]).item()
                v_i = model(x, t_i)
                x_p = x + dt * v_i
                v_n = model(x_p, t_n)
                x = x + 0.5 * dt * (v_i + v_n)

        # RK4 (四阶龙格-库塔) 法积分器：这是一个更精确但计算量更大的四阶数值积分方法，它通过在每个时间步内进行四次模型评估来获得更高精度的更新。
        elif integrator == "rk4":
            for i in range(steps):
                t_i = ts[i].expand(emb_s.size(0))
                t_n = ts[i + 1].expand(emb_s.size(0))
                dt = (t_n[0] - t_i[0]).item()
                k1 = model(x, t_i)
                k2 = model(x + 0.5 * dt * k1, t_i + 0.5 * (t_n - t_i))
                k3 = model(x + 0.5 * dt * k2, t_i + 0.5 * (t_n - t_i))
                k4 = model(x + dt * k3, t_n)
                x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError(f"Unknown integrator: {integrator}")

        if self.dir_align_infer > 0:
            x_proj = self._project_on(x, anchor)
            x = self.dir_align_infer * x_proj + (1.0 - self.dir_align_infer) * x
        if self.norm_out_infer:
            x = F.normalize(x, dim=1) * (th.norm(anchor, dim=1, keepdim=True) + 1e-8)
        if self.delta_scale_infer != 1.0:
            x = self.delta_scale_infer * x
        return x

    '''
    这是一个简单的辅助函数，封装了“加噪”过程。
    它接收真实数据emb_s和时间t，然后返回在t时刻的带噪数据x_t。本质上就是对path_xt的简单调用。'''

    def forward_process(self, emb_s: th.Tensor, t: th.Tensor, noise: th.Tensor = None):
        if noise is None:
            noise = th.randn_like(emb_s)
        return self.path_xt(noise, emb_s, t)

def innerProduct(usrEmbeds, itmEmbeds):
    return th.sum(usrEmbeds * itmEmbeds, dim=-1)


def pairPredict(ancEmbeds, posEmbeds, negEmbeds):
    return innerProduct(ancEmbeds, posEmbeds) - innerProduct(ancEmbeds, negEmbeds)


def calcRegLoss(model):
    ret = 0
    for W in model.parameters():
        ret += W.norm(2).square()
    # ret += (model.usrStruct + model.itmStruct)
    return ret


# def calcReward(bprLoss, keepRate):
# 	# return t.where(bprLoss >= threshold, 1.0, xi)
# 	_, posLocs = t.topk(bprLoss, int(bprLoss.shape[0] * (1 - keepRate)))
# 	ones = t.ones_like(bprLoss).cuda()
# 	reward = t.minimum(bprLoss, ones * (0.5 - 1e-6))
# 	pckBprLoss = bprLoss[posLocs]
# 	ones = t.ones_like(pckBprLoss).cuda()
# 	reward[posLocs] = t.minimum(t.maximum(pckBprLoss, ones * (0.5 + 1e-6)), ones)
# 	return reward

def calcReward(bprLossDiff, keepRate):
    _, posLocs = th.topk(bprLossDiff, int(bprLossDiff.shape[0] * (1 - keepRate)))
    reward = th.zeros_like(bprLossDiff).cuda()
    reward[posLocs] = 1.0
    return reward


def calcGradNorm(model):
    ret = 0
    for p in model.parameters():
        if p.grad is not None:
            ret += p.grad.data.norm(2).square()
    ret = (ret ** 0.5)
    ret.detach()
    return ret


def getFileName(save, epoch, args):
    file = f"autocf-{args.data}--{epoch}.pth.tar"
    return os.path.join(save, 'model', file)


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
        os.mkdir(os.path.join(path, 'model'))

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

class RecFlow(SocialAbstractRecommender):
    def __init__(self, config):
        super(RecFlow, self).__init__(config)
        self.sampling_steps = config["sampling_steps"]
        self.sampling_noise = config["sampling_noise"]
        self.est_K = config["est_K"]
        fix_random_seed_as(config["seed"])
        self.reweight = config["reweight"]
        self.reg = config["reg"]
        self.batch_size = config["batch_size"]
        self.epochs = config["epoch"]
        self.steps = config["steps"]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items

        # self.uuGraph = self._create_social_adj_mat()
        # self.uiGraph = self._create_adj_mat()
        self.uuGraph = self._create_social_adj_mat().to(self.device)
        self.uiGraph = self._create_adj_mat().to(self.device)

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

        self.FlowProcess = FlowMatchingProcess(config["steps"], device=self.device, t_dist="cosine", eps=1e-4).to(self.device)

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
        self.delta_scale = config["delta_scale"]
        self.diff_w = config["diff_w"]
        self.fm_sampling = config["fm_sampling"]
        self.anchor_ratio = config["anchor_ratio"]
        self.ema = EMA(self.SDNet, decay=config["ema_decay"])

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
            train_dataset.negSampling()
            tqdm_dataloader = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}")

            for iteration, batch in enumerate(tqdm_dataloader):
                user_idx, pos_idx, neg_idx = [x.to(self.device) for x in batch]

                uiEmbeds, uuEmbeds = self.GCNModel(self.uiGraph, self.uuGraph, True)
                uEmbeds = uiEmbeds[:self.num_users]
                iEmbeds = uiEmbeds[self.num_users:]
                user, pos, neg = uEmbeds[user_idx], iEmbeds[pos_idx], iEmbeds[neg_idx]

                # 流匹配损失
                uu_terms = self.FlowProcess.caculate_losses(self.SDNet, uuEmbeds[user_idx], self.reweight)
                diffloss = uu_terms["loss"].mean()

                # 增强用户嵌入
                user_enhanced = user + self.delta_scale * uu_terms["pred_xstart"]

                # 推荐损失 (BPR + Reg)
                scoreDiff = pairPredict(user_enhanced, pos, neg)
                bprLoss = - (scoreDiff).sigmoid().log().mean()
                regLoss = (torch.norm(user, 2).pow(2) + torch.norm(pos, 2).pow(2) + torch.norm(neg, 2).pow(2)) / len(
                    user_idx) * self.reg
                baseLoss = bprLoss + regLoss

                # 总损失
                loss = self.diff_w * diffloss + baseLoss

                # 反向传播和优化
                self.optimizer1.zero_grad()
                self.optimizer2.zero_grad()
                loss.backward()
                self.optimizer1.step()
                self.optimizer2.step()

                # 更新EMA
                self.ema.update(self.SDNet)

                tqdm_dataloader.set_postfix(loss=loss.item(), bpr=bprLoss.item(), diff=diffloss.item())

            # 每个epoch结束后进行评估
            result = self.evaluate_model()
            # self.logger.info(f"Epoch {epoch + 1}: {result}")
            # self.logger.info(f"epoch {epoch}:\t{result}")
            self.logger.info("[iter %d:time:%f]" % (epoch, time.time() - training_start_time))
            self.scheduler1.step()
            self.scheduler2.step()

    def evaluate_model(self):
        self.GCNModel.eval()
        self.SDNet.eval()
        return self.evaluator.evaluate(self)

    def predict(self, users):
        users_tensor = torch.from_numpy(np.asarray(users)).long().to(self.device)

        with torch.no_grad():
            # 1. 从GCN获取基础表征
            uiEmbeds, uuEmbeds = self.GCNModel(self.uiGraph, self.uuGraph, isTrain=False)
            uEmbeds = uiEmbeds[:self.num_users]
            iEmbeds = uiEmbeds[self.num_users:]
            user_embeddings = uEmbeds[users_tensor]
            uuemb = uuEmbeds[users_tensor]

            # 2. **重要**: 使用EMA更新后的模型进行预测
            sdnet_eval = copy.deepcopy(self.SDNet).eval().to(self.device)
            self.ema.apply_to(sdnet_eval)

            # 3. **重要**: 根据配置调用正确的采样方法生成增强向量
            user_predict = None
            if self.fm_sampling == "direct":

                user_predict = self.FlowProcess.direct_estimate(
                    sdnet_eval, uuemb, K=self.est_K, anchor=uuemb, anchor_ratio=self.anchor_ratio
                )
            else:  # 默认为 "ode" 采样
                ssteps = self.sampling_steps if self.sampling_steps > 0 else max(40, self.steps)
                user_predict = self.FlowProcess.p_sample(
                    sdnet_eval, uuemb, ssteps, self.sampling_noise,
                    anchor=uuemb, anchor_ratio=self.anchor_ratio
                )

            # 4. **重要**: 使用 delta_scale 增强用户表征
            final_user_embeddings = user_embeddings + self.delta_scale * user_predict

            # 5. 计算分数
            scores = torch.mm(final_user_embeddings, iEmbeds.t())

        return scores.cpu().detach().numpy()
























