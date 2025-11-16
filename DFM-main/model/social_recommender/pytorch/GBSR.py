import torch
import torch.sparse as torch_sp
import torch.nn as nn
import torch.nn.functional as F
import time
from model.base.abstract_recommender import SocialAbstractRecommender
from util.pytorch import inner_product, l2_loss
from util.pytorch import pairwise_loss
from util.common import Reduction
from data import PairwiseSampler
import numpy as np
from reckit import timer
import scipy.sparse as sp
from util.common import normalize_adj_matrix

def kernel_matrix(x, sigma):
    return torch.exp((torch.matmul(x, x.transpose(0,1)) - 1) / sigma)

def hsic(Kx, Ky, m):
    Kxy = torch.mm(Kx, Ky)
    h = torch.trace(Kxy) / m ** 2 + torch.mean(Kx) * torch.mean(Ky) - \
        2 * torch.mean(Kxy) / m
    return h * (m / (m - 1)) ** 2

#模型结构模块----定义模型的核心组件，包括嵌入层、图卷积、社交图去噪模块。
class _GBSR(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, n_layers, l2_reg,
                 adj_matrix, social_index, social_u, social_v, social_weight, beta, sigma, edge_bias, device, batch_size,social_file):
        super(_GBSR, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.n_layers = n_layers
        self.l2_reg = l2_reg

        self.adj_matrix = adj_matrix
        self.social_index = social_index
        self.social_u = social_u
        self.social_v = social_v
        self.social_weight = social_weight
        self.social_file = social_file

        self.beta = beta
        self.sigma = sigma
        self.edge_bias = edge_bias

        self.device = device
        self.batch_size = batch_size

        self.Mask_MLP1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.Mask_MLP2 = nn.Linear(64, 1)
        self.user_embeddings = nn.Embedding(num_users, embed_dim)
        self.item_embeddings = nn.Embedding(num_items, embed_dim)
        self._user_embeddings_final = None
        self._item_embeddings_final = None

        # weight initialization
        self.reset_parameters()

#初始化表征
    def reset_parameters(self):
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)

#图卷积----通过多层邻居信息聚合，让用户和物品的向量包含社交关系和交互信息。

    def _forward_gcn(self, adj_matrix):
        ego_embeddings = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        all_embeddings = [ego_embeddings]

        for k in range(self.n_layers):
            ego_embeddings = torch_sp.mm(adj_matrix, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)

        return user_embeddings, item_embeddings

#社交图去噪（graph_learner）用MLP预测每条社交边的重要性（例如用户A和B是否是有效好友）使用Gumbel-Softmax生成可学习的“掩码”，动态调整边的权重。
#偏好引导的社交图去噪（论文公式6-8）
    def graph_learner(self, user_emb):
        # 计算边权重（公式7）
        row, col = self.social_u, self.social_v
        row_emb = user_emb.weight[row]    # 用户u的嵌入
        col_emb = user_emb.weight[col]    # 用户v的嵌入
        cat_emb = torch.cat([row_emb, col_emb], dim=1)
        out_layer1 = self.relu(self.Mask_MLP1(cat_emb))
        logit = self.Mask_MLP2(out_layer1)
        logit = logit.view(-1)
        ## Gumbel-Softmax生成掩码（公式8）
        eps = torch.rand(logit.shape).to(self.device)
        mask_gate_input = torch.log(eps) - torch.log(1 - eps)
        mask_gate_input = (logit + mask_gate_input) / 0.2
        mask_gate_input = torch.sigmoid(mask_gate_input) + self.edge_bias  # self.edge_bias
        weights = torch.ones_like(self.adj_matrix.values())
        # 生成去噪后的邻接矩阵（公式6）
        weights[self.social_index] = mask_gate_input
        weights = weights.detach()
        masked_Graph = torch.sparse.FloatTensor(self.adj_matrix.indices(), self.adj_matrix.values()*weights, torch.Size(
            [self.num_users + self.num_items, self.num_users + self.num_items]))
        masked_Graph = masked_Graph.coalesce().to(self.device)
        return masked_Graph

    def getEmbedding(self, users, pos_items, neg_items):
        # all_users, all_items = self.forward()
        users_emb = self.user_emb[users]
        pos_emb = self.item_emb[pos_items]
        neg_emb = self.item_emb[neg_items]
        users_emb_ego = self.user_embeddings(users)
        pos_emb_ego = self.item_embeddings(pos_items)
        neg_emb_ego = self.item_embeddings(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

#BPR损失----让用户对正样本（点击过的物品）的评分高于负样本（未点击的）。
#BPR损失与互信息下界（论文公式9）
    def bpr_loss(self, users, pos_items, neg_items):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos_items.long(), neg_items.long())
        reg_loss = 1/2 * (userEmb0.norm(2).pow(2) +
                    posEmb0.norm(2).pow(2) +
                    negEmb0.norm(2).pow(2)) / float(len(users))
        ## 计算用户对正样本和负样本的得分
        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)
        auc = torch.mean((pos_scores > neg_scores).float())
        # BPR损失（对应公式9最后一行）
        bpr_loss = torch.mean(-torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-9))
        return auc, bpr_loss, reg_loss*self.l2_reg
#HSIC损失----减少去噪后的图与原图的依赖，去除冗余信息。
    def hsic_graph(self, users, pos_items):
        ### user part ###
        users = torch.unique(users)
        items = torch.unique(pos_items)
        input_x = self.user_emb_old[users]
        input_y = self.user_emb[users]
        input_x = F.normalize(input_x, p=2, dim=1)
        input_y = F.normalize(input_y, p=2, dim=1)
        Kx = kernel_matrix(input_x, self.sigma)
        Ky = kernel_matrix(input_y, self.sigma)
        loss_user = hsic(Kx, Ky, self.batch_size)
        ### item part ###
        input_i = self.item_emb_old[items]
        input_j = self.item_emb[items]
        input_i = F.normalize(input_i, p=2, dim=1)
        input_j = F.normalize(input_j, p=2, dim=1)
        Ki = kernel_matrix(input_i, self.sigma)
        Kj = kernel_matrix(input_j, self.sigma)
        loss_item = hsic(Ki, Kj, self.batch_size)
        loss = loss_user + loss_item
        return loss

#信息瓶颈目标函数（论文公式5/19）
    def calculate_all_loss(self, users, pos_items, neg_items):
        # 1. learning denoised social graph
        self.masked_adj_matrix = self.graph_learner(self.user_embeddings)
        # 2. learning embeddings from lightgcn-s
        self.user_emb_old, self.item_emb_old = self._forward_gcn(self.adj_matrix)
        self.user_emb, self.item_emb = self._forward_gcn(self.masked_adj_matrix)
        # 3. Max mutual information # 计算BPR损失（对应I(R; S')）
        auc, bpr_loss, reg_loss = self.bpr_loss(users, pos_items, neg_items)
        # 4. Min mutual information 计算HSIC损失（对应βI(S'; S)）
        ib_loss = self.hsic_graph(users, pos_items) * self.beta
        loss = bpr_loss + reg_loss + ib_loss   # 总损失 = BPR + 正则化 + β*HSIC
        return auc, bpr_loss, reg_loss, ib_loss, loss

    def predict(self, users):
        user_embs = F.embedding(users, self._user_embeddings_final)
        ratings = torch.matmul(user_embs, self._item_embeddings_final.T)
        return ratings

    def eval(self):
        super(_GBSR, self).eval()
        self._user_embeddings_final, self._item_embeddings_final = self._forward_gcn(self.masked_adj_matrix)


class GBSR(SocialAbstractRecommender):
    def __init__(self, config):
        super(GBSR, self).__init__(config)
        self.lr = config['lr']
        self.reg = config['reg']
        self.emb_size = config['embed_size']
        self.batch_size = config['batch_size']
        self.epochs = config["epochs"]
        self.n_layers = config['n_layers']
        self.beta = config['beta']
        self.sigma = config['sigma']
        self.edge_bias = config['edge_bias']
        self.social_file = config['social_file']
        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        adj_matrix, social_index = self.get_uu_i_matrix()
        social_u = adj_matrix.indices()[0][social_index]
        social_v = adj_matrix.indices()[1][social_index]
        social_weight = adj_matrix.values()[social_index]

        self.GBSR = _GBSR(self.num_users, self.num_items, self.emb_size, self.n_layers, self.reg,
                          adj_matrix, social_index, social_u, social_v, social_weight,
                          self.beta, self.sigma, self.edge_bias, self.device, self.batch_size,self.social_file).to(self.device)
        self.GBSR.reset_parameters()
        self.optimizer = torch.optim.Adam(self.GBSR.parameters(), lr=self.lr)


    @timer
    def create_adj_mat(self):
        users_items = self.dataset.train_data.to_user_item_pairs()
        users_np, items_np = users_items[:, 0], users_items[:, 1]

        ratings = np.ones_like(users_np, dtype=np.float32)
        n_nodes = self.num_users + self.num_items
        up_left_adj = sp.csr_matrix((ratings, (users_np, items_np+self.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = up_left_adj + up_left_adj.T
        adj_matrix = normalize_adj_matrix(adj_mat, norm_method="symmetric")

        return adj_matrix


    def create_social_adj_mat(self):
        users_items = self.dataset.train_data.to_user_item_pairs()
        users_np, items_np = users_items[:, 0], users_items[:, 1]

        ratings = np.ones_like(users_np, dtype=np.float32)
        n_nodes = self.num_users + self.num_items
        up_left_adj = sp.csr_matrix((ratings, (users_np, items_np + self.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = up_left_adj + up_left_adj.T

        uu_idx = [[ui, uj] for (ui, uj), r in self.social_matrix.todok().items()]
        u1_idx, u2_idx = list(zip(*uu_idx))
        u1_idx = np.array(u1_idx, dtype=np.int32)
        u2_idx = np.array(u2_idx, dtype=np.int32)
        social_ratings = np.ones_like(u1_idx, dtype=np.float32)
        social_adj = sp.csr_matrix((social_ratings, (u1_idx, u2_idx)), shape=(n_nodes, n_nodes))
        adj_mat = adj_mat + social_adj

        adj_matrix = normalize_adj_matrix(adj_mat, norm_method="symmetric")
        return adj_matrix


    def social_index_in_social_lightgcn(self):
        uu_i_matrix = self.create_social_adj_mat()
        adj_indices, adj_values, adj_shape = self.convert_csr_to_sparse_tensor_inputs(uu_i_matrix)
        social_edge_index = []
        for i in range(adj_indices.shape[0]):
            # pdb.set_trace()
            if adj_indices[i, 0] < self.num_users:
                if adj_indices[i, 1] < self.num_users:
                    social_edge_index.append(i)
                else:
                    continue
            else:
                continue
        print('number of social edges:', len(social_edge_index))
        return np.array(social_edge_index)


    # def get_uu_i_matrix(self):
    #     users_items = self.dataset.train_data.to_user_item_pairs()
    #     users_np, items_np = users_items[:, 0], users_items[:, 1]
    #     uu_idx = [[ui, uj] for (ui, uj), r in self.social_matrix.todok().items()]
    #     u1_idx, u2_idx = list(zip(*uu_idx))
    #
    #     user_dim = torch.LongTensor(users_np)
    #     item_dim = torch.LongTensor(items_np) + self.num_users
    #     social_i = torch.LongTensor(u1_idx)
    #     social_j = torch.LongTensor(u2_idx)
    #     first_sub = torch.stack([user_dim, item_dim])
    #     second_sub = torch.stack([item_dim, user_dim])
    #     third_sub = torch.stack([social_i, social_j])
    #     index = torch.cat([first_sub, second_sub, third_sub], dim=1)
    #     data = torch.ones(index.size(-1)).int()
    #     Graph = torch.sparse.IntTensor(index, data,
    #                                         torch.Size([self.num_users + self.num_items, self.num_users + self.num_items]))
    #     dense = Graph.to_dense()
    #     D = torch.sum(dense, dim=1).float()
    #     D[D == 0.] = 1.
    #     D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
    #     dense = dense / D_sqrt
    #     dense = dense / D_sqrt.t()
    #     index = dense.nonzero()
    #     data = dense[dense >= 1e-9]
    #     assert len(index) == len(data)
    #     # pdb.set_trace()
    #     Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size(
    #         [self.num_users + self.num_items, self.num_users + self.num_items]))
    #     Graph = Graph.coalesce().to(self.device)
    #     social_edge_index = []
    #     for i in range(index.shape[0]):
    #         if index[i][0] < self.num_users:
    #             if index[i][1] < self.num_users:
    #                 social_edge_index.append(i)
    #             else:
    #                 continue
    #         else:
    #             continue
    #     return Graph, social_edge_index

    def get_uu_i_matrix(self):
        # 构建用户-物品、物品-用户和社交关系的边索引
        users_items = self.dataset.train_data.to_user_item_pairs()
        users_np, items_np = users_items[:, 0], users_items[:, 1]

        # 用户-物品交互边（双向）
        user_dim = torch.LongTensor(users_np)
        item_dim = torch.LongTensor(items_np) + self.num_users
        first_sub = torch.stack([user_dim, item_dim])  # 用户 -> 物品
        second_sub = torch.stack([item_dim, user_dim])  # 物品 -> 用户

        # 社交关系边（用户-用户）
        uu_idx = [[ui, uj] for (ui, uj), r in self.social_matrix.todok().items()]
        social_i = torch.LongTensor(uu_idx[0])
        social_j = torch.LongTensor(uu_idx[1])
        third_sub = torch.stack([social_i, social_j])

        # 合并所有边索引
        index = torch.cat([first_sub, second_sub, third_sub], dim=1)
        data = torch.ones(index.size(-1), dtype=torch.float32)

        # 构建并合并稀疏邻接矩阵
        graph_size = self.num_users + self.num_items
        Graph = torch.sparse_coo_tensor(
            index,
            data,
            (graph_size, graph_size)
        ).coalesce()  # 关键修复：添加立即合并操作

        # 计算度矩阵（保持稀疏格式）
        D = torch.sparse.sum(Graph, dim=1).to_dense().float()
        D[D == 0.] = 1.  # 防止除零错误

        # 获取合并后的索引
        rows, cols = Graph.indices()  # 现在可以安全访问

        # 计算对称归一化系数
        sqrt_D = torch.sqrt(D)
        norm_factors = 1.0 / (sqrt_D[rows] * sqrt_D[cols])

        # 重建归一化稀疏矩阵
        normalized_Graph = torch.sparse_coo_tensor(
            torch.stack([rows, cols]),
            Graph.values() * norm_factors,
            (graph_size, graph_size)
        ).coalesce().to(self.device)

        # 提取社交关系边的索引
        user_mask = (rows < self.num_users) & (cols < self.num_users)
        social_edge_index = user_mask.nonzero().squeeze()

        return normalized_Graph, social_edge_index


    def convert_csr_to_sparse_tensor_inputs(self, X):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        # pdb.set_trace()
        return indices, coo.data, coo.shape


    def train_model(self):
        data_iter = PairwiseSampler(self.dataset.train_data, num_neg=1,
                                    batch_size=self.batch_size,
                                    shuffle=True, drop_last=False)
        self.logger.info(self.evaluator.metrics_info())
        training_start_time = time.time()
        for epoch in range(self.epochs):
            self.GBSR.train()
            for bat_users, bat_pos_items, bat_neg_items in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_pos_items = torch.from_numpy(bat_pos_items).long().to(self.device)
                bat_neg_items = torch.from_numpy(bat_neg_items).long().to(self.device)
                auc, rank_loss, reg_loss, ib_loss, total_loss = self.GBSR.calculate_all_loss(bat_users, bat_pos_items, bat_neg_items)
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

            result = self.evaluate_model()
            # self.logger.info("epoch %d:\t%s" % (epoch, result))
            self.logger.info("[iter %d:time:%f]" % (epoch, time.time() - training_start_time))


    def evaluate_model(self):
        self.GBSR.eval()
        return self.evaluator.evaluate(self)

    def predict(self, users):
        users = torch.from_numpy(np.asarray(users)).long().to(self.device)
        return self.GBSR.predict(users).cpu().detach().numpy()
