import time

import torch
import torch.sparse as torch_sp
import torch.nn as nn
import torch.nn.functional as F
from model.base.abstract_recommender import SocialAbstractRecommender
from util.pytorch import inner_product, l2_loss
from util.pytorch import pairwise_loss, pointwise_loss
from util.pytorch import get_initializer
from util.common import Reduction
from data import PointwiseSampler, PairwiseSampler
import numpy as np
from reckit import timer
import scipy.sparse as sp
from util.common import normalize_adj_matrix
from util.pytorch import sp_mat_to_sp_tensor


class _DiffNetPlus(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, norm_adj, social_adj_matrix, n_layers, device):
        super(_DiffNetPlus, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.norm_adj = norm_adj
        self.social_norm_adj = social_adj_matrix
        self.n_layers = n_layers
        self.device = device
        self.user_embeddings = nn.Embedding(num_users, embed_dim)
        self.item_embeddings = nn.Embedding(num_items, embed_dim)
        self._user_embeddings_final = None
        self._item_embeddings_final = None

        # weight initialization
        self.reset_parameters()

    def reset_parameters(self, init_method="uniform"):
        init = get_initializer(init_method)
        init(self.user_embeddings.weight)
        init(self.item_embeddings.weight)

    def forward(self, users, items):
        user_embeddings, item_embeddings = self._recsys_gcn()
        user_embs = F.embedding(users, user_embeddings)
        item_embs = F.embedding(items, item_embeddings)
        ratings = inner_product(user_embs, item_embs)
        return ratings

    def _gcn(self, norm_adj, init_embeddings):
        ego_embeddings = init_embeddings
        all_embeddings = [ego_embeddings]
        for k in range(self.n_layers):
            ego_embeddings = torch_sp.mm(norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)

        return all_embeddings

    def _recsys_gcn(self):
        norm_adj = self.norm_adj
        ego_embeddings = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        all_embeddings = self._gcn(norm_adj, ego_embeddings)
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)
        user_embeddings, item_embeddings = user_embeddings.to(self.device), item_embeddings.to(self.device)

        return user_embeddings, item_embeddings

    def _social_gcn(self):
        norm_adj = self.social_norm_adj
        user_embeddings = self._gcn(norm_adj, self.user_embeddings.weight)
        return user_embeddings

    def predict(self, users):
        self.final_user_embdedding = 0.9 * self._user_embeddings_final + 0.1 * self._social_embeddings_final
        user_embs = F.embedding(users, self.final_user_embdedding)
        ratings = torch.matmul(user_embs, self._item_embeddings_final.T)
        return ratings

    def eval(self):
        super(_DiffNetPlus, self).eval()
        self._user_embeddings_final, self._item_embeddings_final = self._recsys_gcn()
        self._social_embeddings_final = self._social_gcn()

class DiffNetPlus(SocialAbstractRecommender):
    def __init__(self, config):
        super(DiffNetPlus, self).__init__(config)
        self.lr = config['lr']
        self.reg = config['reg']
        self.emb_size = config['embed_size']
        self.batch_size = config['batch_size']
        self.epochs = config["epochs"]
        self.n_layers = config['n_layers']
        self.adj_type = config["adj_type"]

        self.param_init = config["param_init"]
        self.is_pairwise = config["is_pairwise"]
        self.loss_func = config["loss_func"]
        self.num_neg = config["num_neg"]

        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        adj_matrix = self.create_adj_mat()
        adj_matrix = sp_mat_to_sp_tensor(adj_matrix).to(self.device)
        social_adj_matrix = self.create_social_adjmat()
        social_adj_matrix = sp_mat_to_sp_tensor(social_adj_matrix).to(self.device)

        self.diffnetplus = _DiffNetPlus(self.num_users, self.num_items, self.emb_size, adj_matrix, social_adj_matrix, self.n_layers, self.device).to(self.device)
        self.diffnetplus.reset_parameters(self.param_init)
        self.optimizer = torch.optim.Adam(self.diffnetplus.parameters(), lr=self.lr)

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

    def create_social_adjmat(self):
        uu_idx = [[ui, uj] for (ui, uj), r in self.social_matrix.todok().items()]
        u1_idx, u2_idx = list(zip(*uu_idx))
        u1_idx = np.array(u1_idx, dtype=np.int32)
        u2_idx = np.array(u2_idx, dtype=np.int32)
        ratings = np.ones_like(u1_idx, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (u1_idx, u2_idx)), shape=(self.num_users, self.num_users))
        adj_mat = tmp_adj + tmp_adj.T
        adj_matrix = normalize_adj_matrix(adj_mat, norm_method="symmetric")
        return adj_matrix

    def train_model(self):
        data_iter = PointwiseSampler(self.dataset.train_data, num_neg=self.num_neg,
                                     batch_size=self.batch_size,
                                     shuffle=True, drop_last=False)
        self.logger.info(self.evaluator.metrics_info())
        training_start_time = time.time()
        for epoch in range(self.epochs):
            self.diffnetplus.train()
            for bat_users, bat_items, bat_labels in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_items = torch.from_numpy(bat_items).long().to(self.device)
                bat_labels = torch.from_numpy(bat_labels).float().to(self.device)
                yui = self.diffnetplus(bat_users, bat_items)
                loss = pointwise_loss(self.loss_func, yui, bat_labels, reduction=Reduction.MEAN)
                reg_loss = l2_loss(self.diffnetplus.user_embeddings(bat_users),
                                   self.diffnetplus.item_embeddings(bat_items)
                                   )
                loss += self.reg * reg_loss / self.emb_size
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            result = self.evaluate_model()
            # self.logger.info("epoch %d:\t%s" % (epoch, result))
            self.logger.info("[iter %d:time:%f]" % (epoch, time.time() - training_start_time))

    # @timer
    def evaluate_model(self):
        self.diffnetplus.eval()
        return self.evaluator.evaluate(self)

    def predict(self, users):
        users = torch.from_numpy(np.asarray(users)).long().to(self.device)
        return self.diffnetplus.predict(users).cpu().detach().numpy()
