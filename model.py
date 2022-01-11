import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from data_loader import build_cross_graph_index


def init_he_weight(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")


def init_xavier_weight(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)

class Initialization(nn.Module):

    def __init__(self, word_dict, feat_dim, device):
        super(Initialization, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(len(word_dict), feat_dim)
        nn.init.xavier_normal_(self.embedding.weight)

    def forward(self, word_indexs):
        X = self.embedding(torch.from_numpy(
            word_indexs).long().to(device=self.device))
        return X


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True,
                 activation=F.relu, batch=False):
        super(MLP, self).__init__()
        self.batch = batch
        self.linear = nn.Linear(input_dim, output_dim, bias)
        if batch:
            self.batchnorm = nn.BatchNorm1d(output_dim)
        self.activation = activation
        if bias:
            nn.init.zeros_(self.linear.bias)
        if self.activation == F.relu:
            nn.init.kaiming_normal_(self.linear.weight, nonlinearity="relu")
        elif self.activation == F.leaky_relu:
            nn.init.kaiming_normal_(self.linear.weight)
        else:
            nn.init.xavier_normal_(self.linear.weight)

    def forward(self, X):
        X = self.linear(X)
        if self.batch:
            if X.shape[0] > 1:
                X = self.batchnorm(X)
        X = self.activation(X)
        return X

class Message_Attention(nn.Module):

    def __init__(self, node_attr_dim, device):
        super(Message_Attention, self).__init__()
        self.device = device
        self.f_w1 = MLP(node_attr_dim, node_attr_dim,
                        bias=False, activation=lambda x: x)
        self.f_w2 = MLP(2 * node_attr_dim, 1,
                        bias=False, activation=lambda x: x)

    def forward(self, X_n, edge_indices, adj):
        if adj is not None:
            N = X_n.shape[0]
            edge_N = edge_indices.shape[1]
            X_n_w = self.f_w1(X_n)
            X_n_stack = torch.cat([X_n_w.repeat(1, N).view(
                N * N, -1), X_n_w.repeat(N, 1)], dim=1).view(N, N, -1)
            weight = self.f_w2(X_n_stack).squeeze(2)
            zero_vec = -9e15 * torch.ones_like(weight)
            attention = F.softmax(torch.where(
                adj > 0, weight, zero_vec), dim=1)

            a_p = attention[edge_indices[1], edge_indices[0]]
            a_c = attention[edge_indices[0], edge_indices[1]]

            edge_N = edge_indices.shape[1]
            atten_matrix = torch.zeros(
                N, edge_N * 2, dtype=torch.float32).to(device=self.device)
            atten_matrix[[edge_indices[1]], np.arange(
                edge_N, dtype=np.int64)] = a_p
            atten_matrix[[edge_indices[0]], np.arange(
                edge_N, dtype=np.int64) + edge_N] = a_c
        else:
            atten_matrix = None
        return atten_matrix


class Message(nn.Module):

    def __init__(self, node_dim, edge_dim, device):
        super(Message, self).__init__()
        self.device = device
        self.f_p = nn.Sequential(MLP(2 * node_dim + edge_dim, node_dim),
                                 MLP(node_dim, node_dim))
        self.f_c = nn.Sequential(MLP(2 * node_dim + edge_dim, node_dim),
                                 MLP(node_dim, node_dim))

    def forward(self, X_h, X_e, edge_indices, atten_matrix):
        if atten_matrix is not None:
            X_p = self.f_p(
                torch.cat([X_h[edge_indices[0]],
                           X_h[edge_indices[1]], X_e], dim=1))
            X_c = self.f_c(
                torch.cat([X_h[edge_indices[0]],
                           X_h[edge_indices[1]], X_e], dim=1))

            X_m = torch.matmul(atten_matrix, torch.cat([X_p, X_c], dim=0))
        else:
            X_m = torch.zeros_like(X_h)
        return X_m


class Cross_Message(nn.Module):

    def __init__(self, node_dim, node_attr_dim, device):
        super(Cross_Message, self).__init__()
        self.device = device
        self.f_s = nn.CosineSimilarity()
        self.f_gate = MLP(node_attr_dim, node_dim,
                          bias=False, activation=torch.sigmoid)

    def forward(self, X_h_1, X_h_2, X_n_1, cross_indices):
        N1 = X_h_1.shape[0]
        indices_len = cross_indices.shape[1]
        if indices_len > 0:
            X_1 = X_h_1[cross_indices[0]]
            X_2 = X_h_2[cross_indices[1]]
            sim = self.f_s(X_1, X_2)
            # print(sim)
            weight_matrix = \
                torch.zeros(N1,
                            indices_len,
                            dtype=torch.float32).to(device=self.device)
            for i in range(N1):
                w_indices = np.where(i == cross_indices[0])
                if len(w_indices[0]) > 0:
                    weight_matrix[i][w_indices] = F.softmax(
                        sim[w_indices], dim=-1)
            gates = self.f_gate(X_n_1)
            # print(gates)
            X_cm = gates * torch.matmul(weight_matrix, X_2)
        else:
            X_cm = torch.zeros_like(X_h_1)
        return X_cm


class Update(nn.Module):

    def __init__(self, node_dim):
        super(Update, self).__init__()
        self.f_rm = MLP(node_dim, node_dim, activation=lambda x: x)
        self.f_rh = MLP(node_dim, node_dim, activation=lambda x: x)
        self.f_um = MLP(node_dim, node_dim, activation=lambda x: x)
        self.f_uh = MLP(node_dim, node_dim, activation=lambda x: x)
        self.f_hm = MLP(node_dim, node_dim, activation=lambda x: x)
        self.f_hh = MLP(node_dim, node_dim, activation=lambda x: x)

    def forward(self, X_m, X_cm, X_h):
        X_cat_m = X_m + X_cm
        reset_gate = torch.sigmoid(self.f_rm(X_cat_m) + self.f_rh(X_h))
        update_gate = torch.sigmoid(self.f_um(X_cat_m) + self.f_uh(X_h))
        X_h_hat = torch.tanh(self.f_hm(X_cat_m) + reset_gate * self.f_hh(X_h))
        X_h_new = (1 - update_gate) * X_h + update_gate * X_h_hat
        return X_h_new


class Aggregation(nn.Module):

    def __init__(self, node_dim, node_attr_dim):
        super(Aggregation, self).__init__()
        self.f_n = MLP(node_dim, node_dim, activation=lambda x: x)
        self.f_g = MLP(node_attr_dim, node_dim, activation=lambda x: x)
        self.f_G = MLP(node_dim, node_dim)

    def forward(self, X_h, X_n):
        X_temp = torch.sigmoid(self.f_g(X_n)) * self.f_n(X_h)
        X_G = self.f_G(torch.sum(X_temp, dim=0).view(1, -1)).view(-1)
        # print(X_G)
        return X_G

class Graph_Embedding(nn.Module):

    def __init__(self, node_dict, node_attr_dict, edge_dict, node_dim,
                 node_attr_dim, edge_dim, steps, device):
        super(Graph_Embedding, self).__init__()
        self.steps = steps
        self.device = device
        self.node_initial = Initialization(node_dict, node_dim, device)
        self.edge_initial = Initialization(edge_dict, edge_dim, device)
        self.node_attr_initial = Initialization(
            node_attr_dict, node_attr_dim, device)
        self.attention = Message_Attention(node_attr_dim, device)
        self.message = Message(node_dim, edge_dim, device)
        self.cross_message = Cross_Message(node_dim, node_attr_dim, device)
        self.update = Update(node_dim)
        self.aggregation = Aggregation(node_dim, node_attr_dim)

    def forward(self, G1, G2):
        cross_indices_1, cross_indices_2 = build_cross_graph_index(G1, G2)
        if G1.adj is not None:
            adj_1 = torch.from_numpy(G1.adj).to(device=self.device)
        else:
            adj_1 = None
        if G2.adj is not None:
            adj_2 = torch.from_numpy(G2.adj).to(device=self.device)
        else:
            adj_2 = None

        X_h_1 = self.node_initial(G1.data)
        # print(X_h_1)
        X_h_2 = self.node_initial(G2.data)
        # print(X_h_2)

        X_e_1 = self.edge_initial(G1.edge_attr)
        X_e_2 = self.edge_initial(G2.edge_attr)
        # print(X_e_1)
        # print(X_e_2)

        X_n_1 = self.node_attr_initial(G1.node_attr)
        X_n_2 = self.node_attr_initial(G2.node_attr)

        attention_1 = self.attention(X_n_1, G1.indices, adj_1)
        attention_2 = self.attention(X_n_2, G2.indices, adj_2)

        for step in range(self.steps):
            X_m_1 = self.message(X_h_1, X_e_1, G1.indices, attention_1)
            X_m_2 = self.message(X_h_2, X_e_2, G2.indices, attention_2)
            X_cm_1 = self.cross_message(X_h_1, X_h_2, X_n_1, cross_indices_1)
            X_cm_2 = self.cross_message(X_h_2, X_h_1, X_n_2, cross_indices_2)
            X_h_1 = self.update(X_m_1, X_cm_1, X_h_1)
            X_h_2 = self.update(X_m_2, X_cm_2, X_h_2)
        X_G_1 = self.aggregation(X_h_1, X_n_1)
        X_G_2 = self.aggregation(X_h_2, X_n_2)
        return X_G_1, X_G_2


class Classifier(nn.Module):

    def __init__(self, node_dim):
        super(Classifier, self).__init__()
        self.f_conj = MLP(node_dim, node_dim, activation=F.relu, batch=True)
        self.f_prem = MLP(node_dim, node_dim, activation=F.relu, batch=True)
        self.f_comb = MLP(2 * node_dim, node_dim,
                          activation=F.relu, batch=True)
        self.f_class = MLP(node_dim, 2, activation=lambda x: x, batch=True)
        # self.criterion = nn.CrossEntropyLoss()
        self.pred_labels = None
        self.pred_scores = None

    def forward(self, X_conj, X_prem):
        X_conj = self.f_conj(X_conj)
        X_prem = self.f_prem(X_prem)
        X_comb = torch.cat(
            ([X_conj * X_prem, torch.abs(X_conj - X_prem)]), dim=1)
        outputs = self.f_class(self.f_comb(X_comb))
        # loss = self.criterion(outputs, labels)
        self.pred_labels = torch.max(outputs, dim=1)[1]
        self.pred_scores = F.softmax(outputs, dim=1)[:, 1]
        return outputs
