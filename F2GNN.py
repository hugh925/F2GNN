import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
import dgl

class F2Layer(nn.Module):
    def __init__(self, in_dim, head_num, dropout):
        super(F2Layer, self).__init__()
        self.in_dim = in_dim
        self.head_num = head_num
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(2 * in_dim, 1)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)

    def sign_edges(self, edges):
        if self.head_num > 1:
            h1 = edges.dst['h'].view(-1, self.head_num, self.in_dim)
            h2 = edges.src['h'].view(-1, self.head_num, self.in_dim)
            h_cat = torch.cat([h1, h2], dim=-1)
            # s = F.cosine_similarity(edges.src['h'], edges.dst['h'], dim=1)
            s = torch.tanh(self.gate(h_cat)).squeeze(dim=-1)
            s = s*edges.dst['d'].view(-1,1) * edges.src['d'].view(-1,1)
            s = s.unsqueeze(dim=-1)
            e_h = h2 * s
            e_h = e_h.view(-1, self.in_dim*self.head_num)

        else:
            h_cat = torch.cat([edges.dst['h'], edges.src['h']], dim=-1)
            s = torch.tanh(self.gate(h_cat)).squeeze()
            e_h = (s * edges.dst['d'] * edges.src['d']).unsqueeze(dim=-1)*edges.src['h']

        return {'e_h': e_h}

    def forward(self, g, h):
        self.g = g
        self.g.ndata['h'] = h
        deg = g.in_degrees().to(g.device).float().clamp(min=1)
        norm = torch.pow(deg, -0.5)
        g.ndata['d'] = norm
        self.g.apply_edges(self.sign_edges)
        #The message passing matrix is exposed.
        self.g.edata['e_h'] = self.dropout(self.g.edata['e_h'])
        self.g.update_all(fn.copy_e('e_h', '_'), fn.sum('_', 'z'))

        return self.g.ndata['z']


class F2GNN(nn.Module):
    def __init__(self, graph, in_dim, hid_dim, out_dim, dropout, eps,head_num, layer_num):
        super(F2GNN, self).__init__()
        self.graph = graph
        self.eps = eps
        self.layer_num = layer_num
        self.dropout = dropout

        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(F2Layer(hid_dim, head_num, dropout))

        self.t1 = nn.Linear(in_dim, hid_dim*head_num)
        self.t2 = nn.Linear(hid_dim*head_num, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward(self, h):
        h = torch.relu(self.t1(h))
        raw = h

        for i in range(self.layer_num):
            h0 = self.layers[i](self.graph, h)
            h1 = self.eps * raw + h0
        h = self.t2(h1)
        return h


class F2GNNhetero(nn.Module):
    def __init__(self, graph, in_dim, hid_dim, out_dim, dropout, eps, head_num, layer_num):
        super(F2GNNhetero, self).__init__()
        self.graph = graph
        self.eps = eps
        self.layer_num = layer_num
        self.dropout = dropout

        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(F2Layer(hid_dim, head_num, dropout))

        self.t1 = nn.Linear(in_dim, hid_dim*head_num)
        self.t2 = nn.Linear(hid_dim*head_num, out_dim)
        self.linear3 = nn.Linear(hid_dim *head_num * 3, hid_dim*head_num)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)


    def forward(self, h):
        h = torch.relu(self.t1(h))
        raw = h

        for i in range(self.layer_num):
            h_final = torch.zeros([len(h), 0]).to(h.device)
            for relation in self.graph.canonical_etypes:
                sub_graph = self.graph[relation]
                # sub_graph = dgl.add_self_loop(sub_graph)
                h0 = self.layers[i](sub_graph,h)
                h1 = self.eps * raw + h0
                h_final = torch.cat([h_final, h1], -1)
            h = self.linear3(h_final)
        h = self.t2(h)
        return h













