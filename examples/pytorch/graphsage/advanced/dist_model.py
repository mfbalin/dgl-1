import torch as th
import torch.nn as nn
import torch.nn.functional as F

import dgl

import dgl.nn as dglnn

from dgl.contrib.dist_sampling import DistConv, DistConvFunction

class SAGE(nn.Module):
    def __init__(self, num_feats, dropout, replicated=False):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(num_feats) - 1):
            last = i == len(num_feats) - 2
            conv = dglnn.SAGEConv(num_feats[i], num_feats[i + 1], 'mean', feat_drop=0 if last else dropout, activation=nn.Identity() if last else nn.ReLU())
            self.layers.append(DistConv(conv, i != 0 and not replicated))
        self.num_feats = num_feats
    
    def forward(self, blocks, h):
        # h is the dsttensor
        for layer, block in zip(self.layers, blocks):
            h = layer(block, h)
        return h

class RGAT(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        num_etypes,
        num_layers,
        num_heads,
        dropout,
        pred_ntype,
        gat=True,
        replicated=False
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skips = nn.ModuleList()

        self.convs.append(
            nn.ModuleList(
                [
                    dglnn.GATConv(
                        in_channels,
                        hidden_channels // num_heads,
                        num_heads,
                        allow_zero_in_degree=True,
                    )
                    if gat else
                    dglnn.SAGEConv(in_channels, hidden_channels, 'mean', dropout, activation=nn.Identity())
                    for _ in range(num_etypes)
                ]
            )
        )
        self.norms.append(nn.BatchNorm1d(hidden_channels))
        self.skips.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(
                nn.ModuleList(
                    [
                        dglnn.GATConv(
                            hidden_channels,
                            hidden_channels // num_heads,
                            num_heads,
                            allow_zero_in_degree=True,
                        )
                        if gat else
                        dglnn.SAGEConv(hidden_channels, hidden_channels, 'mean', dropout, activation=nn.Identity())
                        for _ in range(num_etypes)
                    ]
                )
            )
            self.norms.append(nn.BatchNorm1d(hidden_channels))
            self.skips.append(nn.Linear(hidden_channels, hidden_channels))

        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )
        self.dropout = nn.Dropout(dropout)

        self.hidden_channels = hidden_channels
        self.pred_ntype = pred_ntype
        self.num_etypes = num_etypes
        self.replicated = replicated

    def forward(self, mfgs, x):
        for i in range(len(mfgs)):
            mfg = mfgs[i]
            if i != 0 and not self.replicated:
                x = DistConvFunction.apply(mfg.cached_variables, x)
            x_dst = x[mfg.dst_in_src]
            mfg = dgl.block_to_graph(mfg)
            x_skip = self.skips[i](x_dst)
            for j in range(self.num_etypes):
                subg = mfg.edge_subgraph(
                    mfg.edata["etype"] == j, relabel_nodes=False
                )
                x_skip += self.convs[i][j](subg, (x, x_dst)).view(
                    -1, self.hidden_channels
                )
            x = self.norms[i](x_skip)
            x = F.elu(x)
            x = self.dropout(x)
        return self.mlp(x)

def cross_entropy(block_outputs, cached_variables, pos_graph, neg_graph):
    block_outputs = DistConvFunction.apply(cached_variables, block_outputs)
    with pos_graph.local_scope():
        pos_graph.ndata['h'] = block_outputs
        pos_graph.apply_edges(dgl.function.u_dot_v('h', 'h', 'score'))
        pos_score = pos_graph.edata['score']
    with neg_graph.local_scope():
        neg_graph.ndata['h'] = block_outputs
        neg_graph.apply_edges(dgl.function.u_dot_v('h', 'h', 'score'))
        neg_score = neg_graph.edata['score']

    score = th.cat([pos_score, neg_score])
    label = th.cat([th.ones_like(pos_score), th.zeros_like(neg_score)]).long()
    loss = F.binary_cross_entropy_with_logits(score, label.float())
    acc = th.sum((score >= 0.5) == (label >= 0.5)) / score.shape[0]
    return loss, acc
