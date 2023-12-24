"""
This flowchart describes the main functional sequence of the provided example.
main
│
├───> Instantiate DataModule
│     │
│     └───> Load dataset
│     │
│     └───> Create train and valid dataloader[HIGHLIGHT]
│           │
│           └───> ItemSampler (Distribute data to minibatchs)
│           │
│           └───> sample_neighbor or sample_layer_neighbor
                  (Sample a subgraph for a minibatch)
│           │
│           └───> fetch_feature (Fetch features for the sampled subgraph)
│
├───> Instantiate GraphSAGE model
│     │
│     ├───> SAGEConvLayer (input to hidden)
│     │
│     └───> SAGEConvLayer (hidden to hidden)
│     │
│     └───> SAGEConvLayer (hidden to output)
│     │
│     └───> DropoutLayer
│
└───> Run
      │
      │
      └───> Trainer[HIGHLIGHT]
            │
            ├───> SAGE.forward (GraphSAGE model forward pass)
            │
            └───> Validate
"""
import argparse

import dgl.graphbolt as gb
import dgl.nn.pytorch as dglnn

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from torchmetrics import Accuracy

from dgl import function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair
from dgl.nn.functional import edge_softmax

class GATv2Conv(nn.Module):
    r"""GATv2 from `How Attentive are Graph Attention Networks?
    <https://arxiv.org/pdf/2105.14491.pdf>`__

    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{ij}^{(l)} W^{(l)}_{right} h_j^{(l)}

    where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and
    node :math:`j`:

    .. math::
        \alpha_{ij}^{(l)} &= \mathrm{softmax_i} (e_{ij}^{(l)})

        e_{ij}^{(l)} &= {\vec{a}^T}^{(l)}\mathrm{LeakyReLU}\left(
            W^{(l)}_{left} h_{i} + W^{(l)}_{right} h_{j}\right)

    Parameters
    ----------
    in_feats : int, or pair of ints
        Input feature size; i.e, the number of dimensions of :math:`h_i^{(l)}`.
        If the layer is to be applied to a unidirectional bipartite graph, `in_feats`
        specifies the input feature size on both the source and destination nodes.
        If a scalar is given, the source and destination node feature size
        would take the same value.
    out_feats : int
        Output feature size; i.e, the number of dimensions of :math:`h_i^{(l+1)}`.
    num_heads : int
        Number of heads in Multi-Head Attention.
    feat_drop : float, optional
        Dropout rate on feature. Defaults: ``0``.
    attn_drop : float, optional
        Dropout rate on attention weight. Defaults: ``0``.
    negative_slope : float, optional
        LeakyReLU angle of negative slope. Defaults: ``0.2``.
    residual : bool, optional
        If True, use residual connection. Defaults: ``False``.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Defaults: ``False``.
    bias : bool, optional
        If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    share_weights : bool, optional
        If set to :obj:`True`, the same matrix for :math:`W_{left}` and :math:`W_{right}` in
        the above equations, will be applied to the source and the target node of every edge.
        (default: :obj:`False`)

    Note
    ----
    Zero in-degree nodes will lead to invalid output value. This is because no message
    will be passed to those nodes, the aggregation function will be applied on empty input.
    A common practice to avoid this is to add a self-loop for each node in the graph if
    it is homogeneous, which can be achieved by:

    >>> g = ... # a DGLGraph
    >>> g = dgl.add_self_loop(g)

    Calling ``add_self_loop`` will not work for some graphs, for example, heterogeneous graph
    since the edge type can not be decided for self_loop edges. Set ``allow_zero_in_degree``
    to ``True`` for those cases to unblock the code and handle zero-in-degree nodes manually.
    A common practise to handle this is to filter out the nodes with zero-in-degree when use
    after conv.

    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import GATv2Conv

    >>> # Case 1: Homogeneous graph
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> g = dgl.add_self_loop(g)
    >>> feat = torch.ones(6, 10)
    >>> gatv2conv = GATv2Conv(10, 2, num_heads=3)
    >>> res = gatv2conv(g, feat)
    >>> res
    tensor([[[ 1.9599,  1.0239],
            [ 3.2015, -0.5512],
            [ 2.3700, -2.2182]],
            [[ 1.9599,  1.0239],
            [ 3.2015, -0.5512],
            [ 2.3700, -2.2182]],
            [[ 1.9599,  1.0239],
            [ 3.2015, -0.5512],
            [ 2.3700, -2.2182]],
            [[ 1.9599,  1.0239],
            [ 3.2015, -0.5512],
            [ 2.3700, -2.2182]],
            [[ 1.9599,  1.0239],
            [ 3.2015, -0.5512],
            [ 2.3700, -2.2182]],
            [[ 1.9599,  1.0239],
            [ 3.2015, -0.5512],
            [ 2.3700, -2.2182]]], grad_fn=<GSpMMBackward>)

    >>> # Case 2: Unidirectional bipartite graph
    >>> u = [0, 1, 0, 0, 1]
    >>> v = [0, 1, 2, 3, 2]
    >>> g = dgl.heterograph({('A', 'r', 'B'): (u, v)})
    >>> u_feat = torch.tensor(np.random.rand(2, 5).astype(np.float32))
    >>> v_feat = torch.tensor(np.random.rand(4, 10).astype(np.float32))
    >>> gatv2conv = GATv2Conv((5,10), 2, 3)
    >>> res = gatv2conv(g, (u_feat, v_feat))
    >>> res
    tensor([[[-0.0935, -0.4273],
            [-1.1850,  0.1123],
            [-0.2002,  0.1155]],
            [[ 0.1908, -1.2095],
            [-0.0129,  0.6408],
            [-0.8135,  0.1157]],
            [[ 0.0596, -0.8487],
            [-0.5421,  0.4022],
            [-0.4805,  0.1156]],
            [[-0.0935, -0.4273],
            [-1.1850,  0.1123],
            [-0.2002,  0.1155]]], grad_fn=<GSpMMBackward>)
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        bias=True,
        share_weights=False,
    ):
        super(GATv2Conv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=bias
            )
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=bias
            )
        else:
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=bias
            )
            if share_weights:
                self.fc_dst = self.fc_src
            else:
                self.fc_dst = nn.Linear(
                    self._in_src_feats, out_feats * num_heads, bias=bias
                )
        self.attn = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=bias
                )
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer("res_fc", None)
        self.activation = activation
        self.share_weights = share_weights
        self.bias = bias
        self.reset_parameters()

    def reset_parameters(self):
        """
        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
        if self.bias:
            nn.init.constant_(self.fc_src.bias, 0)
        if not self.share_weights:
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
            if self.bias:
                nn.init.constant_(self.fc_dst.bias, 0)
        nn.init.xavier_normal_(self.attn, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if self.bias:
                nn.init.constant_(self.res_fc.bias, 0)

    def set_allow_zero_in_degree(self, set_value):
        r"""
        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, get_attention=False):
        r"""
        Description
        -----------
        Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        get_attention : bool, optional
            Whether to return the attention values. Default to False.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                feat_src = self.fc_src(h_src).view(
                    -1, self._num_heads, self._out_feats
                )
                feat_dst = self.fc_dst(h_dst).view(
                    -1, self._num_heads, self._out_feats
                )
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = self.fc_src(h_src).view(
                    -1, self._num_heads, self._out_feats
                )
                if self.share_weights:
                    feat_dst = feat_src
                else:
                    feat_dst = self.fc_dst(h_dst).view(
                        -1, self._num_heads, self._out_feats
                    )
                if graph.is_block:
                    feat_dst = feat_dst[: graph.number_of_dst_nodes()]
                    h_dst = h_dst[: graph.number_of_dst_nodes()]
            graph.srcdata.update(
                {"el": feat_src}
            )  # (num_src_edge, num_heads, out_dim)
            graph.dstdata.update({"er": feat_dst})
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            e = self.leaky_relu(
                graph.edata.pop("e")
            )  # (num_src_edge, num_heads, out_dim)
            e = (
                (e * self.attn).sum(dim=-1).unsqueeze(dim=2)
            )  # (num_edge, num_heads, 1)
            # compute softmax
            e = self.attn_drop(
                edge_softmax(graph, e)
            )  # (num_edge, num_heads)
            graph.edata["a"] = e
            # message passing
            graph.update_all(fn.u_mul_e("el", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(
                    h_dst.shape[0], -1, self._out_feats
                )
                rst = rst + resval
            # activation
            if self.activation:
                rst = self.activation(rst)

        if get_attention:
            graph.edata["a"] = e
        return rst


class GATv2(LightningModule):
    def __init__(self, num_layers, in_dim, n_hidden, n_classes, heads,
                 activation, feat_drop, attn_drop, negative_slope, residual):
        super().__init__()
        self.save_hyperparameters()
        self.layers = nn.ModuleList()
        for l in range(num_layers):
            self.layers.append(
                GATv2Conv(
                    in_dim if l == 0 else n_hidden,
                    n_hidden // heads[l],
                    heads[l],
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    residual,
                    activation,
                    allow_zero_in_degree=True,
                    bias=True,
                    share_weights=False,
                )
            )
        self.final_layer = nn.Linear(n_hidden, n_classes, bias=False)
        self.dropout = nn.Dropout(0.5)
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.train_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=n_classes)

    def forward(self, blocks, x):
        h = x
        for layer, block in zip(self.layers, blocks):
            h = layer(block, h, get_attention=True).flatten(1)
        return self.final_layer(h)

    def log_node_and_edge_counts(self, blocks):
        node_counts = [block.num_src_nodes() for block in blocks] + [
            blocks[-1].num_dst_nodes()
        ]
        edge_counts = [block.num_edges() for block in blocks]
        for i, c in enumerate(node_counts):
            self.log(
                f"num_nodes/{i}",
                float(c),
                prog_bar=True,
                on_step=True,
                on_epoch=False,
            )
            if i < len(edge_counts):
                self.log(
                    f"num_edges/{i}",
                    float(edge_counts[i]),
                    prog_bar=True,
                    on_step=True,
                    on_epoch=False,
                )

    def training_step(self, batch, batch_idx):
        batch.blocks2 = [block.to("cuda") for block in batch.blocks]
        x = batch.node_features["feat"]
        y = batch.labels.to("cuda")
        y_hat = self(batch.blocks2, x)
        loss = F.cross_entropy(y_hat, y)
        self.train_acc(torch.argmax(y_hat, 1), y)
        self.log(
            "train_acc",
            self.train_acc,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )
        self.log_node_and_edge_counts(batch.blocks2)
        return loss

    def validation_step(self, batch, batch_idx):
        batch.blocks2 = [block.to("cuda") for block in batch.blocks]
        x = batch.node_features["feat"]
        y = batch.labels.to("cuda")
        y_hat = self(batch.blocks2, x)
        self.val_acc(torch.argmax(y_hat, 1), y)
        self.log(
            "val_acc",
            self.val_acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log_node_and_edge_counts(batch.blocks2)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=0.001, weight_decay=5e-4
        )
        return optimizer

class BanditFeedbackCallback(Callback):
    def __init__(self):
        super().__init__()
    
    def on_train_batch_end(self, trainer, datamodule, outputs, batch, batch_idx):
        trainer.datamodule.sampler.provide_feedback(batch)
    
    def on_validation_batch_end(self, trainer, datamodule, outputs, batch, batch_idx):
        trainer.datamodule.sampler.provide_feedback(batch)

class DataModule(LightningDataModule):
    def __init__(self, dataset, fanouts, batch_size, num_workers, bandit):
        super().__init__()
        self.fanouts = fanouts
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.feature_store = dataset.feature
        self.graph = dataset.graph
        self.train_set = dataset.tasks[0].train_set
        self.valid_set = dataset.tasks[0].validation_set
        self.num_classes = dataset.tasks[0].metadata["num_classes"]
        self.bandit = bandit

    def create_dataloader(self, node_set, is_train):
        datapipe = gb.ItemSampler(
            node_set,
            batch_size=self.batch_size,
            shuffle=is_train,
            drop_last=is_train,
        )
        sampler = datapipe.sample_bandit_layer_neighbor if self.bandit else datapipe.sample_layer_neighbor
        datapipe = sampler(self.graph, self.fanouts)
        if self.bandit:
            self.sampler = datapipe
        datapipe = datapipe.fetch_feature(self.feature_store, ["feat"])
        dataloader = gb.DataLoader(datapipe, num_workers=self.num_workers)
        return dataloader

    ########################################################################
    # (HIGHLIGHT) The 'train_dataloader' and 'val_dataloader' hooks are
    # essential components of the Lightning framework, defining how data is
    # loaded during training and validation. In this example, we utilize a
    # specialized 'graphbolt dataloader', which are concatenated by a series
    # of datapipes, for these purposes.
    ########################################################################
    def train_dataloader(self):
        return self.create_dataloader(self.train_set, is_train=True)

    def val_dataloader(self):
        return self.create_dataloader(self.valid_set, is_train=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GNN baselines on ogbn-products data with GraphBolt"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="number of GPUs used for computing (default: 1)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="input batch size for training (default: 1024)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="number of epochs to train (default: 40)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="number of workers (default: 0)",
    )
    parser.add_argument("--bandit", action="store_true")
    args = parser.parse_args()

    dataset = gb.BuiltinDataset("ogbn-products").load()
    datamodule = DataModule(
        dataset,
        [10, 10, 10],
        args.batch_size,
        args.num_workers,
        args.bandit
    )
    in_size = dataset.feature.size("node", None, "feat")[0]
    model = GATv2(
        3,
        in_size,
        256,
        datamodule.num_classes,
        [1, 1, 1],
        F.elu,
        0,
        0,
        0.2,
        True
    )

    # Train.
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max")
    early_stopping_callback = EarlyStopping(monitor="val_acc", mode="max")
    ########################################################################
    # (HIGHLIGHT) The `Trainer` is the key Class in lightning, which automates
    # everything after defining `LightningDataModule` and
    # `LightningDataModule`. More details can be found in
    # https://lightning.ai/docs/pytorch/stable/common/trainer.html.
    ########################################################################
    trainer = Trainer(
        accelerator="gpu",
        devices=args.num_gpus,
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, early_stopping_callback] + ([BanditFeedbackCallback()] if args.bandit else []),
    )
    trainer.fit(model, datamodule=datamodule)
