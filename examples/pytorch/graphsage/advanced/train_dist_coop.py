# /*!
#  *   Copyright (c) 2022, NVIDIA Corporation
#  *   All rights reserved.
#  *
#  *   Licensed under the Apache License, Version 2.0 (the "License");
#  *   you may not use this file except in compliance with the License.
#  *   You may obtain a copy of the License at
#  *
#  *       http://www.apache.org/licenses/LICENSE-2.0
#  *
#  *   Unless required by applicable law or agreed to in writing, software
#  *   distributed under the License is distributed on an "AS IS" BASIS,
#  *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  *   See the License for the specific language governing permissions and
#  *   limitations under the License.
#  *
#  * \file train_dist_layer.py
#  * \brief distributed cooperative minibatching example
#  */

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as thd
import torch.distributed.optim
import torchmetrics.functional as MF
from torch.utils.tensorboard import SummaryWriter
import dgl
from dgl.contrib.dist_sampling import DistGraph, DistSampler, metis_partition, uniform_partition, uniform_partition_balanced, reorder_graph_wrapper
from dgl.transforms.functional import remove_self_loop
import argparse
import sys
import os
import glob
import math
from contextlib import nullcontext
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from load_graph import load_reddit, load_ogb, load_mag240m, to_bidirected_with_reverse_mapping
from dist_model import SAGE, RGAT, RGCN, cross_entropy

import nvtx

def producer(args, g, idxs, reverse_eids, device, prefetch_edge_feats=[]):
    fanouts = [int(_) for _ in args.fan_out.split(',')]

    if args.sampler == 'labor':
        sampler = DistSampler(g, dgl.dataloading.LaborSampler, fanouts, ['features'], prefetch_edge_feats, [] if args.edge_pred else ['labels'], importance_sampling=args.importance_sampling, layer_dependency=args.layer_dependency, batch_dependency=args.batch_dependency)
    else:
        sampler = DistSampler(g, dgl.dataloading.NeighborSampler, fanouts, ['features'], prefetch_edge_feats, [] if args.edge_pred else ['labels'])
    unbiased_sampler = DistSampler(g, dgl.dataloading.NeighborSampler, fanouts if True else [-1] * len(fanouts), ['features'], prefetch_edge_feats, [] if args.edge_pred else ['labels'])
    if args.edge_pred:
        sampler = dgl.dataloading.as_edge_prediction_sampler(sampler, exclude='reverse_id', reverse_eids=reverse_eids,
                    negative_sampler=dgl.dataloading.negative_sampler.Uniform(1))
    it = 0
    outputs = [None, None]
    num_itemss = [idx.shape[0] if not args.edge_pred else g.g.num_edges() for idx in idxs]
    total_itemss = th.tensor(num_itemss, device=device)
    thd.all_reduce(total_itemss, thd.ReduceOp.SUM, g.comm)
    num_iterss = total_itemss // (g.world_size * args.batch_size)
    events = [[th.cuda.Event(enable_timing=True) for _ in range(3)] for _ in range(2)]
    for epoch in range(args.num_epochs):
        with nvtx.annotate("epoch: {}".format(epoch), color="orange"):
            for dataloader_idx, (idx, num_items, num_iters) in enumerate(zip(idxs, num_itemss, num_iterss)):
                perm = th.randperm(num_items, device=device) if args.batch_size < num_items else th.arange(num_items, device=device)
                for j in range(num_iters):
                    i = slice(j * num_items // num_iters, (j + 1) * num_items // num_iters)
                    with nvtx.annotate("iteration: {}".format(it), color="yellow"):
                        seeds = idx[perm[i]] if not args.edge_pred else perm[i]
                        events[it % 2][0].record()
                        out = sampler.sample(g.g, seeds.to(device)) if dataloader_idx < 2 else unbiased_sampler.sample(g.g, seeds.to(device))
                        events[it % 2][1].record()
                        wait = out[-1][0].slice_features(out[-1][0])
                        out[-1][-1].slice_labels(out[-1][-1])
                        for block in out[-1]:
                            block.slice_edges(block)
                        events[it % 2][2].record()
                        outputs[it % 2] = [dataloader_idx, it, epoch, out, wait]
                        it += 1
                        if it > 1:
                            out = outputs[it % 2]
                            out[-1]()
                            yield out[:-1] + [events[it % 2]]
    it += 1
    out = outputs[it % 2]
    out[-1]()
    yield out[:-1] + [events[it % 2]]

def train(local_rank, local_size, group_rank, world_size, g, parts, num_classes, args):
    th.set_num_threads(os.cpu_count() // local_size)
    th.cuda.set_device(local_rank)
    device = th.cuda.current_device()
    global_rank = group_rank * local_size + local_rank
    thd.init_process_group('nccl', 'env://', world_size=world_size, rank=global_rank)

    g = DistGraph(g, parts, args.replication, args.uva_data, args.uva_ndata.split(','), cache_size=args.cache_size, compress=False)

    train_idx = (th.nonzero(g.dstdata['train_mask'], as_tuple=True)[0] + g.l_offset).to(device, g.g.idtype)
    val_idx = (th.nonzero(g.dstdata['val_mask'], as_tuple=True)[0] + g.l_offset).to(device, g.g.idtype)
    test_idx = (th.nonzero(~(g.dstdata['train_mask'] | g.dstdata['val_mask']), as_tuple=True)[0] + g.l_offset).to(device, g.g.idtype)
    reverse_eids = None if 'is_reverse' not in g.g.edata else th.nonzero(g.g.edata['is_reverse'], as_tuple=True)[0]
    
    num_layers = args.num_layers
    num_hidden = args.num_hidden

    if args.dataset in ['ogbn-mag240M']:
        if args.model == 'rgat':
            model = RGAT(
                g.dstdata['features'].shape[1],
                num_classes,
                num_hidden,
                5,
                num_layers,
                4,
                args.dropout,
                args.model == 'rgat',
                args.replication==1
            ).to(device)
        else:
            model = RGCN([g.dstdata['features'].shape[1]] + [num_hidden for _ in range(num_layers - 1)] + [num_classes], 5, 2, args.dropout, args.replication==1).to(device)
        # convert BN to SyncBatchNorm. see https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    else:
        model = SAGE([g.dstdata['features'].shape[1]] + [num_hidden for _ in range(num_layers - 1)] + [num_classes], args.dropout, args.replication==1).to(device)

    model = nn.parallel.DistributedDataParallel(model.to(device), device_ids=[local_rank], output_device=local_rank)
    opt = th.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    sched = th.optim.lr_scheduler.StepLR(opt, step_size=25, gamma=0.25)

    logdir = os.path.join(args.logdir, '{}_{}_{}_{}_{}'.format(args.dataset, args.sampler, args.importance_sampling, args.layer_dependency, args.batch_dependency))
    dirs = glob.glob('{}/*'.format(logdir))
    version = (1 + max([int(os.path.split(x)[-1].split('_')[-1]) for x in dirs])) if len(dirs) > 0 else 0
    logdir = '{}/version_{}_{}'.format(logdir, global_rank, version)

    thd.barrier(g.comm)
    
    writer = SummaryWriter(logdir)
    
    st, end = th.cuda.Event(enable_timing=True), th.cuda.Event(enable_timing=True)
    fw_st = th.cuda.Event(enable_timing=True)
    st.record()
    last_epoch = 0
    val_accs = [0, 0]
    val_losses = [0, 0]
    cnts = [0, 0]
    for out in producer(args, g, ([train_idx, val_idx] if not args.edge_pred else [None]), reverse_eids, device, [dgl.ETYPE] if args.dataset in ['ogbn-mag240M'] else []):
        dataloader_idx, it, epoch = out[:3]
        events = out[4]
        out = out[3]
        input_nodes = out[0]
        blocks = out[-1]
        block_stats = [(block.num_src_nodes(), block.num_dst_nodes(), block.num_edges()) for block in blocks]
        writer.add_scalar('dataloader_idx', dataloader_idx, it)
        for i, mfg in enumerate(blocks):
            writer.add_scalar('num_src_nodes/{}'.format(i), mfg.num_src_nodes(), it)
            writer.add_scalar('num_edges/{}'.format(i), mfg.num_edges(), it)
            writer.add_scalar('num_nodes/{}'.format(i), mfg.cached_variables[3].shape[0], it)
        writer.add_scalar('num_nodes/{}'.format(len(blocks)), blocks[-1].num_dst_nodes(), it)
        x = blocks[0].srcdata.pop('features')
        if not args.edge_pred:
            y = blocks[-1].dstdata.pop('labels')
        model.train(dataloader_idx == 0)
        is_grad_enabled = nullcontext() if model.training else torch.no_grad()
        fw_st.record()
        with nvtx.annotate("forward", color="purple"), is_grad_enabled:
            y_hat = model(blocks, x)
            if args.edge_pred:
                loss, acc = cross_entropy(y_hat, blocks[-1].cached_variables2, out[1], out[2])
            else:
                loss = F.cross_entropy(y_hat, y)
        if model.training:
            with nvtx.annotate("backward", color="purple"):
                opt.zero_grad()
                loss.backward()
            with nvtx.annotate("optimizer", color="purple"):
                opt.step()
        if not args.edge_pred:
            with nvtx.annotate("accuracy", color="purple"):
                acc = MF.accuracy(y_hat, y)
        end.record()
        if dataloader_idx >= 1:
            val_accs[dataloader_idx - 1] += acc.item() * y_hat.shape[0]
            val_losses[dataloader_idx - 1] += loss.item() * y_hat.shape[0]
            cnts[dataloader_idx - 1] += y_hat.shape[0]
        mem = th.cuda.max_memory_allocated() >> 20
        if epoch != last_epoch:
            sched.step()
            last_epoch = epoch
            if not args.edge_pred:
                for k in range(1):
                    writer.add_scalar('val_acc/dataloader_idx_{}'.format(k), val_accs[k] / cnts[k], it)
                    writer.add_scalar('val_loss/dataloader_idx_{}'.format(k), val_losses[k] / cnts[k], it)
                    val_losses[k] = val_accs[k] = cnts[k] = 0
        end.synchronize()
        events[2].synchronize()
        iter_time = st.elapsed_time(end)
        writer.add_scalar('time/iter', iter_time, it)
        writer.add_scalar('time/sampling', events[0].elapsed_time(events[1]), it)
        writer.add_scalar('time/feature_copy', events[1].elapsed_time(events[2]), it)
        writer.add_scalar('time/forward_backward', fw_st.elapsed_time(end), it)
        writer.add_scalar('epoch', epoch, it)
        writer.add_scalar('cache_miss', x.cache_miss, it)
        if model.training:
            writer.add_scalar('train_loss_step', loss.item(), it)
            writer.add_scalar('train_acc_step', acc.item(), it)
        print('rank: {}, it: {}, dataloader_idx: {}, Loss: {:.4f}, Acc: {:.4f}, GPU Mem: {:.0f} MB, time: {:.3f}ms, stats: {}'.format(global_rank, it, dataloader_idx, loss.item(), acc.item(), mem, iter_time, block_stats))
        st, end = end, st
    
    writer.close()

    thd.barrier(g.comm)

def main(args):
    # use all available CPUs
    th.set_num_threads(os.cpu_count())
    # use all available GPUs
    local_size = th.cuda.device_count()
    group_rank = int(os.environ["GROUP_RANK"])
    num_groups = int(os.environ["WORLD_SIZE"])
    world_size = local_size * num_groups
    if args.replication <= 0:
        args.replication = world_size

    undirected_suffix = '-undirected' if args.undirected else ''

    fn_list = [fn for fn in os.listdir(args.root_dir) if fn.startswith(args.dataset + undirected_suffix)]
    if fn_list:
        gs, ls = dgl.load_graphs(os.path.join(args.root_dir, fn_list[0]))
        g = gs[0]
        n_classes = ls['n_classes'][0].item()
        if 'etype' in g.edata:
            g.edata[dgl.ETYPE] = g.edata.pop('etype')
    else:
        if args.dataset in ['ogbn-mag240M']:
            g, n_classes = load_mag240m(args.root_dir)
        else:
            g, n_classes = load_ogb(args.dataset, args.root_dir) if args.dataset.startswith("ogbn") else load_reddit()

        if args.undirected:
            g, reverse_eids = to_bidirected_with_reverse_mapping(remove_self_loop(g))
            g.edata['is_reverse'] = th.zeros(g.num_edges(), dtype=th.bool)
            g.edata['is_reverse'][reverse_eids] = True

        if args.partition == 'metis':
            parts = metis_partition(g, world_size)
        elif args.partition == 'random':
            th.manual_seed(0)
            parts = uniform_partition(g, world_size)
        elif args.partition == 'random-balanced':
            th.manual_seed(0)
            parts = uniform_partition_balanced(g, 1680 * world_size // math.gcd(world_size, 1680))
        else:
            parts = uniform_partition(g, world_size, False)
        g = reorder_graph_wrapper(g, parts)

        dgl.save_graphs(os.path.join(args.root_dir, '{}_{}_{}'.format(args.dataset + undirected_suffix, g.number_of_nodes(), g.number_of_edges())), [g], {'n_classes': th.tensor([n_classes])})

    cast_to_int = max(g.num_nodes(), g.num_edges()) <= 2e9
    if cast_to_int:
        g = g.int()
    g.create_formats_()
    
    parts = [th.arange(i * g.num_nodes() // world_size, (i + 1) * g.num_nodes() // world_size) for i in range(world_size)]

    args.dataset += undirected_suffix

    th.multiprocessing.spawn(train, args=(local_size, group_rank, world_size, g, [len(part) for part in parts], n_classes, args), nprocs=local_size)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, default='reddit')
    argparser.add_argument('--num-epochs', type=int, default=2)
    argparser.add_argument('--num-steps', type=int, default=5000)
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--fan-out', type=str, default='10,10,10')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--lr', type=float, default=0.001)
    argparser.add_argument('--model', type=str, default='rgcn')
    argparser.add_argument('--sampler', type=str, default='labor')
    argparser.add_argument('--importance-sampling', type=int, default=0)
    argparser.add_argument('--layer-dependency', action='store_true')
    argparser.add_argument('--batch-dependency', type=int, default=1)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--edge-pred', action='store_true')
    argparser.add_argument('--partition', type=str, default='random-balanced')
    argparser.add_argument('--undirected', action='store_true')
    argparser.add_argument('--replication', type=int, default=0)
    argparser.add_argument('--root-dir', type=str, default='/localscratch/ogb')
    argparser.add_argument('--uva-data', action='store_true')
    argparser.add_argument('--uva-ndata', type=str, default='')
    argparser.add_argument('--cache-size', type=int, default=0)
    argparser.add_argument('--logdir', type=str, default='tb_logs')
    args = argparser.parse_args()
    main(args)
