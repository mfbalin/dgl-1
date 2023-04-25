#!/usr/bin/env python
# coding: utf-8
import argparse
import sys
import os
from collections import OrderedDict

import dgl

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as thd
import torchmetrics.functional as MF
from torch.utils.tensorboard import SummaryWriter
import tqdm
from torch.nn.parallel import DistributedDataParallel

from dgl.contrib.gpu_cache import GPUCache

import glob
from itertools import chain
from contextlib import nullcontext

from dist_model import SAGE, RGAT, RGCN
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from load_graph import load_reddit, load_ogb, load_mag240m

from itertools import chain, repeat

def cuda_index_tensor(tensor, idx):
    assert(idx.device != torch.device('cpu'))
    if tensor.is_pinned():
        return dgl.utils.gather_pinned_tensor_rows(tensor, idx)
    else:
        return tensor[idx.long()]

def train(proc_id, n_gpus, args, g, num_classes, devices):
    torch.set_num_threads(os.cpu_count() // n_gpus)
    device = devices[proc_id]
    torch.cuda.set_device(device)
    world_size = n_gpus
    thd.init_process_group('nccl', 'env://', world_size=world_size, rank=proc_id)

    train_idx = (torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]).to(device, g.idtype)
    val_idx = (torch.nonzero(g.ndata['val_mask'], as_tuple=True)[0]).to(device, g.idtype)
    test_idx = (torch.nonzero(~(g.ndata['train_mask'] | g.ndata['val_mask']), as_tuple=True)[0]).to(device, g.idtype)

    num_layers = args.num_layers

    print("Initializing dataloader...")
    fanouts = [int(_) for _ in args.fan_out.split(',')]
    prefetch_edge_feats = [dgl.ETYPE] if args.dataset in ['ogbn-mag240M'] else []
    if args.sampler == 'labor':
        sampler = dgl.dataloading.LaborSampler(fanouts, importance_sampling=args.importance_sampling, layer_dependency=args.layer_dependency, batch_dependency=args.batch_dependency)
    else:
        sampler = dgl.dataloading.NeighborSampler(fanouts)

    ndata = {k: g.ndata.pop(k) for k in ['features', 'labels']}
    edata = {k: g.edata.pop(k) for k in prefetch_edge_feats}
    pindata = {k: dgl.utils.pin_memory_inplace(v) for k, v in chain(ndata.items(), edata.items())}

    for data in [g.ndata, g.edata]:
        for k in list(data):
            data.pop(k)

    train_dataloader = dgl.dataloading.DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=args.batch_size,
        use_uva=True,
        use_ddp=True,
        num_workers=0,
        shuffle=True,
        drop_last=True)

    valid_dataloader = dgl.dataloading.DataLoader(
        g,
        val_idx,
        sampler,
        device=device,
        batch_size=args.batch_size,
        use_uva=True,
        use_ddp=True,
        num_workers=0,
        shuffle=False)

    num_hidden = args.num_hidden

    print("Initializing model...")
    if args.dataset in ['ogbn-mag240M']:
        if args.model == 'rgat':
            model = RGAT(
                ndata['features'].shape[1],
                num_classes,
                num_hidden,
                5,
                num_layers,
                4,
                args.dropout,
                args.model == 'rgat',
                True
            ).to(device)
        else:
            model = RGCN([ndata['features'].shape[1]] + [num_hidden for _ in range(num_layers - 1)] + [num_classes], 5, 2, args.dropout, True).to(device)
        # convert BN to SyncBatchNorm. see https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    else:
        model = SAGE([ndata['features'].shape[1]] + [num_hidden for _ in range(num_layers - 1)] + [num_classes], args.dropout, True).to(device)

    model = DistributedDataParallel(model.to(device), device_ids=[device], output_device=device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=25, gamma=0.25)

    caches = {} if args.cache_size <= 0 else {key: GPUCache(args.cache_size, ndata[key].shape[1], g.idtype) for key in args.uva_ndata.split(',')}

    it = 0

    logdir = os.path.join(args.logdir, '{}_{}_{}_{}_{}'.format(args.dataset, args.sampler, args.importance_sampling, args.layer_dependency, args.batch_dependency))
    dirs = glob.glob('{}/*'.format(logdir))
    version = (1 + max([int(os.path.split(x)[-1].split('_')[-1]) for x in dirs])) if len(dirs) > 0 else 0
    logdir = '{}/version_{}_{}'.format(logdir, proc_id, version)

    thd.barrier()

    writer = SummaryWriter(logdir)
    
    st, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    fw_st = torch.cuda.Event(enable_timing=True)
    st.record()
    val_accs = [0, 0]
    val_losses = [0, 0]
    cnts = [0, 0]
    events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]

    for epoch in range(args.num_epochs):
        def process_blocks(blocks):
            for k in ['features']:
                cache_miss = 1
                if k in caches:
                    cache = caches[k]
                    tensor, missing_index, missing_keys = cache.query(input_nodes)
                    missing_values = cuda_index_tensor(ndata[k], missing_keys)
                    cache.replace(missing_keys, missing_values.to(torch.float))
                    cache_miss = missing_keys.shape[0] / input_nodes.shape[0]
                    tensor[missing_index] = missing_values.to(tensor.dtype)
                else:
                    tensor = cuda_index_tensor(ndata[k], blocks[0].srcdata[dgl.NID]).to(torch.float)
                tensor.cache_miss = cache_miss
                blocks[0].srcdata[k] = tensor
            for k in prefetch_edge_feats:
                for block in blocks:
                    block.edata[k] = cuda_index_tensor(edata[k], block.edata[dgl.EID])
            blocks[-1].dstdata['labels'] = cuda_index_tensor(ndata['labels'], blocks[-1].dstdata[dgl.NID])
            return blocks
        
        events[0].record()

        for dataloader_idx, (input_nodes, output_nodes, blocks) in chain(zip(repeat(0), train_dataloader), zip(repeat(1), valid_dataloader)):
            events[1].record()
            block_stats = [(block.num_src_nodes(), block.num_dst_nodes(), block.num_edges()) for block in blocks]
            blocks = process_blocks(blocks)
            events[2].record()
            x = blocks[0].srcdata.pop('features')
            y = blocks[-1].dstdata.pop('labels')
            writer.add_scalar('dataloader_idx', dataloader_idx, it)
            for i, mfg in enumerate(blocks):
                writer.add_scalar('num_nodes/{}'.format(i), mfg.num_src_nodes(), it)
                writer.add_scalar('num_edges/{}'.format(i), mfg.num_edges(), it)
            writer.add_scalar('num_nodes/{}'.format(len(blocks)), blocks[-1].num_dst_nodes(), it)
            model.train(dataloader_idx == 0)
            is_grad_enabled = nullcontext() if model.training else torch.no_grad()
            thd.barrier()
            fw_st.record()
            with is_grad_enabled:
                y_hat = model(blocks, x)
                loss = F.cross_entropy(y_hat, y)
            if model.training:
                opt.zero_grad()
                loss.backward()
                opt.step()
            acc = MF.accuracy(y_hat, y)
            end.record()
            if dataloader_idx >= 1:
                val_accs[dataloader_idx - 1] += acc.item() * y_hat.shape[0]
                val_losses[dataloader_idx - 1] += loss.item() * y_hat.shape[0]
                cnts[dataloader_idx - 1] += y_hat.shape[0]
            mem = torch.cuda.max_memory_allocated() >> 20
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
            print('rank: {}, it: {}, dataloader_idx: {}, Loss: {:.4f}, Acc: {:.4f}, GPU Mem: {:.0f} MB, time: {:.3f}ms, stats: {}'.format(proc_id, it, dataloader_idx, loss.item(), acc.item(), mem, iter_time, block_stats))
            st, end = end, st
            it += 1
            events[0].record()

        sched.step()
        for k in range(1):
            writer.add_scalar('val_acc/dataloader_idx_{}'.format(k), val_accs[k] / cnts[k], it)
            writer.add_scalar('val_loss/dataloader_idx_{}'.format(k), val_losses[k] / cnts[k], it)
            val_losses[k] = val_accs[k] = cnts[k] = 0
    
    writer.close()

    thd.barrier()

def test(args, dataset, g, split_idx, paper_offset):
    print("Loading masks and labels...")
    valid_idx = split_idx['valid']
    test_idx = split_idx['test']
    label = dataset.paper_label

    print("Initializing data loader...")
    sampler = dgl.dataloading.MultiLayerNeighborSampler([160, 160])
    valid_collator = ExternalNodeCollator(
        g, valid_idx, sampler, paper_offset, feats, label
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_collator.dataset,
        batch_size=16,
        shuffle=False,
        drop_last=False,
        collate_fn=valid_collator.collate,
        num_workers=2,
    )
    test_collator = ExternalNodeCollator(
        g, test_idx, sampler, paper_offset, feats, label
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_collator.dataset,
        batch_size=16,
        shuffle=False,
        drop_last=False,
        collate_fn=test_collator.collate,
        num_workers=4,
    )

    print("Loading model...")
    model = RGAT(
        dataset.num_paper_features,
        dataset.num_classes,
        1024,
        5,
        2,
        4,
        0.5,
        "paper",
    ).cuda()

    # load ddp's model parameters, we need to remove the name of 'module.'
    state_dict = torch.load(args.model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    model.eval()
    correct = total = 0
    for i, (input_nodes, output_nodes, mfgs) in enumerate(
        tqdm.tqdm(valid_dataloader)
    ):
        with torch.no_grad():
            mfgs = [g.to("cuda") for g in mfgs]
            x = mfgs[0].srcdata["x"]
            y = mfgs[-1].dstdata["y"]
            y_hat = model(mfgs, x)
            correct += (y_hat.argmax(1) == y).sum().item()
            total += y_hat.shape[0]
    acc = correct / total
    print("Validation accuracy:", acc)
    evaluator = MAG240MEvaluator()
    y_preds = []
    for i, (input_nodes, output_nodes, mfgs) in enumerate(
        tqdm.tqdm(test_dataloader)
    ):
        with torch.no_grad():
            mfgs = [g.to("cuda") for g in mfgs]
            x = mfgs[0].srcdata["x"]
            y = mfgs[-1].dstdata["y"]
            y_hat = model(mfgs, x)
            y_preds.append(y_hat.argmax(1).cpu())
    evaluator.save_test_submission(
        {"y_pred": torch.cat(y_preds)}, args.submission_path
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, default='reddit')
    argparser.add_argument('--root-dir', type=str, default='/localscratch/ogb')
    argparser.add_argument("--num-epochs", type=int, default=100, help="Number of epochs.")
    argparser.add_argument('--logdir', type=str, default='tb_logs')
    argparser.add_argument("--submission-path", type=str, default="./results_ddp", help="Submission directory.")
    argparser.add_argument('--undirected', action='store_true')
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--model', type=str, default='rgcn')
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--cache-size', type=int, default=0)
    argparser.add_argument('--uva-ndata', type=str, default='')
    argparser.add_argument('--sampler', type=str, default='labor')
    argparser.add_argument('--importance-sampling', type=int, default=0)
    argparser.add_argument('--layer-dependency', action='store_true')
    argparser.add_argument('--batch-dependency', type=int, default=1)
    argparser.add_argument('--fan-out', type=str, default='10,10,10')
    args = argparser.parse_args()

    devices = list(range(torch.cuda.device_count()))
    n_gpus = len(devices)

    if n_gpus < 1:
        print("make sure the number of gpus greater than 0!")
        sys.exit()
    
    torch.set_num_threads(os.cpu_count())

    if args.dataset in ['ogbn-mag240M']:
        g, n_classes = load_mag240m(args.root_dir)
        g.ndata['features'].copy_(g.ndata['features'])
    else:
        g, n_classes = load_ogb(args.dataset, args.root_dir) if args.dataset.startswith("ogbn") else load_reddit()

    if args.undirected:
        # g, reverse_eids = to_bidirected_with_reverse_mapping(dgl.remove_self_loop(g))
        # g.edata['is_reverse'] = torch.zeros(g.num_edges(), dtype=torch.bool)
        # g.edata['is_reverse'][reverse_eids] = True
        src, dst = g.all_edges()
        g.add_edges(dst, src)
    
    g.create_formats_()
    os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count() // 2 // n_gpus)

    mp.spawn(
        train,
        args=(n_gpus, args, g, n_classes, devices),
        nprocs=n_gpus,
    )

    # test(args, dataset, g, split_idx, labels, paper_offset)