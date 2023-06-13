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
#  * \brief distributed cooperative minibatching implementation
#  */

from random import shuffle
import time
from datetime import timedelta

import numpy as np

import nvtx

import torch as th
import torch.distributed as thd

from ...subgraph import in_subgraph
from ...base import NID, EID

from ...utils import gather_pinned_tensor_rows, pin_memory_inplace

from ...convert import graph
from ...heterograph_index import create_unitgraph_from_coo
from ...sparse import lighter_gspmm

from ...partition import metis_partition_assignment, make_symmetric_hetero, metis_partition_hetero

from ... import backend as F
from ... import ndarray as nd

from ...dataloading.base import Sampler

from ..gpu_cache import GPUCache

def reorder_graph_wrapper(g, parts):
    return g.reorder_graph(node_permute_algo='custom', edge_permute_algo='dst', 
    store_ids=False, permute_config={'nodes_perm': th.cat(parts)})

def _split_idx(idx, n, random):
    N = idx.shape[0]
    perm = th.randperm(N, device=idx.device) if random else th.arange(N, device=idx.device)
    return [idx[perm[i * N // n: (i + 1) * N // n]] for i in range(n)]

def uniform_partition(g, n_procs, random=True):
    N = g.num_nodes()
    train_idx, = th.nonzero(g.ndata['train_mask'], as_tuple=True) if 'train_mask' in g.ndata else th.zeros([0])
    train_part = _split_idx(train_idx, n_procs, random)

    val_idx, = th.nonzero(g.ndata['val_mask'], as_tuple=True) if 'val_mask' in g.ndata else th.zeros([0])
    val_part = _split_idx(val_idx, n_procs, random)

    test_idx, = th.nonzero(g.ndata['test_mask'], as_tuple=True) if 'test_mask' in g.ndata else th.zeros([0])
    test_part = _split_idx(test_idx, n_procs, random)

    other_mask = th.zeros([N], device=g.device, dtype=th.bool)
    other_mask[train_idx] = True
    other_mask[val_idx] = True
    other_mask[test_idx] = True

    other_idx, = th.nonzero(~other_mask, as_tuple=True)
    other_part = _split_idx(other_idx, n_procs, random)

    out = [[] for i in range(n_procs)]
    for parts in [train_part, val_part, test_part, other_part]:
        for j, part in enumerate(parts):
            out[j].append(part)
    return [th.cat(parts, dim=0) for parts in out]

def metis_partition_old(g, n_procs):
    n_types = th.zeros([g.num_nodes()], dtype=th.int64)
    for i, mask in enumerate(['train_mask', 'val_mask']):
        if mask in g.ndata:
            n_types[g.ndata[mask]] = i + 1
    parts = metis_partition_assignment(g, n_procs, balance_ntypes=n_types, balance_edges=True)
    idx = th.argsort(parts)
    partition = th.searchsorted(parts[idx], th.arange(0, n_procs + 1))
    parts = [idx[partition[i]: partition[i + 1]] for i in range(n_procs)]
    shuffle(parts)
    return parts

def metis_partition(g, n_procs):
    assert (g.idtype == F.int64), "IdType of graph is required to be int64."
    start = time.time()
    vwgt = []
    vwgt.append(F.ones(g.num_nodes(), F.int64, F.cpu()))
    vwgt.append(F.astype(g.in_degrees(), F.int64))
    vwgt.append(F.astype(g.ndata['train_mask'], F.int64))
    vwgt.append(F.astype(g.ndata['val_mask'], F.int64))

    # The vertex weights have to be stored in a vector.
    vwgt = F.stack(vwgt, 1)
    shape = (np.prod(F.shape(vwgt),),)
    vwgt = F.reshape(vwgt, shape)
    vwgt = F.to_dgl_nd(vwgt)
    print("Construct multi-constraint weights: {:.3f} seconds".format(
            time.time() - start))

    start = time.time()
    sym_g = make_symmetric_hetero(g)
    print("Convert a graph into a bidirected graph: {:.3f} seconds".format(
            time.time() - start))

    start = time.time()
    parts = metis_partition_hetero(sym_g, n_procs, vwgt, "k-way", "cut")
    print("Metis partitioning: {:.3f} seconds".format(time.time() - start))

    idx = th.argsort(parts)
    partition = th.searchsorted(parts[idx], th.arange(0, n_procs + 1))
    parts = [idx[partition[i]: partition[i + 1]] for i in range(n_procs)]
    shuffle(parts)
    return parts

class DistConvFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, cached_variables, h):
        request_counts, requested_nodes, requested_sizes, seed_nodes, inv_ids, perm, distg_handle = ctx.cached_variables = cached_variables
        h = distg_handle.pull(h, request_counts, requested_nodes, requested_sizes, seed_nodes, inv_ids)
        if perm == slice(None):
            return h
        else:
            a = th.empty_like(h)
            a[perm] = h
            return a

    @staticmethod
    def backward(ctx, grad_output):
        request_counts, requested_nodes, requested_sizes, seed_nodes, inv_ids, perm, distg_handle = ctx.cached_variables
        del ctx.cached_variables
        g_h = distg_handle.rpull(grad_output[perm], request_counts, requested_sizes, seed_nodes, inv_ids)
        return None, g_h

class DistConv(th.nn.Module):
    def __init__(self, conv, pull=True):
        super().__init__()
        self.pull = pull
        self.layer = conv
    
    def forward(self, block, h, *args):
        if self.pull:
            h = DistConvFunction.apply(block.cached_variables, h)
        return self.layer(block, h, *args)

class DistSampler(Sampler):
    def __init__(self, g, sampler_t, fanouts, prefetch_node_feats=[], prefetch_edge_feats=[], prefetch_labels=[], **kwargs):
        super().__init__()
        self.g = g
        self.samplers = [sampler_t([fanout], sort_src=True, output_device=self.g.device, **kwargs) for fanout in fanouts]
        self.prefetch_node_feats = prefetch_node_feats
        self.prefetch_edge_feats = prefetch_edge_feats
        self.prefetch_labels = prefetch_labels
        self.output_device = self.g.device

        for i, sampler in enumerate(reversed(self.samplers)):
            if hasattr(sampler, 'set_seed'):
                sampler.set_seed(self.g.random_seed.item() + (0 if sampler.layer_dependency else i))
    
    def sample(self, g, seed_nodes, exclude_eids=None):
        # ignore g as we already store DistGraph
        return self.g.sample_blocks(seed_nodes, self.samplers, exclude_eids, self.prefetch_node_feats, self.prefetch_edge_feats, self.prefetch_labels)

def cuda_index_tensor(tensor, idx):
    assert(idx.device != th.device('cpu'))
    if tensor.is_pinned():
        bits = th.finfo(tensor.dtype).bits if tensor.is_floating_point() else th.iinfo(tensor.dtype).bits
        vtensor = tensor.view(th.int32) if tensor.dim() >= 2 and bits * tensor.shape[-1] % 32 == 0 else tensor
        return gather_pinned_tensor_rows(vtensor, idx).view(tensor.dtype)
    else:
        return tensor[idx.long()]

class DistGraph(object):
    '''Distributed Graph object for GPUs
    
    We assume that torch.cuda.device() is called to set the GPU for the all processes
    We will rely on torch.cuda.current_device() to get the device.
    '''
    def __init__(self, g, g_parts, replication=0, uva_data=False, uva_ndata=[], cache_size=0, compress=False):

        assert(thd.is_available()
            and thd.is_initialized()
            and thd.is_nccl_available())

        self.rank = thd.get_rank()
        self.world_size = thd.get_world_size()
        self.group_size = self.world_size if replication <= 0 else replication
        self.num_groups = self.world_size // self.group_size
        self.l_rank = self.rank % self.group_size
        self.group = self.rank // self.group_size
        self.group_start = self.group * self.group_size
        self.group_end = (self.group + 1) * self.group_size
        self.compress = compress

        assert(self.world_size % self.group_size == 0)
        assert(self.world_size == len(g_parts))
        
        self.device = th.cuda.current_device()
        cpu_device = th.device('cpu')
        
        assert(g.device == cpu_device)

        self.comm = None
        self.l_comm = self.comm
        if self.group_size < self.world_size:
            pg_options = th._C._distributed_c10d.ProcessGroupNCCL.Options()
            pg_options.is_high_priority_stream = True
            pg_options._timeout = timedelta(minutes=1)
            self.l_comms = [thd.new_group(ranks=range(group * self.group_size, (group + 1) * self.group_size), backend='nccl', pg_options=pg_options) for group in range(self.num_groups)]
            self.l_comm = self.l_comms[self.group]

        parts = [sum(g_parts[i * self.num_groups: (i + 1) * self.num_groups]) for i in range(self.group_size)]

        node_ranges = th.cumsum(th.tensor([0] + parts, device=cpu_device), dim=0)

        num_dst_nodes = node_ranges[self.l_rank + 1] - node_ranges[self.l_rank]

        if uva_data:
            self.g = g
            # self.g.create_formats_()
            self.g.pin_memory_()
            self.node_ranges = th.tensor([0] * (self.group * self.group_size) + node_ranges.tolist() + [node_ranges[-1].item()] * ((self.num_groups - self.group - 1) * self.group_size), device=self.device)

            g_node_ranges = []
            cnts = [0] * self.group_size
            permute = []
            for i in range(self.num_groups):
                for j in range(self.group_size):
                    rank = j * self.num_groups + i
                    permute.append(rank)
                    g_node_ranges.append(node_ranges[j].item() + cnts[j])
                    cnts[j] += g_parts[rank]
            g_node_ranges.append(node_ranges[-1].item())
            inv_permute = sorted(range(len(permute)), key=permute.__getitem__)        
            self.g_node_ranges = th.tensor(g_node_ranges, device=self.device)[inv_permute + [-1]]
            self.permute = th.tensor(permute, device=self.device)
            self.inv_permute = th.tensor(inv_permute, device=self.device)

            self.pr = self.node_ranges
            assert self.pr[self.rank + 1] - self.pr[self.rank] == num_dst_nodes
            self.g_pr = self.g_node_ranges
            
            self.l_offset = self.g_pr[permute[self.rank]].item()

            g_NID = slice(self.g_pr[self.permute[self.rank]], self.g_pr[self.permute[self.rank] + 1])

            self.dstdata = {k: v[g_NID] for k, v in self.g.ndata.items()}
        else:
            storage_device = self.device if not uva_data else cpu_device
            g_ndata = {k: g.ndata.pop(k) for k in list(g.ndata)}
            g_edata = {k: g.edata.pop(k) for k in list(g.edata)}

            my_g = in_subgraph(g, th.arange(node_ranges[self.l_rank], node_ranges[self.l_rank + 1], dtype=g.idtype))

            if compress:
                max_num_dst_nodes = max(parts)
                self.log_pow_of_two = min(i for i in range(60) if 2 ** i >= max_num_dst_nodes)
                self.pow_of_two = 2 ** self.log_pow_of_two

                src, dst = my_g.edges()

                src_part = th.searchsorted(node_ranges, src + 1) - 1
                dst_part = self.l_rank # th.searchsorted(node_ranges, dst + 1) - 1

                new_src = src - node_ranges[src_part] + src_part * self.pow_of_two
                new_dst = dst - node_ranges[dst_part] + dst_part * self.pow_of_two
        
                g_dst_start = dst_part * self.pow_of_two
                # we make sure that all destination nodes assigned to us are in this list so that we are not missing any nodes
                unique_src = th.unique(th.cat((new_src, th.arange(g_dst_start, g_dst_start + num_dst_nodes))))

                uni_src = th.searchsorted(unique_src, new_src)
                uni_dst = th.searchsorted(unique_src, new_dst)
                # consider using dgl.create_block
                self.g = graph((uni_src, uni_dst), num_nodes=unique_src.shape[0], device=storage_device)
                self.g.ndata[NID] = unique_src.to(storage_device)
                self.g.edata[EID] = my_g.edata[EID].to(storage_device)

                self.node_ranges = th.tensor([0] * (self.group * self.group_size) + list(range(0, self.group_size + 1)) + [self.group_size] * ((self.num_groups - self.group - 1) * self.group_size), device=storage_device) * self.pow_of_two

                g_node_ranges = []
                cnts = [0] * self.group_size
                permute = []
                for i in range(self.num_groups):
                    for j in range(self.group_size):
                        rank = j * self.num_groups + i
                        permute.append(rank)
                        g_node_ranges.append(self.pow_of_two * j + cnts[j])
                        cnts[j] += g_parts[rank]
                g_node_ranges.append(self.pow_of_two * self.group_size)
                inv_permute = sorted(range(len(permute)), key=permute.__getitem__)        
                self.g_node_ranges = th.tensor(g_node_ranges, device=storage_device)[inv_permute + [-1]]
                self.permute = th.tensor(permute, device=self.device)
                self.inv_permute = th.tensor(inv_permute, device=self.device)

                self.pr = self.sorted_global_partition(self.g.ndata[NID], False).to(self.device)
                assert self.pr[self.rank + 1] - self.pr[self.rank] == num_dst_nodes
                self.g_pr = self.sorted_global_partition(self.g.ndata[NID], True).to(self.device)
    
                self.l_offset = self.g_pr[permute[self.rank]].item()

                g_offset = self.l_rank * self.pow_of_two

                self.node_ranges = self.node_ranges.to(self.device)
                self.g_node_ranges = self.g_node_ranges.to(self.device)

                self.dstdata = {NID: self.g.ndata[NID][self.g_pr[self.permute[self.rank]].item(): self.g_pr[self.permute[self.rank] + 1].item()]}
                g_NID = (self.dstdata[NID] - g_offset + node_ranges[self.l_rank]).to(cpu_device)
            else:
                self.g = my_g.to(storage_device)

                self.node_ranges = th.tensor([0] * (self.group * self.group_size) + node_ranges.tolist() + [node_ranges[-1].item()] * ((self.num_groups - self.group - 1) * self.group_size), device=self.device)

                g_node_ranges = []
                cnts = [0] * self.group_size
                permute = []
                for i in range(self.num_groups):
                    for j in range(self.group_size):
                        rank = j * self.num_groups + i
                        permute.append(rank)
                        g_node_ranges.append(node_ranges[j].item() + cnts[j])
                        cnts[j] += g_parts[rank]
                g_node_ranges.append(node_ranges[-1].item())
                inv_permute = sorted(range(len(permute)), key=permute.__getitem__)        
                self.g_node_ranges = th.tensor(g_node_ranges, device=self.device)[inv_permute + [-1]]
                self.permute = th.tensor(permute, device=self.device)
                self.inv_permute = th.tensor(inv_permute, device=self.device)

                self.pr = self.node_ranges
                assert self.pr[self.rank + 1] - self.pr[self.rank] == num_dst_nodes
                self.g_pr = self.g_node_ranges
                
                self.l_offset = self.g_pr[permute[self.rank]].item()

                self.dstdata = {}
                g_NID = slice(self.g_pr[self.permute[self.rank]], self.g_pr[self.permute[self.rank] + 1])

            del my_g

            g_EID = self.g.edata[EID].to(cpu_device, th.int64)

            self.g = self.g.formats(['csc'])
            self.pindata = {}

            for k, v in list(g_ndata.items()):
                if k != NID:
                    this_uva_data = uva_data or k in uva_ndata
                    this_storage_device = cpu_device if this_uva_data else storage_device
                    self.dstdata[k] = v[g_NID].to(this_storage_device)
                    if this_uva_data:
                        if compress:
                            self.pindata[k] = pin_memory_inplace(self.dstdata[k])
                        else:
                            self.pindata[k] = pin_memory_inplace(v)
                g_ndata.pop(k)

            for k, v in list(g_edata.items()):
                if k != EID:
                    self.g.edata[k] = v[g_EID].to(storage_device)
                g_edata.pop(k)

        self.random_seed = th.randint(0, 10000000000000, (1,), device=self.device)
        thd.all_reduce(self.random_seed, thd.ReduceOp.SUM, self.comm)
        random_seed = self.random_seed.item()
        self.permute_host = self.permute.tolist()
        
        print(self.rank, self.g.num_nodes(), self.g.num_edges(), self.pr, self.g_pr, self.l_offset, self.node_ranges, self.g_node_ranges, self.permute, self.inv_permute, random_seed)
        self.last_comm = self.comm
        self.works = []
        self.caches = {} if cache_size <= 0 else {key: GPUCache(cache_size, self.dstdata[key].shape[1], g.idtype) for key in uva_ndata}

    def sorted_global_partition(self, ids, g_comm):
        return th.searchsorted(ids, self.g_node_ranges if g_comm else self.node_ranges)
    
    def global_part(self, ids):
        return th.bitwise_right_shift(ids, self.log_pow_of_two) if self.compress else self.local_part(ids)
    
    def local_part(self, ids):
        return th.searchsorted(self.pr, ids + 1) - 1
    
    def global_to_local(self, ids, i=None):
        if not self.compress:
            return ids
        if i is None:
            i = self.global_part(ids)
        return ids - (i % self.group_size) * self.pow_of_two + self.pr[i]
    
    def local_to_global(self, ids, i=None):
        if not self.compress:
            return ids
        if i is None:
            i = self.local_part(ids)
        return ids - self.pr[i] + i * self.pow_of_two
    
    def synchronize_works(self, ins=None):
        for work in self.works:
            if work is not None:
                work.wait()
        self.works = []
        if ins is not None:
            th.flatten(ins[0])[0] += 0
    
    def all_to_all(self, outs, ins, async_op=False):
        g_comm = any(th.numel(t) > 0 and not (self.group_start <= i and i < self.group_end) for ts in [ins, outs] for i, t in enumerate(ts))
        comm = self.comm if g_comm else self.l_comm
        if not g_comm:
            outs = outs[self.group_start: self.group_end]
            ins = ins[self.group_start: self.group_end]
        if self.last_comm != comm:
            self.last_comm = comm
            self.synchronize_works(ins)
        work = thd.all_to_all(outs, ins, comm, async_op)
        if self.comm != self.l_comm:
            self.works.append(work)
        return work
    
    # local ids in, local ids out
    @nvtx.annotate("id exchange", color="purple")
    def exchange_node_ids(self, nodes, g_comm):
        nodes = cuda_index_tensor(self.g.ndata[NID], nodes) if self.compress else nodes
        partition = self.sorted_global_partition(nodes, g_comm)
        request_counts = th.diff(partition)
        received_request_counts = th.empty_like(request_counts)
        self.all_to_all(list(th.split(received_request_counts, 1)), list(th.split(request_counts[self.permute] if g_comm else request_counts, 1)))
        requested_sizes = received_request_counts.tolist()
        requested_nodes = th.empty(sum(requested_sizes), dtype=nodes.dtype, device=self.device)
        request_counts = request_counts.tolist()
        par_nodes = list(th.split(nodes, request_counts))
        if g_comm:
            par_nodes = [par_nodes[i] for i in self.permute_host]
        self.all_to_all(list(th.split(requested_nodes, requested_sizes)), par_nodes)
        requested_nodes = self.global_to_local(requested_nodes, self.rank)
        return requested_nodes, requested_sizes, request_counts
    
    @nvtx.annotate("pull", color="purple")
    def pull(self, dsttensor, request_counts, requested_nodes, requested_sizes, dstnodes, inv_ids):
        out = th.empty((sum(request_counts),) + dsttensor.shape[1:], dtype=dsttensor.dtype, device=dsttensor.device)
        self.all_to_all(list(th.split(out, request_counts)), list(th.split(dsttensor[inv_ids], requested_sizes)))
        return out
    
    @nvtx.annotate("pull", color="purple")
    def pull_ex(self, dsttensor, srcnodes=None):
        if srcnodes is None:
            srcnodes = th.arange(self.g.num_nodes(), device=self.device)
        
        requested_nodes, requested_sizes, request_counts = self.exchange_node_ids(srcnodes)
        dstnodes, inv_ids = th.unique(requested_nodes, return_inverse=True)
        
        return self.pull(dsttensor, request_counts, requested_nodes, requested_sizes, dstnodes, inv_ids)
    
    @nvtx.annotate("rpull", color="purple")
    def rpull(self, srctensor, request_counts, requested_sizes, dstnodes, inv_ids):
        out = th.empty((sum(requested_sizes),) + srctensor.shape[1:], dtype=srctensor.dtype, device=srctensor.device)
        src = th.arange(out.shape[0], device=self.device)
        dst = inv_ids
        self.all_to_all(list(th.split(out, requested_sizes)), list(th.split(srctensor, request_counts)))
        _graph = create_unitgraph_from_coo(2, out.shape[0], dstnodes.shape[0], src, dst, ['coo'], row_sorted=True)
        rout = th.zeros((dstnodes.shape[0],) + srctensor.shape[1:], dtype=srctensor.dtype, device=srctensor.device)
        lighter_gspmm(_graph, 'copy_lhs', 'sum',
                            F.zerocopy_to_dgl_ndarray(out),
                            nd.NULL['int64'],
                            F.zerocopy_to_dgl_ndarray_for_write(rout),
                            nd.NULL['int64'],
                            nd.NULL['int64'])
        return rout

    @nvtx.annotate("rpull", color="purple")
    def rpull_ex(self, srctensor, srcnodes=None):
        if srcnodes is None:
            srcnodes = th.arange(self.g.num_nodes(), device=self.device)
        
        requested_nodes, requested_sizes, request_counts = self.exchange_node_ids(srcnodes)
        dstnodes, inv_ids = th.unique(requested_nodes, return_inverse=True)

        return self.rpull(srctensor, request_counts, requested_sizes, dstnodes, inv_ids)

    def load_balance(self):
        if self.g.device != th.device('cpu') or self.group_size != self.world_size or not hasattr(self, 'lbt'):
            return
        x = self.lbt
        x = x * th.ones([self.world_size], device=self.device, dtype=th.int64)
        xs = th.empty([self.world_size], device=self.device, dtype=th.int64)
        self.all_to_all(list(th.split(xs, 1)), list(th.split(x, 1)))
        xs = xs.cpu()
        N = self.g.num_nodes()
        K = xs.sum().item()
        ds = th.cumsum(K / xs.shape[0] - xs, dim=0)
        u = N / K * 0.01
        dsu = ds * u
        self.node_ranges[1:] += dsu.to(self.node_ranges.device, self.node_ranges.dtype)
        self.g_node_ranges[1:] += dsu.to(self.g_node_ranges.device, self.g_node_ranges.dtype)
        self.pr[1:] += dsu.to(self.pr.device, self.pr.dtype)
        self.g_pr[1:] += dsu.to(self.g_pr.device, self.g_pr.dtype)
        self.l_offset = self.g_pr[self.permute[self.rank]].item()
        g_NID = slice(self.g_pr[self.permute[self.rank]], self.g_pr[self.permute[self.rank] + 1])

        self.dstdata = {k: v[g_NID] for k, v in self.g.ndata.items()}

    @nvtx.annotate("sample_blocks", color="purple")
    def sample_blocks(self, seed_nodes, samplers, exclude_eids=None, prefetch_node_feats=[], prefetch_edge_feats=[], prefetch_labels=[]):
        self.load_balance()
        blocks = []
        # start_seed_exchange = th.tensor((not (th.all(self.pr[self.rank] <= seed_nodes) and th.all(seed_nodes < self.pr[self.rank + 1]))), device=self.device, dtype=th.int64)
        # thd.all_reduce(start_seed_exchange, thd.ReduceOp.SUM, self.comm)
        # if start_seed_exchange.item() > 0:
        if not (th.all(self.pr[self.rank] <= seed_nodes) and th.all(seed_nodes < self.pr[self.rank + 1])):
            seed_nodes, seed_nodes_inv = th.sort(seed_nodes)
            requested_nodes, requested_sizes, request_counts = self.exchange_node_ids(seed_nodes, False)
            seed_nodes, inv_ids = th.unique(requested_nodes, return_inverse=True)
            
            cached_variables = (request_counts, requested_nodes, requested_sizes, seed_nodes, inv_ids, seed_nodes_inv, self)
        else:
            cached_variables = None
        output_nodes = seed_nodes
        for i, sampler in enumerate(reversed(samplers)):
            assert th.all(self.pr[self.rank] <= seed_nodes) and th.all(seed_nodes < self.pr[self.rank + 1])

            seed_nodes, _, blocks_i = sampler.sample_blocks(self.g, seed_nodes, exclude_eids=exclude_eids)
            
            requested_nodes, requested_sizes, request_counts = self.exchange_node_ids(seed_nodes, i == len(samplers) - 1)
            seed_nodes, inv_ids = th.unique(requested_nodes, return_inverse=True)
            
            blocks_i[0].cached_variables = request_counts, requested_nodes, requested_sizes, seed_nodes, inv_ids, slice(None), self

            blocks.insert(0, blocks_i[0])
        
        def feature_slicer(block):
            srcdataevents = {}
            cache_miss = 1
            for k in prefetch_node_feats:
                input_nodes = block.cached_variables[1] - self.l_offset
                if k in self.caches:
                    cache = self.caches[k]
                    tensor, missing_index, missing_keys = cache.query(input_nodes)
                    missing_values = cuda_index_tensor(self.dstdata[k], missing_keys)
                    cache.replace(missing_keys, missing_values.to(th.float))
                    cache_miss = missing_keys.shape[0] / input_nodes.shape[0]
                    tensor[missing_index] = missing_values.to(tensor.dtype)
                else:
                    tensor = cuda_index_tensor(self.dstdata[k], input_nodes).to(self.device, th.float) #
                out = th.empty((sum(request_counts),) + tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)
                par_out = list(th.split(out, request_counts))
                par_out = [par_out[i] for i in self.permute_host]
                work = self.all_to_all(par_out, list(th.split(tensor, requested_sizes)), True)
                out.cache_miss = cache_miss
                block.srcdata[k] = out # .to(th.float)
                srcdataevents[k] = work
            
            def wait(k=None):
                if k is None:
                    for k, work in srcdataevents.items():
                        work.wait()
                else:
                    srcdataevents[k].wait()
            return wait
        
        def label_slicer(block):
            for k in prefetch_labels:
                block.dstdata[k] = cuda_index_tensor(self.dstdata[k], output_nodes - self.l_offset).to(self.device)

        def edge_slicer(block):
            for k in prefetch_edge_feats:
                block.edata[k] = cuda_index_tensor(self.g.edata[k], block.edata[EID]).to(self.device)

        blocks[0].slice_features = feature_slicer
        blocks[-1].slice_labels = label_slicer
        for block in blocks:
            block.slice_edges = edge_slicer

        blocks[-1].cached_variables2 = cached_variables

        # self.lbt = seed_nodes.shape[0]
        # self.lbt = blocks[0].num_src_nodes()

        return seed_nodes, output_nodes, blocks