import torch as th
import torch.distributed as thd

def barrier(before, after, comm=None):
    value = th.zeros(1, dtype=before.dtype, device=before.device)
    value[0] = th.flatten(before)[0] - th.flatten(before)[0]
    thd.all_reduce(value, group=comm)
    th.flatten(after)[0] += value[0]
    