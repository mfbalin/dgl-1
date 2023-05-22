from torch.utils.tensorboard import SummaryWriter
import time
from torch.cuda import nvtx
import torch

BUFFER_SIZE = 1 << 24

class BufferedWriter:
    def __init__(self, logdir):
        self.logdir = logdir
        self.tag_id = torch.empty([BUFFER_SIZE], dtype=torch.int64)
        self.scalar = torch.empty([BUFFER_SIZE])
        self.global_step = torch.empty([BUFFER_SIZE], dtype=torch.int64)
        self.time = torch.empty([BUFFER_SIZE], dtype=torch.int64)
        self.tags = {}
        self.rev_tags = {}
        self.num_entries = 0

    def add_scalar(self, tag, scalar_value, global_step=-1):
        if self.num_entries == len(self.tag_id):
            new_size = self.num_entries * 2
            self.tag_id.resize_([new_size])
            self.scalar.resize_([new_size])
            self.global_step.resize_([new_size])
            self.time.resize_([new_size])

        if tag not in self.tags:
            tag_id = len(self.tags)
            self.tags[tag] = tag_id
            self.rev_tags[tag_id] = tag
        else:
            tag_id = self.tags[tag]

        self.tag_id[self.num_entries] = tag_id
        self.scalar[self.num_entries] = scalar_value
        self.global_step[self.num_entries] = global_step
        self.time[self.num_entries] = time.time()

        self.num_entries += 1

    def flush(self):
        nvtx.range_push("BufferedWriter.flush()")
        writer = SummaryWriter(self.logdir)
        for i in range(self.num_entries):
            tag = self.rev_tags[self.tag_id[i].item()]
            scalar_value = self.scalar[i].item()
            global_step = self.global_step[i].item()
            if global_step < 0:
                global_step = None
            walltime = self.time[i].item()
            writer.add_scalar(tag=tag,
                              scalar_value=scalar_value,
                              global_step=global_step,
                              walltime=walltime)
        writer.close()
        self.num_entries = 0
        nvtx.range_pop()

    def close(self):
        if self.num_entries > 0:
            self.flush()