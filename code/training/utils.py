'''
Author: Niki
Date: 2021-01-08 12:44:48
Description: 
'''
import torch
from torch import distributed
from torch import autograd
from torch.nn.parallel import DistributedDataParallel as DDP


def print_if_rank0(*args):
    if distributed.get_rank() == 0:
        print(*args)


class awesome_allgather_function(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        world_size = distributed.get_world_size()
        # create a destination list for the allgather.  I'm assuming you're gathering from 3 workers.
        allgather_list = [torch.empty_like(input) for _ in range(world_size)]
        #if distributed.get_rank() == 0:
        #    import IPython;IPython.embed()
        distributed.all_gather(allgather_list, input)
        return torch.cat(allgather_list, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        #print_if_rank0("backward grad_output len", len(grad_output))
        #print_if_rank0("backward grad_output shape", grad_output.shape)
        grads_per_rank = grad_output.shape[0] // distributed.get_world_size()
        rank = distributed.get_rank()
        # We'll receive gradients for the entire catted forward output, so to mimic DataParallel,
        # return only the slice that corresponds to this process's input:
        sl = slice(rank * grads_per_rank, (rank + 1) * grads_per_rank)
        #print("worker", rank, "backward slice", sl)
        return grad_output[sl]
