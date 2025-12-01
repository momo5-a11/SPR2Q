import functools
import os
import subprocess
import paddle
import paddle.distributed as dist

def init_dist(launcher, backend='nccl', **kwargs):
    if launcher == 'paddle':
        _init_dist_paddle(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')


def _init_dist_paddle(backend='nccl', **kwargs):
    dist.init_parallel_env()


def _init_dist_slurm(backend='nccl', port=None):
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = paddle.device.cuda.device_count()
    local_rank = proc_id % num_gpus
    paddle.device.set_device(f'gpu:{local_rank}')

    addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
    os.environ['MASTER_ADDR'] = addr
    os.environ['MASTER_PORT'] = str(port or os.environ.get('MASTER_PORT', '29500'))
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    os.environ['LOCAL_RANK'] = str(local_rank)   
    dist.init_parallel_env()


def get_dist_info():
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def master_only(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper
