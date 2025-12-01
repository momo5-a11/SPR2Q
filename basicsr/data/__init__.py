import importlib
import numpy as np
import random
import paddle
from copy import deepcopy
from functools import partial
from os import path as osp

from basicsr.data.prefetch_dataloader import PrefetchDataLoader
from basicsr.utils import get_root_logger, scandir
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.registry import DATASET_REGISTRY

__all__ = ['build_dataset', 'build_dataloader']

# automatically scan and import dataset modules for registry
# scan all the files under the data folder with '_dataset' in file names
data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(data_folder) if v.endswith('_dataset.py')]
# import all the dataset modules
_dataset_modules = [importlib.import_module(f'basicsr.data.{file_name}') for file_name in dataset_filenames]


def build_dataset(dataset_opt):
    """Build dataset from options.

    Args:
        dataset_opt (dict): Configuration for dataset. It must contain:
            name (str): Dataset name.
            type (str): Dataset type.
    """
    dataset_opt = deepcopy(dataset_opt)
    dataset = DATASET_REGISTRY.get(dataset_opt['type'])(dataset_opt)
    logger = get_root_logger()
    logger.info(f'Dataset [{dataset.__class__.__name__}] - {dataset_opt["name"]} is built.')
    return dataset


import paddle
from functools import partial
from basicsr.data.prefetch_dataloader import PrefetchDataLoader
from basicsr.utils import get_root_logger, get_dist_info


def build_dataloader(dataset, dataset_opt, num_gpu=1, dist=False, sampler=None, seed=None):
    """Build dataloader (PaddlePaddle version)."""
    phase = dataset_opt['phase']
    rank, _ = get_dist_info()

    if phase == 'train':
        if dist:  # distributed training
            batch_size = dataset_opt['batch_size_per_gpu']
            num_workers = dataset_opt['num_worker_per_gpu']
        else:
            multiplier = 1 if num_gpu == 0 else num_gpu
            batch_size = dataset_opt['batch_size_per_gpu'] * multiplier
            num_workers = dataset_opt['num_worker_per_gpu'] * multiplier

        if sampler is not None:
            # 1. 导入 BatchSampler
            from paddle.io import BatchSampler
            
            batch_sampler = BatchSampler(
                sampler=sampler,
                batch_size=batch_size,
                shuffle=False,  # shuffle 功能由 sampler 控制，这里设为 False
                drop_last=True
            )

            dataloader_args = dict(
                dataset=dataset,
                batch_sampler=batch_sampler, # <-- 传递真正的批次采样器
                num_workers=num_workers,
            )

        if seed is not None:
            dataloader_args["worker_init_fn"] = partial(
                worker_init_fn, num_workers=num_workers, rank=rank, seed=seed
            )

    elif phase in ["val", "test"]:
        dataloader_args = dict(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
    else:
        raise ValueError(f"Wrong dataset phase: {phase}. Supported ones are 'train', 'val' and 'test'.")

    # Paddle 没有 pin_memory/persistent_workers 参数，略过
    prefetch_mode = dataset_opt.get("prefetch_mode")
    if prefetch_mode == "cpu":
        num_prefetch_queue = dataset_opt.get("num_prefetch_queue", 1)
        logger = get_root_logger()
        logger.info(f"Use {prefetch_mode} prefetch dataloader: num_prefetch_queue = {num_prefetch_queue}")
        return PrefetchDataLoader(num_prefetch_queue=num_prefetch_queue, **dataloader_args)
    else:
        return paddle.io.DataLoader(**dataloader_args)


def worker_init_fn(worker_id, num_workers, rank, seed):
    # Set the worker seed to num_workers * rank + worker_id + seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
