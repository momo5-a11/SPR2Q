import cv2
import numpy as np
import paddle
import paddle.nn.functional as F
from os import path as osp
from basicsr.utils import img2tensor, scandir


def generate_frame_indices(crt_idx, max_frame_num, num_frames, padding='reflection'):
    """Generate an index list for reading `num_frames` frames from a sequence."""
    assert num_frames % 2 == 1, 'num_frames should be an odd number.'
    assert padding in ('replicate', 'reflection', 'reflection_circle', 'circle'), f'Wrong padding mode: {padding}.'

    max_frame_num = max_frame_num - 1
    num_pad = num_frames // 2

    indices = []
    for i in range(crt_idx - num_pad, crt_idx + num_pad + 1):
        if i < 0:
            if padding == 'replicate':
                pad_idx = 0
            elif padding == 'reflection':
                pad_idx = -i
            elif padding == 'reflection_circle':
                pad_idx = crt_idx + num_pad - i
            else:
                pad_idx = num_frames + i
        elif i > max_frame_num:
            if padding == 'replicate':
                pad_idx = max_frame_num
            elif padding == 'reflection':
                pad_idx = max_frame_num * 2 - i
            elif padding == 'reflection_circle':
                pad_idx = (crt_idx - num_pad) - (i - max_frame_num)
            else:
                pad_idx = i - num_frames
        else:
            pad_idx = i
        indices.append(pad_idx)
    return indices


def paired_paths_from_lmdb(folders, keys):
    """Generate paired paths from lmdb files."""
    assert len(folders) == 2
    assert len(keys) == 2
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    if not (input_folder.endswith('.lmdb') and gt_folder.endswith('.lmdb')):
        raise ValueError(f'{input_key} and {gt_key} should both be lmdb.')

    with open(osp.join(input_folder, 'meta_info.txt')) as fin:
        input_lmdb_keys = [line.split('.')[0] for line in fin]
    with open(osp.join(gt_folder, 'meta_info.txt')) as fin:
        gt_lmdb_keys = [line.split('.')[0] for line in fin]

    if set(input_lmdb_keys) != set(gt_lmdb_keys):
        raise ValueError(f'Keys in {input_key} and {gt_key} folders are different.')
    else:
        paths = []
        for lmdb_key in sorted(input_lmdb_keys):
            paths.append({f'{input_key}_path': lmdb_key, f'{gt_key}_path': lmdb_key})
        return paths


def paired_paths_from_meta_info_file(folders, keys, meta_info_file, filename_tmpl):
    """Generate paired paths from meta info file."""
    assert len(folders) == 2
    assert len(keys) == 2
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    with open(meta_info_file, 'r') as fin:
        gt_names = [line.strip().split(' ')[0] for line in fin]

    paths = []
    for gt_name in gt_names:
        basename, ext = osp.splitext(osp.basename(gt_name))
        input_name = f'{filename_tmpl.format(basename)}{ext}'
        input_path = osp.join(input_folder, input_name)
        gt_path = osp.join(gt_folder, gt_name)
        paths.append({f'{input_key}_path': input_path, f'{gt_key}_path': gt_path})
    return paths


def paired_paths_from_folder(folders, keys, filename_tmpl):
    """Generate paired paths from two folders."""
    assert len(folders) == 2
    assert len(keys) == 2
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    input_paths = list(scandir(input_folder))
    gt_paths = list(scandir(gt_folder))
    assert len(input_paths) == len(gt_paths), f'{input_key} and {gt_key} folder size mismatch.'

    paths = []
    for gt_path in gt_paths:
        basename, ext = osp.splitext(osp.basename(gt_path))
        input_name = f'{filename_tmpl.format(basename)}{ext}'
        input_path = osp.join(input_folder, input_name)
        assert input_name in input_paths, f'{input_name} not found in {input_key} folder.'
        gt_path = osp.join(gt_folder, gt_path)
        paths.append({f'{input_key}_path': input_path, f'{gt_key}_path': gt_path})
    return paths


def paths_from_folder(folder):
    """Generate paths from folder."""
    paths = list(scandir(folder))
    paths = [osp.join(folder, path) for path in paths]
    return paths


def paths_from_lmdb(folder):
    """Generate paths from lmdb."""
    if not folder.endswith('.lmdb'):
        raise ValueError(f'Folder {folder} should be lmdb format.')
    with open(osp.join(folder, 'meta_info.txt')) as fin:
        paths = [line.split('.')[0] for line in fin]
    return paths

from scipy.ndimage import filters
def generate_gaussian_kernel(kernel_size=13, sigma=1.6):
    """Generate Gaussian kernel used in `duf_downsample`."""
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, kernel_size // 2] = 1
    kernel = filters.gaussian_filter(kernel, sigma)
    return kernel


def duf_downsample(x, kernel_size=13, scale=4):
    """
    Downsampling with Gaussian kernel used in DUF.

    Args:
        x (Tensor): shape (b, t, c, h, w) or (t, c, h, w)
        kernel_size (int): kernel size, default 13
        scale (int): downsampling factor (2, 3, or 4)
    Returns:
        Tensor: downsampled frames (same shape convention)
    """
    assert scale in (2, 3, 4), f"Only support scale (2, 3, 4), but got {scale}."

    squeeze_flag = False
    if len(x.shape) == 4:
        # Input shape: (t, c, h, w)
        squeeze_flag = True
        # 修正: 使用 paddle.unsqueeze 并指定轴
        x = paddle.unsqueeze(x, axis=0)  # -> (1, t, c, h, w)

    b, t, c, h, w = x.shape
    # 合并 batch 和 time 维度: (b*t, c, h, w)
    # 修正: 使用 paddle.reshape
    x = paddle.reshape(x, [b * t, c, h, w])

    # 反射填充
    pad_w = kernel_size // 2 + scale * 2
    pad_h = kernel_size // 2 + scale * 2
    x = F.pad(x, [pad_w, pad_w, pad_h, pad_h], mode='reflect')

    # 生成高斯滤波核
    gaussian_filter = generate_gaussian_kernel(kernel_size, 0.4 * scale)
    gaussian_filter = paddle.to_tensor(gaussian_filter, dtype=x.dtype)

    # 扩展为 (out_channels, in_channels, kH, kW)
    # 修正: 使用 paddle.reshape
    gaussian_filter = paddle.reshape(gaussian_filter, [1, 1, kernel_size, kernel_size])
    gaussian_filter = paddle.concat([gaussian_filter] * c, axis=0)

    # 分通道卷积（depthwise）
    x = F.conv2d(
        x,
        weight=gaussian_filter,
        stride=scale,
        groups=c
    )

    x = x[:, :, 2:-2, 2:-2]

    x = paddle.reshape(x, [b, t, c, x.shape[2], x.shape[3]])
    if squeeze_flag:
        x = paddle.squeeze(x, axis=0)

    return x
