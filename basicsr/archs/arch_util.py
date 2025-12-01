import collections.abc
import math
import warnings
from itertools import repeat
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# from basicsr.utils import get_root_logger


@paddle.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0):
    """Initialize network weights (Paddle version)."""
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.sublayers():
            if isinstance(m, nn.Conv2D):
                nn.initializer.KaimingNormal()(m.weight)
                m.weight.set_value(m.weight * scale)
                if m.bias is not None:
                    m.bias.set_value(paddle.full_like(m.bias, bias_fill))
            elif isinstance(m, nn.Linear):
                nn.initializer.KaimingNormal()(m.weight)
                m.weight.set_value(m.weight * scale)
                if m.bias is not None:
                    m.bias.set_value(paddle.full_like(m.bias, bias_fill))
            elif isinstance(m, nn.BatchNorm2D):
                m.weight.set_value(paddle.full_like(m.weight, 1.0))
                if m.bias is not None:
                    m.bias.set_value(paddle.full_like(m.bias, bias_fill))


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks."""
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Layer):
    """Residual block without BN."""

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2D(num_feat, num_feat, 3, 1, 1)
        self.conv2 = nn.Conv2D(num_feat, num_feat, 3, 1, 1)
        self.relu = nn.ReLU()

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class Upsample(nn.Sequential):
    """Upsample module."""

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2D(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2D(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super().__init__(*m)


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow."""
    assert x.shape[-2:] == flow.shape[1:3]
    n, c, h, w = x.shape
    # mesh grid
    grid_y, grid_x = paddle.meshgrid(paddle.arange(0, h, dtype=x.dtype), paddle.arange(0, w, dtype=x.dtype))
    grid = paddle.stack((grid_x, grid_y), axis=2)
    grid = grid.unsqueeze(0).tile([n, 1, 1, 1])
    vgrid = grid + flow

    # normalize to [-1, 1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = paddle.stack((vgrid_x, vgrid_y), axis=3)

    output = F.grid_sample(
        x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners
    )
    return output


def resize_flow(flow, size_type, sizes, interp_mode='bilinear', align_corners=False):
    """Resize a flow according to ratio or shape."""
    _, _, flow_h, flow_w = flow.shape
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(f'Size type should be ratio or shape, but got {size_type}.')

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(input_flow, size=(output_h, output_w), mode=interp_mode, align_corners=align_corners)
    return resized_flow


def pixel_unshuffle(x, scale):
    """Pixel unshuffle."""
    b, c, hh, hw = x.shape
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.reshape([b, c, h, scale, w, scale])
    return x_view.transpose([0, 1, 3, 5, 2, 4]).reshape([b, out_channel, h, w])


# ===== truncated normal init =====

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """Truncated normal init (Paddle version)."""
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in trunc_normal_. The distribution may be incorrect.',
            stacklevel=2,
        )

    with paddle.no_grad():
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * low - 1, 2 * up - 1)
        tensor.erfinv_()
        tensor.scale_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clip_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Fill the input tensor with truncated normal distribution."""
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# ===== tuple utilities =====
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
