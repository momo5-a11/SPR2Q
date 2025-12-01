import math
import numpy as np
import paddle


def cubic(x):
    """cubic function used for calculate_weights_indices (Paddle version)."""
    absx = paddle.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    # 修正: 使用 paddle.logical_and
    term1 = (1.5 * absx3 - 2.5 * absx2 + 1) * paddle.cast(absx <= 1, x.dtype)
    term2 = (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * paddle.cast(
        paddle.logical_and(absx > 1, absx <= 2), x.dtype)
    return term1 + term2


def calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    """Calculate weights and indices, used for imresize function (Paddle version)."""

    if (scale < 1) and antialiasing:
        kernel_width = kernel_width / scale

    x = paddle.linspace(1, out_length, out_length)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = paddle.floor(u - kernel_width / 2)
    p = math.ceil(kernel_width) + 2

    # 修正: 使用 paddle.unsqueeze
    indices = paddle.unsqueeze(left, axis=1) + paddle.unsqueeze(paddle.linspace(0, p - 1, p), axis=0)
    distance_to_center = paddle.unsqueeze(u, axis=1) - indices

    if (scale < 1) and antialiasing:
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)

    weights_sum = paddle.sum(weights, axis=1, keepdim=True)
    weights = weights / weights_sum

    # 这里使用 != 0 是可以的，因为 weights_zero_tmp 是0和1的和，即计数
    weights_zero_tmp = paddle.sum(paddle.cast(weights == 0, "int32"), axis=0)
    if not math.isclose(weights_zero_tmp[0].item(), 0):
        indices = indices[:, 1:p - 1]
        weights = weights[:, 1:p - 1]
    if not math.isclose(weights_zero_tmp[-1].item(), 0):
        indices = indices[:, 0:p - 2]
        weights = weights[:, 0:p - 2]

    sym_len_s = int(-paddle.min(indices) + 1)
    sym_len_e = int(paddle.max(indices) - in_length)
    indices = indices + sym_len_s - 1
    return weights, indices, sym_len_s, sym_len_e


@paddle.no_grad()
def imresize(img, scale, antialiasing=True):
    """imresize function same as MATLAB (Paddle version)."""
    squeeze_flag = False
    if isinstance(img, np.ndarray):
        numpy_type = True
        if img.ndim == 2:
            img = img[:, :, None]
            squeeze_flag = True
        img = paddle.to_tensor(img.transpose(2, 0, 1), dtype='float32')
    else:
        numpy_type = False
        if img.ndim == 2:
            # 修正: 使用 paddle.unsqueeze
            img = paddle.unsqueeze(img, axis=0)
            squeeze_flag = True

    in_c, in_h, in_w = img.shape
    out_h, out_w = math.ceil(in_h * scale), math.ceil(in_w * scale)
    kernel_width = 4
    kernel = 'cubic'

    weights_h, indices_h, sym_len_hs, sym_len_he = calculate_weights_indices(
        in_h, out_h, scale, kernel, kernel_width, antialiasing)
    weights_w, indices_w, sym_len_ws, sym_len_we = calculate_weights_indices(
        in_w, out_w, scale, kernel, kernel_width, antialiasing)

    # process H dimension
    img_aug = paddle.zeros([in_c, in_h + sym_len_hs + sym_len_he, in_w], dtype='float32')
    img_aug[:, sym_len_hs:sym_len_hs + in_h, :] = img

    sym_patch = img[:, :sym_len_hs, :]
    img_aug[:, :sym_len_hs, :] = sym_patch[:, ::-1, :]

    sym_patch = img[:, -sym_len_he:, :]
    img_aug[:, sym_len_hs + in_h:, :] = sym_patch[:, ::-1, :]

    out_1 = paddle.zeros([in_c, out_h, in_w], dtype='float32')
    kernel_width_h = weights_h.shape[1]
    for i in range(out_h):
        idx = int(indices_h[i][0])
        slice_h = img_aug[:, idx:idx + kernel_width_h, :]
        out_1[:, i, :] = paddle.sum(slice_h * paddle.reshape(weights_h[i], [1, -1, 1]), axis=1)

    # process W dimension
    out_1_aug = paddle.zeros([in_c, out_h, in_w + sym_len_ws + sym_len_we], dtype='float32')
    out_1_aug[:, :, sym_len_ws:sym_len_ws + in_w] = out_1

    sym_patch = out_1[:, :, :sym_len_ws]
    out_1_aug[:, :, :sym_len_ws] = sym_patch[:, :, ::-1]

    sym_patch = out_1[:, :, -sym_len_we:]
    out_1_aug[:, :, sym_len_ws + in_w:] = sym_patch[:, :, ::-1]

    out_2 = paddle.zeros([in_c, out_h, out_w], dtype='float32')
    kernel_width_w = weights_w.shape[1]
    for i in range(out_w):
        idx = int(indices_w[i][0])
        slice_w = out_1_aug[:, :, idx:idx + kernel_width_w]
        out_2[:, :, i] = paddle.sum(slice_w * paddle.reshape(weights_w[i], [1, 1, -1]), axis=2)

    if squeeze_flag:
        # 修正: 使用 paddle.squeeze
        out_2 = paddle.squeeze(out_2, axis=0)
        
    if numpy_type:
        out_2 = out_2.numpy()
        if not squeeze_flag:
            out_2 = out_2.transpose(1, 2, 0)

    return out_2


def rgb2ycbcr(img, y_only=False):
    """Convert a RGB image to YCbCr image.

    This function produces the same results as Matlab's `rgb2ycbcr` function.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `RGB <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [65.481, 128.553, 24.966]) + 16.0
    else:
        out_img = np.matmul(
            img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def bgr2ycbcr(img, y_only=False):
    """Convert a BGR image to YCbCr image.

    The bgr version of rgb2ycbcr.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `BGR <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def ycbcr2rgb(img):
    """Convert a YCbCr image to RGB image.

    This function produces the same results as Matlab's ycbcr2rgb function.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `YCrCb <-> RGB`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].

    Returns:
        ndarray: The converted RGB image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img) * 255
    out_img = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                              [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]  # noqa: E126
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def ycbcr2bgr(img):
    """Convert a YCbCr image to BGR image.

    The bgr version of ycbcr2rgb.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `YCrCb <-> BGR`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].

    Returns:
        ndarray: The converted BGR image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img) * 255
    out_img = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0.00791071, -0.00153632, 0],
                              [0, -0.00318811, 0.00625893]]) * 255.0 + [-276.836, 135.576, -222.921]  # noqa: E126
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def _convert_input_type_range(img):
    """Convert the type and range of the input image.

    It converts the input image to np.float32 type and range of [0, 1].
    It is mainly used for pre-processing the input image in colorspace
    conversion functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].

    Returns:
        (ndarray): The converted image with type of np.float32 and range of
            [0, 1].
    """
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError(f'The img type should be np.float32 or np.uint8, but got {img_type}')
    return img


def _convert_output_type_range(img, dst_type):
    """Convert the type and range of the image according to dst_type.

    It converts the image to desired type and range. If `dst_type` is np.uint8,
    images will be converted to np.uint8 type with range [0, 255]. If
    `dst_type` is np.float32, it converts the image to np.float32 type with
    range [0, 1].
    It is mainly used for post-processing images in colorspace conversion
    functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The image to be converted with np.float32 type and
            range [0, 255].
        dst_type (np.uint8 | np.float32): If dst_type is np.uint8, it
            converts the image to np.uint8 type with range [0, 255]. If
            dst_type is np.float32, it converts the image to np.float32 type
            with range [0, 1].

    Returns:
        (ndarray): The converted image with desired type and range.
    """
    if dst_type not in (np.uint8, np.float32):
        raise TypeError(f'The dst_type should be np.float32 or np.uint8, but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)
