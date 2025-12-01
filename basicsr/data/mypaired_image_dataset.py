import paddle
from paddle.io import Dataset
from paddle.vision.transforms.functional import normalize
import numpy as np

from basicsr.data.data_util import (
    paired_paths_from_folder,
    paired_paths_from_lmdb,
    paired_paths_from_meta_info_file,
)
from basicsr.data.transforms import augment, mypaired_random_crop, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.matlab_functions import bgr2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class myPairedImageDataset(Dataset):
    """Paired image dataset for image restoration (Paddle version).

    Supports:
        - lmdb backend
        - meta_info_file backend
        - folder scan backend

    Args:
        opt (dict): Dataset config (same format as PyTorch version)
    """

    def __init__(self, opt):
        super(myPairedImageDataset, self).__init__()
        self.opt = opt
        self.train = self.opt.get('train', False)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)

        self.gt_folder = opt.get('dataroot_gt', None)
        self.lq_folder = opt.get('dataroot_lq', None)
        self.filename_tmpl = opt.get('filename_tmpl', '{}')

        # build paths
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt']
            )
        elif 'meta_info_file' in opt and opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder],
                ['lq', 'gt'],
                opt['meta_info_file'],
                self.filename_tmpl,
            )
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl
            )

    def __getitem__(self, index):
        """Return one training or validation/test sample."""
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # -----------------------------
        # Validation / Testing phase
        # -----------------------------
        if not self.train:
            gt_path = self.paths[index]['gt_path']
            lq_path = self.paths[index]['lq_path']

            img_gt = imfrombytes(self.file_client.get(gt_path, 'gt'), float32=True)
            img_lq = imfrombytes(self.file_client.get(lq_path, 'lq'), float32=True)

            # random crop + augment
            if self.opt['phase'] == 'train':
                gt_size = self.opt['gt_size']
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
                img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

            # color space
            if self.opt.get('color', None) == 'y':
                img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
                img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

            # crop unmatched gt
            if self.opt['phase'] != 'train':
                img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

            # numpy → paddle tensor
            img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

            # normalize
            if self.mean is not None or self.std is not None:
                img_lq = normalize(img_lq, mean=self.mean, std=self.std)
                img_gt = normalize(img_gt, mean=self.mean, std=self.std)

            # convert to paddle tensor explicitly
            img_gt = paddle.to_tensor(np.array(img_gt))
            img_lq = paddle.to_tensor(np.array(img_lq))

            return {
                'lq': img_lq,
                'gt': img_gt,
                'lq_path': lq_path,
                'gt_path': gt_path,
            }

        # -----------------------------
        # Training phase (LQ only)
        # -----------------------------
        else:
            lq_path = self.paths[index]['lq_path']
            img_lq = imfrombytes(self.file_client.get(lq_path, 'lq'), float32=True)

            if self.opt['phase'] == 'train':
                gt_size = self.opt['gt_size']
                img_lq = mypaired_random_crop(img_lq, gt_size, scale)
                img_lq = augment([img_lq], self.opt['use_hflip'], self.opt['use_rot'])[0]

            # color space
            if self.opt.get('color', None) == 'y':
                img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

            # numpy → paddle tensor
            img_lq = img2tensor([img_lq], bgr2rgb=True, float32=True)[0]

            if self.mean is not None or self.std is not None:
                img_lq = normalize(img_lq, mean=self.mean, std=self.std)

            img_lq = paddle.to_tensor(np.array(img_lq))

            return {'lq': img_lq, 'lq_path': lq_path}

    def __len__(self):
        return len(self.paths)
