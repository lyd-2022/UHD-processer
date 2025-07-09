from os import path as osp
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paths_from_lmdb
from basicsr.utils import FileClient, imfrombytes, img2tensor, scandir
from basicsr.utils.matlab_functions import rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.data.transforms import augment, paired_random_crop, paired_random_crop_DP, random_augmentation,single_random_crop
import random

@DATASET_REGISTRY.register()
class SinglecropImageDataset(data.Dataset):
    """Read only lq images in the test phase.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc).

    There are two modes:
    1. 'meta_info_file': Use meta information file to generate paths.
    2. 'folder': Scan folders to generate paths.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
    """

    def __init__(self, opt):
        super(SinglecropImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.rand_num = opt['rand_num']
        self.paths = [line.strip() for line in open(opt['root_file'], 'r')]


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # load lq image

        lq_path = self.paths[index]
        if self.rand_num > 0:
            num =random.randint(0,int(self.rand_num))
            file_name = str(num)+'_'+ lq_path.split('/')[-1]
            lq_path = lq_path.replace(lq_path.split('/')[-1],file_name)
        img_bytes = self.file_client.get(lq_path, 'lq')

        img_lq = imfrombytes(img_bytes, float32=True)
        


        if self.opt['phase'] == 'train':
            img_lq = single_random_crop(img_lq, self.opt['gt_size'],lq_path )
            img_lq = random_augmentation(img_lq)[0]
        elif self.opt['phase'] == 'val':
            if self.opt['gt_size'] > 0:
                img_lq = single_random_crop(img_lq, self.opt['gt_size'],lq_path )
             

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
        return {'lq': img_lq, 'lq_path': lq_path}

    def __len__(self):
        return len(self.paths)
