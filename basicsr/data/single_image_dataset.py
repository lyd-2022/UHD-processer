from os import path as osp
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import sys
sys.path.append('/code/UHD-allinone')
from basicsr.data.data_util import paths_from_lmdb
from basicsr.utils import FileClient, imfrombytes, img2tensor, scandir
from basicsr.utils.matlab_functions import rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.data.transforms import augment, paired_random_crop, paired_random_crop_DP, random_augmentation,single_random_crop
import numpy as np
from PIL import Image
import os
import random
from torchvision.transforms import ToPILImage, Compose, ToTensor

def _add_gaussian_noise(clean_patch, sigma):

    noise = np.random.randn(*clean_patch.shape)
    noisy_patch = np.clip(clean_patch + noise * sigma, 0, 255).astype(np.uint8)
    return noisy_patch, clean_patch

@DATASET_REGISTRY.register()
class SingleImageDataset(data.Dataset):
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
        super(SingleImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.lq_folder = opt['dataroot_lq']
        self.deg= opt['deg'] if 'deg' in opt else 0

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder]
            self.io_backend_opt['client_keys'] = ['lq']
            self.paths = paths_from_lmdb(self.lq_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [osp.join(self.lq_folder, line.rstrip().split(' ')[0]) for line in fin]
        else:
            self.paths = sorted(list(scandir(self.lq_folder, full_path=True)))
        
        
        self.crop_transform = Compose([
            ToPILImage(),
        ])

        self.toTensor = ToTensor()
    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        if self.deg > 0:
            # num = random.randint(0, 9)
            lq_path = self.paths[index]
            img_lq = np.array(Image.open(lq_path).convert('RGB'))
            img_gt = img_lq
            img_lq, _ = _add_gaussian_noise(img_lq, self.deg)
            img_gt = self.toTensor(self.crop_transform(img_gt))
            img_lq = self.toTensor(self.crop_transform(img_lq))
            
            


            if self.mean is not None or self.std is not None:
                normalize(img_lq, self.mean, self.std, inplace=True)
                normalize(img_gt, self.mean, self.std, inplace=True)
            return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path}
        else:
            # load lq image
            lq_path = self.paths[index]
            img_bytes = self.file_client.get(lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            


            if self.opt['phase'] == 'train':
                if self.opt['gt_size'] > 0:
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


#测试dataset
if __name__ == '__main__':
    import yaml
    import os
    import sys

    from basicsr.data import create_dataloader, create_dataset
    with open('/code/UHD-allinone/options/debug.yml') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)
    dataset = create_dataset(opt['datasets']['val'])
    dataset_opt = opt['datasets']['val']
    dataset_opt['phase'] = 'val'

    # dataset = create_dataset(opt['datasets']['train'])
    # dataset_opt = opt['datasets']['train']
    # dataset_opt['phase'] = 'train'
    dataloader = create_dataloader(dataset, dataset_opt)

    for i, data in enumerate(dataloader):
        print(i)
        print(data['lq'].shape)
        print(data['lq_path'])
        # break
    print('done')