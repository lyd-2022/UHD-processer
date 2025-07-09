from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_DP_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop, paired_random_crop_DP, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor, Grayscale
from basicsr.utils.img_util import random_augmentation, crop_img
from PIL import Image
from basicsr.archs.wtconv.util import wavelet

import random
import numpy as np
import torch
import cv2
from basicsr.utils.registry import DATASET_REGISTRY
import os
import torch
import torch.nn as nn


wt_filter, iwt_filter = wavelet.create_wavelet_filter('db1', 3, 3, torch.float)
wt_filter = nn.Parameter(wt_filter, requires_grad=False)
iwt_filter = nn.Parameter(iwt_filter, requires_grad=False)

wt_function = wavelet.wavelet_transform_init(wt_filter)
iwt_function = wavelet.inverse_wavelet_transform_init(iwt_filter)

class Degradation(object):
    def __init__(self, opt):
        super(Degradation, self).__init__()
        self.opt = opt
        self.toTensor = ToTensor()
        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(opt['gt_size']),
        ])

    def _add_gaussian_noise(self, clean_patch, sigma):

        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def _degrade_by_type(self, clean_patch, degrade_type):
        if degrade_type == 0:
            # denoise sigma=15
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=15)
        elif degrade_type == 1:
            # denoise sigma=25
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=25)
        elif degrade_type == 2:
            # denoise sigma=50
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=50)

        return degraded_patch, clean_patch

    def degrade(self, clean_patch_1, clean_patch_2, degrade_type=None):
        if degrade_type == None:
            degrade_type = random.randint(0, 3)
        else:
            degrade_type = degrade_type

        degrad_patch_1, _ = self._degrade_by_type(clean_patch_1, degrade_type)
        degrad_patch_2, _ = self._degrade_by_type(clean_patch_2, degrade_type)
        return degrad_patch_1, degrad_patch_2

    def single_degrade(self,clean_patch,degrade_type = None):
        if degrade_type == None:
            degrade_type = random.randint(0, 3)
        else:
            degrade_type = degrade_type

        degrad_patch_1, _ = self._degrade_by_type(clean_patch, degrade_type)
        return degrad_patch_1

@DATASET_REGISTRY.register()
class mutiPairedcropImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    opt:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(mutiPairedcropImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.rs_ids = []
        self.hazy_ids = []
        self.D = Degradation(opt)
        self.de_temp = 0
        de_type_scale = self.opt['de_type_scale']
        self.de_type = self.opt['de_type']
        print(self.de_type)

        self.de_dict = {'denoise_15': 0, 'denoise_25': 1, 'denoise_50': 2, 'derain': 3, 'dehaze': 4, 'deblur' : 5, 'enhance' : 6,'snow':7}
        self.scale_dict = dict(zip(self.de_type, de_type_scale))

        self._init_ids()
        self._merge_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(opt['gt_size']),
        ])

        self.toTensor = ToTensor()

    
    def _init_ids(self):
        if 'denoise_15' in self.de_type or 'denoise_25' in self.de_type or 'denoise_50' in self.de_type:
            self._init_clean_ids()
        if 'derain' in self.de_type:
            self._init_rs_ids()
        if 'dehaze' in self.de_type:
            self._init_hazy_ids()
        if 'deblur' in self.de_type:
            self._init_deblur_ids()
        if 'enhance' in self.de_type:
            self._init_enhance_ids()
        if 'snow' in self.de_type:
            self._init_snow_ids()

        random.shuffle(self.de_type)
    
    def _init_clean_ids(self):
        ref_file = self.opt['data_file_dir'] + "noisy/denoise.txt"
        clean_ids = []
        clean_ids+= [id_.strip() for id_ in open(ref_file)]
        

        if 'denoise_15' in self.de_type:
            self.s15_ids = [{"clean_id": x,"de_type":0} for x in clean_ids]
            self.s15_ids = self.s15_ids * self.scale_dict['denoise_15']
            random.shuffle(self.s15_ids)
            self.s15_counter = 0
        if 'denoise_25' in self.de_type:
            self.s25_ids = [{"clean_id": x,"de_type":1} for x in clean_ids]
            self.s25_ids = self.s25_ids * self.scale_dict['denoise_25']
            random.shuffle(self.s25_ids)
            self.s25_counter = 0
        if 'denoise_50' in self.de_type:
            self.s50_ids = [{"clean_id": x,"de_type":2} for x in clean_ids]
            self.s50_ids = self.s50_ids * self.scale_dict['denoise_50']
            random.shuffle(self.s50_ids)
            self.s50_counter = 0

        self.num_clean = len(clean_ids)
        print("Total Denoise Ids : {}".format(self.num_clean))

    def _init_hazy_ids(self):
        temp_ids = []
        hazy = self.opt['data_file_dir'] + "hazy/hazy.txt"
        temp_ids+= [ id_.strip() for id_ in open(hazy)]
        self.hazy_ids = [{"clean_id" : x,"de_type":4} for x in temp_ids]
        self.hazy_ids = self.hazy_ids * self.scale_dict['dehaze']
        self.hazy_counter = 0
        
        self.num_hazy = len(self.hazy_ids)
        print("Total Hazy Ids : {}".format(self.num_hazy))

    def _init_deblur_ids(self):
        temp_ids = []

        image_list = self.opt['data_file_dir'] +"bulr/bulr.txt"
        temp_ids = [ id_.strip() for id_ in open(image_list)]
        self.deblur_ids = [{"clean_id" : x,"de_type":5} for x in temp_ids]
        self.deblur_ids = self.deblur_ids * 1
        self.deblur_counter = 0
        self.num_deblur = len(self.deblur_ids)*self.scale_dict['deblur']
        print('Total Blur Ids : {}'.format(self.num_deblur))

    def _init_enhance_ids(self):
        temp_ids = []
        image_list = self.opt['data_file_dir'] +"ll/ll.txt"
        temp_ids = [ id_.strip() for id_ in open(image_list)]
        self.enhance_ids= [{"clean_id" : x,"de_type":6} for x in temp_ids]
        self.enhance_ids = self.enhance_ids *self.scale_dict['enhance']
        self.num_enhance = len(self.enhance_ids)
        print('Total enhance Ids : {}'.format(self.num_enhance))

    def _init_rs_ids(self):
        temp_ids = []
        rs = self.opt['data_file_dir'] + "rainy/rainTrain.txt"
        temp_ids+= [ id_.strip() for id_ in open(rs)]
        self.rs_ids = [{"clean_id":x,"de_type":3} for x in temp_ids]
        self.rs_ids = self.rs_ids * self.scale_dict['derain']

        self.rl_counter = 0
        self.num_rl = len(self.rs_ids)
        print("Total Rainy Ids : {}".format(self.num_rl))
    
    def _init_snow_ids(self):
        temp_ids = []
        snow = self.opt['data_file_dir'] + "snow/snow_train.txt"
        temp_ids+= [ id_.strip() for id_ in open(snow)]
        self.snow_ids = [{"clean_id":x,"de_type":7} for x in temp_ids]
        self.snow_ids = self.snow_ids * self.scale_dict['snow']

        self.snow_counter = 0
        self.num_snow = len(self.snow_ids)
        print("Total Snow Ids : {}".format(self.num_snow))

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        if H == self.opt['gt_size'] and W == self.opt['gt_size']:
            return img_1, img_2
        else:
            ind_H = random.randint(0, H - self.opt['gt_size'])
            ind_W = random.randint(0, W - self.opt['gt_size'])

            patch_1 = img_1[ind_H:ind_H + self.opt['gt_size'], ind_W:ind_W + self.opt['gt_size']]
            patch_2 = img_2[ind_H:ind_H + self.opt['gt_size'], ind_W:ind_W + self.opt['gt_size']]

            return patch_1, patch_2

    def _get_gt_name(self, rainy_name):
        gt_name = rainy_name.split("rainy")[0] + 'gt/norain-' + rainy_name.split('rain-')[-1]
        return gt_name


    def _get_deblur_name(self, deblur_name):
        gt_name = deblur_name.replace("blur", "sharp")
        return gt_name
    

    def _get_enhance_name(self, enhance_name):
        gt_name = enhance_name.replace("low", "gt")
        return gt_name


    def _get_nonhazy_name(self, hazy_name):
        dir_name = hazy_name.split("synthetic")[0] + 'original/'
        name = hazy_name.split('/')[-1].split('_')[0]
        suffix = '.' + hazy_name.split('.')[-1]
        nonhazy_name = dir_name + name + suffix
        return nonhazy_name

    def _merge_ids(self):
        self.sample_ids = []
        if "denoise_15" in self.de_type:
            self.sample_ids += self.s15_ids
        if "denoise_25" in self.de_type:
            self.sample_ids += self.s25_ids
        if "denoise_50" in self.de_type:
            self.sample_ids += self.s50_ids
        if "derain" in self.de_type:
            self.sample_ids+= self.rs_ids
        
        if "dehaze" in self.de_type:
            self.sample_ids+= self.hazy_ids
        if "deblur" in self.de_type:
            self.sample_ids += self.deblur_ids
        if "enhance" in self.de_type:
            self.sample_ids += self.enhance_ids
        if "snow" in self.de_type:
            self.sample_ids += self.snow_ids


        print(len(self.sample_ids))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        index = index % len(self.sample_ids)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32. 

        sample = self.sample_ids[index]
        de_id = sample["de_type"]
        if de_id < 3:
            if de_id == 0:
                clean_id = sample["clean_id"]
            elif de_id == 1:
                clean_id = sample["clean_id"]
            elif de_id == 2:
                clean_id = sample["clean_id"]

            num = random.randint(0, int(9))
            clean_name = clean_id.split("/")[-1]

            clean_img = crop_img(np.array(Image.open(os.path.join(self.opt['denoise_dir'],  str(num)+'_'+clean_name)).convert('RGB')), base=16)
            clean_patch = self.crop_transform(clean_img)
            clean_patch= np.array(clean_patch)

            

            clean_patch = random_augmentation(clean_patch)[0]

            degrad_patch = self.D.single_degrade(clean_patch, de_id)
        else:
            if de_id == 3:
                # Rain Streak Removal
                num =random.randint(0,int(9))
                degrad_img = np.array(Image.open(os.path.join(self.opt['derain_dir'], 'input/', str(num)+'_'+sample["clean_id"])).convert('RGB'))
                clean_img = np.array(Image.open(os.path.join(self.opt['derain_dir'], 'gt/', str(num)+'_'+sample["clean_id"])).convert('RGB'))
                clean_name = os.path.join(self.opt['derain_dir'], 'input/', str(num)+'_'+sample["clean_id"])
            elif de_id == 4:
                # Dehazing with SOTS outdoor training set
                num =random.randint(0,int(9))
                degrad_img = np.array(Image.open(os.path.join(self.opt['dehaze_dir'], 'input/', str(num)+'_'+sample["clean_id"])).convert('RGB'))
                clean_img = np.array(Image.open(os.path.join(self.opt['dehaze_dir'], 'gt/', str(num)+'_'+sample["clean_id"])).convert('RGB'))

                clean_name = os.path.join( self.opt['dehaze_dir'], 'input/', str(num)+'_'+sample["clean_id"])
                
            elif de_id == 5:
                # Deblur with bulr set
                num =random.randint(0,int(9))
                degrad_img = np.array(Image.open(os.path.join(self.opt['bulr_dir'], 'input/',  str(num)+'_'+sample["clean_id"])).convert('RGB'))
                clean_img = np.array(Image.open(os.path.join(self.opt['bulr_dir'], 'gt/',  str(num)+'_'+sample["clean_id"])).convert('RGB'))
                clean_name = self._get_deblur_name(sample["clean_id"])
            elif de_id == 6:
                # Enhancement with LOL training set
                num =random.randint(0,int(9))
                
                degrad_img =np.array(Image.open(os.path.join(self.opt['enhance_dir'], 'input/', str(num)+'_'+sample["clean_id"])).convert('RGB'))
                clean_img =np.array(Image.open(os.path.join(self.opt['enhance_dir'], 'gt/', str(num)+'_'+sample["clean_id"])).convert('RGB'))
                clean_name = self._get_enhance_name(sample["clean_id"])
            elif de_id == 7:
                # Snow
                num =random.randint(0,int(9))
                degrad_img = np.array(Image.open(os.path.join(self.opt['snow_dir'], 'input/', str(num)+'_'+sample["clean_id"])).convert('RGB'))
                clean_img = np.array(Image.open(os.path.join(self.opt['snow_dir'], 'gt/', str(num)+'_'+sample["clean_id"])).convert('RGB'))
                clean_name = os.path.join(self.opt['snow_dir'], 'input/', str(num)+'_'+sample["clean_id"])

            degrad_patch, clean_patch = random_augmentation(*self._crop_patch(degrad_img, clean_img))
            # degrad_patch, clean_patch = self._crop_patch(degrad_img, clean_img)

        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degrad_patch)
            

        return {
            'lq': degrad_patch,
            'gt': clean_patch,
            'gt_path': clean_name
        }

    def __len__(self):
        return len(self.sample_ids)


class AirNetDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    opt:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(AirNetDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.rs_ids = []
        self.hazy_ids = []
        self.D = Degradation(opt)
        self.de_temp = 0
        de_type_scale = self.opt['de_type_scale']
        self.de_type = self.opt['de_type']
        print(self.de_type)

        self.de_dict = {'denoise_15': 0, 'denoise_25': 1, 'denoise_50': 2, 'derain': 3, 'dehaze': 4, 'deblur' : 5, 'enhance' : 6,'snow':7}
        self.scale_dict = dict(zip(self.de_type, de_type_scale))

        self._init_ids()
        self._merge_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(opt['gt_size']),
        ])

        self.toTensor = ToTensor()

    
    def _init_ids(self):
        if 'denoise_15' in self.de_type or 'denoise_25' in self.de_type or 'denoise_50' in self.de_type:
            self._init_clean_ids()
        if 'derain' in self.de_type:
            self._init_rs_ids()
        if 'dehaze' in self.de_type:
            self._init_hazy_ids()
        if 'deblur' in self.de_type:
            self._init_deblur_ids()
        if 'enhance' in self.de_type:
            self._init_enhance_ids()
        if 'snow' in self.de_type:
            self._init_snow_ids()

        random.shuffle(self.de_type)
    
    def _init_clean_ids(self):
        ref_file = self.opt['data_file_dir'] + "noisy/denoise.txt"
        clean_ids = []
        clean_ids+= [id_.strip() for id_ in open(ref_file)]
        

        if 'denoise_15' in self.de_type:
            self.s15_ids = [{"clean_id": x,"de_type":0} for x in clean_ids]
            self.s15_ids = self.s15_ids * self.scale_dict['denoise_15']
            random.shuffle(self.s15_ids)
            self.s15_counter = 0
        if 'denoise_25' in self.de_type:
            self.s25_ids = [{"clean_id": x,"de_type":1} for x in clean_ids]
            self.s25_ids = self.s25_ids * self.scale_dict['denoise_25']
            random.shuffle(self.s25_ids)
            self.s25_counter = 0
        if 'denoise_50' in self.de_type:
            self.s50_ids = [{"clean_id": x,"de_type":2} for x in clean_ids]
            self.s50_ids = self.s50_ids * self.scale_dict['denoise_50']
            random.shuffle(self.s50_ids)
            self.s50_counter = 0

        self.num_clean = len(clean_ids)
        print("Total Denoise Ids : {}".format(self.num_clean))

    def _init_hazy_ids(self):
        temp_ids = []
        hazy = self.opt['data_file_dir'] + "hazy/hazy.txt"
        temp_ids+= [ id_.strip() for id_ in open(hazy)]
        self.hazy_ids = [{"clean_id" : x,"de_type":4} for x in temp_ids]
        self.hazy_ids = self.hazy_ids * self.scale_dict['dehaze']
        self.hazy_counter = 0
        
        self.num_hazy = len(self.hazy_ids)
        print("Total Hazy Ids : {}".format(self.num_hazy))

    def _init_deblur_ids(self):
        temp_ids = []

        image_list = self.opt['data_file_dir'] +"bulr/bulr.txt"
        temp_ids = [ id_.strip() for id_ in open(image_list)]
        self.deblur_ids = [{"clean_id" : x,"de_type":5} for x in temp_ids]
        self.deblur_ids = self.deblur_ids * 1
        self.deblur_counter = 0
        self.num_deblur = len(self.deblur_ids)*self.scale_dict['deblur']
        print('Total Blur Ids : {}'.format(self.num_deblur))

    def _init_enhance_ids(self):
        temp_ids = []
        image_list = self.opt['data_file_dir'] +"ll/ll.txt"
        temp_ids = [ id_.strip() for id_ in open(image_list)]
        self.enhance_ids= [{"clean_id" : x,"de_type":6} for x in temp_ids]
        self.enhance_ids = self.enhance_ids *self.scale_dict['enhance']
        self.num_enhance = len(self.enhance_ids)
        print('Total enhance Ids : {}'.format(self.num_enhance))

    def _init_rs_ids(self):
        temp_ids = []
        rs = self.opt['data_file_dir'] + "rainy/rainTrain.txt"
        temp_ids+= [ id_.strip() for id_ in open(rs)]
        self.rs_ids = [{"clean_id":x,"de_type":3} for x in temp_ids]
        self.rs_ids = self.rs_ids * self.scale_dict['derain']

        self.rl_counter = 0
        self.num_rl = len(self.rs_ids)
        print("Total Rainy Ids : {}".format(self.num_rl))
    
    def _init_snow_ids(self):
        temp_ids = []
        snow = self.opt['data_file_dir'] + "snow/snow_train.txt"
        temp_ids+= [ id_.strip() for id_ in open(snow)]
        self.snow_ids = [{"clean_id":x,"de_type":7} for x in temp_ids]
        self.snow_ids = self.snow_ids * self.scale_dict['snow']

        self.snow_counter = 0
        self.num_snow = len(self.snow_ids)
        print("Total Snow Ids : {}".format(self.num_snow))

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        if H == self.opt['gt_size'] and W == self.opt['gt_size']:
            return img_1, img_2
        else:
            ind_H = random.randint(0, H - self.opt['gt_size'])
            ind_W = random.randint(0, W - self.opt['gt_size'])

            patch_1 = img_1[ind_H:ind_H + self.opt['gt_size'], ind_W:ind_W + self.opt['gt_size']]
            patch_2 = img_2[ind_H:ind_H + self.opt['gt_size'], ind_W:ind_W + self.opt['gt_size']]

            return patch_1, patch_2

    def _get_gt_name(self, rainy_name):
        gt_name = rainy_name.split("rainy")[0] + 'gt/norain-' + rainy_name.split('rain-')[-1]
        return gt_name


    def _get_deblur_name(self, deblur_name):
        gt_name = deblur_name.replace("blur", "sharp")
        return gt_name
    

    def _get_enhance_name(self, enhance_name):
        gt_name = enhance_name.replace("low", "gt")
        return gt_name


    def _get_nonhazy_name(self, hazy_name):
        dir_name = hazy_name.split("synthetic")[0] + 'original/'
        name = hazy_name.split('/')[-1].split('_')[0]
        suffix = '.' + hazy_name.split('.')[-1]
        nonhazy_name = dir_name + name + suffix
        return nonhazy_name

    def _merge_ids(self):
        self.sample_ids = []
        if "denoise_15" in self.de_type:
            self.sample_ids += self.s15_ids
        if "denoise_25" in self.de_type:
            self.sample_ids += self.s25_ids
        if "denoise_50" in self.de_type:
            self.sample_ids += self.s50_ids
        if "derain" in self.de_type:
            self.sample_ids+= self.rs_ids
        
        if "dehaze" in self.de_type:
            self.sample_ids+= self.hazy_ids
        if "deblur" in self.de_type:
            self.sample_ids += self.deblur_ids
        if "enhance" in self.de_type:
            self.sample_ids += self.enhance_ids
        if "snow" in self.de_type:
            self.sample_ids += self.snow_ids


        print(len(self.sample_ids))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        index = index % len(self.sample_ids)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32. 

        sample = self.sample_ids[index]
        de_id = sample["de_type"]
        if de_id < 3:
            if de_id == 0:
                clean_id = sample["clean_id"]
            elif de_id == 1:
                clean_id = sample["clean_id"]
            elif de_id == 2:
                clean_id = sample["clean_id"]

            num = random.randint(0, int(9))
            clean_name = clean_id.split("/")[-1]

            clean_img = crop_img(np.array(Image.open(os.path.join(self.opt['denoise_dir'],  str(num)+'_'+clean_name)).convert('RGB')), base=16)
            clean_patch_1, clean_patch_2 = self.crop_transform(clean_img), self.crop_transform(clean_img)
            clean_patch_1, clean_patch_2 = np.array(clean_patch_1), np.array(clean_patch_2)

            

            clean_patch_1, clean_patch_2 = random_augmentation(clean_patch_1, clean_patch_2)
            degrad_patch_1, degrad_patch_2 = self.D.degrade(clean_patch_1, clean_patch_2, de_id)
        else:
            if de_id == 3:
                # Rain Streak Removal
                num =random.randint(0,int(9))
                degrad_img = np.array(Image.open(os.path.join(self.opt['derain_dir'], 'input/', str(num)+'_'+sample["clean_id"])).convert('RGB'))
                clean_img = np.array(Image.open(os.path.join(self.opt['derain_dir'], 'gt/', str(num)+'_'+sample["clean_id"])).convert('RGB'))
                clean_name = os.path.join(self.opt['derain_dir'], 'input/', str(num)+'_'+sample["clean_id"])
            elif de_id == 4:
                # Dehazing with SOTS outdoor training set
                num =random.randint(0,int(9))
                degrad_img = np.array(Image.open(os.path.join(self.opt['dehaze_dir'], 'input/', str(num)+'_'+sample["clean_id"])).convert('RGB'))
                clean_img = np.array(Image.open(os.path.join(self.opt['dehaze_dir'], 'gt/', str(num)+'_'+sample["clean_id"])).convert('RGB'))

                clean_name = os.path.join( self.opt['dehaze_dir'], 'input/', str(num)+'_'+sample["clean_id"])
                
            elif de_id == 5:
                # Deblur with bulr set
                num =random.randint(0,int(9))
                degrad_img = np.array(Image.open(os.path.join(self.opt['bulr_dir'], 'input/',  str(num)+'_'+sample["clean_id"])).convert('RGB'))
                clean_img = np.array(Image.open(os.path.join(self.opt['bulr_dir'], 'gt/',  str(num)+'_'+sample["clean_id"])).convert('RGB'))
                clean_name = self._get_deblur_name(sample["clean_id"])
            elif de_id == 6:
                # Enhancement with LOL training set
                num =random.randint(0,int(9))
                
                degrad_img =np.array(Image.open(os.path.join(self.opt['enhance_dir'], 'input/', str(num)+'_'+sample["clean_id"])).convert('RGB'))
                clean_img =np.array(Image.open(os.path.join(self.opt['enhance_dir'], 'gt/', str(num)+'_'+sample["clean_id"])).convert('RGB'))
                clean_name = self._get_enhance_name(sample["clean_id"])
            elif de_id == 7:
                # Snow
                num =random.randint(0,int(9))
                degrad_img = np.array(Image.open(os.path.join(self.opt['snow_dir'], 'input/', str(num)+'_'+sample["clean_id"])).convert('RGB'))
                clean_img = np.array(Image.open(os.path.join(self.opt['snow_dir'], 'gt/', str(num)+'_'+sample["clean_id"])).convert('RGB'))
                clean_name = os.path.join(self.opt['snow_dir'], 'input/', str(num)+'_'+sample["clean_id"])

            degrad_patch_1, clean_patch_1 = random_augmentation(*self._crop_patch(degrad_img, clean_img))
            degrad_patch_2, clean_patch_2 = random_augmentation(*self._crop_patch(degrad_img, clean_img))
            # degrad_patch, clean_patch = self._crop_patch(degrad_img, clean_img)

        clean_patch_1, clean_patch_2 = self.toTensor(clean_patch_1), self.toTensor(clean_patch_2)
        degrad_patch_1, degrad_patch_2 = self.toTensor(degrad_patch_1), self.toTensor(degrad_patch_2)
            

        return [clean_name, de_id], degrad_patch_1, degrad_patch_2, clean_patch_1, clean_patch_2

    def __len__(self):
        return len(self.sample_ids)

class Dataset_GaussianDenoising(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    opt:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_GaussianDenoising, self).__init__()
        self.opt = opt

        if self.opt['phase'] == 'train':
            self.sigma_type = opt['sigma_type']
            self.sigma_range = opt['sigma_range']
            assert self.sigma_type in ['constant', 'random', 'choice']
        else:
            self.sigma_test = opt['sigma_test']
        self.in_ch = opt['in_ch']

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder = opt['dataroot_gt']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            self.paths = paths_from_lmdb(self.gt_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [
                    osp.join(self.gt_folder,
                             line.split(' ')[0]) for line in fin
                ]
        else:
            self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')

        if self.in_ch == 3:
            try:
                img_gt = imfrombytes(img_bytes, float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        else:
            try:
                img_gt = imfrombytes(img_bytes, flag='grayscale', float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = np.expand_dims(img_gt, axis=2)
        img_lq = img_gt.copy()

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)
            # flip, rotation
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

            img_gt, img_lq = img2tensor([img_gt, img_lq],
                                        bgr2rgb=False,
                                        float32=True)

            if self.sigma_type == 'constant':
                sigma_value = self.sigma_range
            elif self.sigma_type == 'random':
                sigma_value = random.uniform(self.sigma_range[0], self.sigma_range[1])
            elif self.sigma_type == 'choice':
                sigma_value = random.choice(self.sigma_range)

            noise_level = torch.FloatTensor([sigma_value]) / 255.0
            # noise_level_map = torch.ones((1, img_lq.size(1), img_lq.size(2))).mul_(noise_level).float()
            noise = torch.randn(img_lq.size()).mul_(noise_level).float()
            img_lq.add_(noise)

        else:
            np.random.seed(seed=0)
            img_lq += np.random.normal(0, self.sigma_test / 255.0, img_lq.shape)
            # noise_level_map = torch.ones((1, img_lq.shape[0], img_lq.shape[1])).mul_(self.sigma_test/255.0).float()

            img_gt, img_lq = img2tensor([img_gt, img_lq],
                                        bgr2rgb=False,
                                        float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': gt_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)


class Dataset_DefocusDeblur_DualPixel_16bit(data.Dataset):
    def __init__(self, opt):
        super(Dataset_DefocusDeblur_DualPixel_16bit, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lqL_folder, self.lqR_folder = opt['dataroot_gt'], opt['dataroot_lqL'], opt['dataroot_lqR']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.paths = paired_DP_paths_from_folder(
            [self.lqL_folder, self.lqR_folder, self.gt_folder], ['lqL', 'lqR', 'gt'],
            self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = self.opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lqL_path = self.paths[index]['lqL_path']
        img_bytes = self.file_client.get(lqL_path, 'lqL')
        try:
            img_lqL = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("lqL path {} not working".format(lqL_path))

        lqR_path = self.paths[index]['lqR_path']
        img_bytes = self.file_client.get(lqR_path, 'lqR')
        try:
            img_lqR = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("lqR path {} not working".format(lqR_path))

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_lqL, img_lqR, img_gt = padding_DP(img_lqL, img_lqR, img_gt, gt_size)

            # random crop
            img_lqL, img_lqR, img_gt = paired_random_crop_DP(img_lqL, img_lqR, img_gt, gt_size, scale, gt_path)

            # flip, rotation
            if self.geometric_augs:
                img_lqL, img_lqR, img_gt = random_augmentation(img_lqL, img_lqR, img_gt)
        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lqL, img_lqR, img_gt = img2tensor([img_lqL, img_lqR, img_gt],
                                              bgr2rgb=True,
                                              float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lqL, self.mean, self.std, inplace=True)
            normalize(img_lqR, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        img_lq = torch.cat([img_lqL, img_lqR], 0)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lqL_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)
