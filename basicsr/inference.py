import argparse
import cv2
import glob
import os
from tqdm import tqdm
import torch
from yaml import load
import sys
import numpy as np
from basicsr.utils import img2tensor, tensor2img, imwrite
from basicsr.archs.femasr_arch import FeMaSRNet
# from basicsr.archs.SAFMN_arch import SAFMN
# from basicsr.archs.SAFMN2_arch import SAFMN2

# from basicsr.archs.SAFMN_ori_arch import SAFMN_ori
from basicsr.utils.download_util import load_file_from_url
from basicsr.archs import build_network
from PIL import Image
from torchvision.transforms import ToTensor
import math
import torch

_ = torch.manual_seed(123)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex')
import yaml

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# from comput_psnr_ssim import calculate_ssim as ssim_gray
# from comput_psnr_ssim import calculate_psnr as psnr_gray

def ssim_gray(imgA, imgB, gray_scale=True):
    if gray_scale:
        score, diff = ssim(cv2.cvtColor(imgA, cv2.COLOR_RGB2GRAY), cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY), full=True,
                           multichannel=False)
    # multichannel: If True, treat the last dimension of the array as channels. Similarity calculations are done independently for each channel then averaged.
    else:
        score, diff = ssim(imgA, imgB, full=True, multichannel=True)
    return score


def psnr_gray(imgA, imgB, gray_scale=True):
    if gray_scale:
        psnr_val = psnr(cv2.cvtColor(imgA, cv2.COLOR_RGB2GRAY), cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY))
        return psnr_val
    else:
        psnr_val = psnr(imgA, imgB)
        return psnr_val


# pretrain_model_url = {
#     'x4': 'https://github.com/chaofengc/FeMaSR/releases/download/v0.1-pretrain_models/FeMaSR_SRX4_model_g.pth',
#     'x2': 'https://github.com/chaofengc/FeMaSR/releases/download/v0.1-pretrain_models/FeMaSR_SRX2_model_g.pth',
# }

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def equalize_hist_color(img):
    # 使用 cv2.split() 分割 BGR 图像
    channels = cv2.split(img)
    eq_channels = []
    # 将 cv2.equalizeHist() 函数应用于每个通道
    for ch in channels:
        eq_channels.append(cv2.equalizeHist(ch))
    # 使用 cv2.merge() 合并所有结果通道
    eq_image = cv2.merge(eq_channels)
    return eq_image

    # def get_residue_structure_mean(self, tensor, r_dim=1):
    #     max_channel = torch.max(tensor, dim=r_dim, keepdim=True)  # keepdim
    #     min_channel = torch.min(tensor, dim=r_dim, keepdim=True)
    #     res_channel = (max_channel[0] - min_channel[0])
    #     mean = torch.mean(tensor, dim=r_dim, keepdim=True)
    #
    #     device = mean.device
    #     res_channel = res_channel / torch.max(mean, torch.full(size=mean.size(), fill_value=0.000001).to(device))
    #     return res_channel

def get_residue_structure_mean(tensor, r_dim=1):
    max_channel = torch.max(tensor, dim=r_dim, keepdim=True)  # keepdim
    min_channel = torch.min(tensor, dim=r_dim, keepdim=True)
    res_channel = (max_channel[0] - min_channel[0])
    mean = torch.mean(tensor, dim=r_dim, keepdim=True)
    device = mean.device
    res_channel = res_channel / torch.max(mean, torch.full(size=mean.size(), fill_value=0.000001).to(device))
    return res_channel
import torch.nn.functional as F
def check_image_size(x,window_size=128):
    _, _, h, w = x.size()
    mod_pad_h = (window_size  - h % (window_size)) % (
                window_size )
    mod_pad_w = (window_size  - w % (window_size)) % (
                window_size)
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    # print('F.pad(x, (0, mod_pad_w, 0, mod_pad_h)', x.size())
    return x

def print_network(model):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print("The number of parameters: {}".format(num_params))
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'


@torch.no_grad()
def test_tile(model, input, tile_size=512, tile_pad=16):
    # return self.test(input)
    """It will first crop input images to tiles, and then process each tile.
    Finally, all the processed tiles are merged into one images.
    Modified from: https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py
    """
    batch, channel, height, width = input.shape
    output_height = height 
    output_width = width 
    output_shape = (batch, channel, output_height, output_width)

    # start with black image
    output = input.new_zeros(output_shape)
    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)

    # loop over all tiles
    for y in range(tiles_y):
        for x in range(tiles_x):
            # extract tile from input image
            ofs_x = x * tile_size
            ofs_y = y * tile_size
            # input tile area on total image
            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile_size, width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile_size, height)

            # input tile area on total image with padding
            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, width)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, height)

            # input tile dimensions
            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y
            tile_idx = y * tiles_x + x + 1
            input_tile = input[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

            # upscale tile
            output_tile = model(input_tile)
            # print(output_tile.size())

            # output tile area on total image
            output_start_x = input_start_x 
            output_end_x = input_end_x 
            output_start_y = input_start_y 
            output_end_y = input_end_y 

            # output tile area without padding
            output_start_x_tile = (input_start_x - input_start_x_pad) 
            output_end_x_tile = output_start_x_tile + input_tile_width 
            output_start_y_tile = (input_start_y - input_start_y_pad) 
            output_end_y_tile = output_start_y_tile + input_tile_height 

            # put tile into output image
            output[:, :, output_start_y:output_end_y,
            output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                            output_start_x_tile:output_end_x_tile]
    return output
def _add_gaussian_noise(clean_patch, sigma):

        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch
def main():
    """Inference demo for FeMaSR
    """
    parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--input', type=str, default='/data_8T1/wangcong/dataset/Rain13K/rain13ktest/Rain100H/input',
    #                     help='Input image or folder')
    # parser.add_argument('-g', '--gt', type=str, default='/data_8T1/wangcong/dataset/Rain13K/rain13ktest/Rain100H/target',
    #                     help='groundtruth image')
    # parser.add_argument('-i', '--input', type=str,
    #                     default='/data_8T1/wangcong/dataset/real-world-images/real-input',
    #                     help='Input image or folder')
    # parser.add_argument('-g', '--gt', type=str,
    #                     default='/data_8T1/wangcong/dataset/real-world-images/real-input',
    #                     help='groundtruth image')
    parser.add_argument('-i', '--input', type=str,
                        default='/data/liuyidi/nitre_2023_dehaze/data_dehaze/UHD_noise/test',
                        help='Input image or folder')
    parser.add_argument('-g', '--gt', type=str,
                        default='/data/liuyidi/nitre_2023_dehaze/data_dehaze/UHD_LL/testing_set/gt',
                        help='groundtruth image')
    # parser.add_argument('-i', '--input', type=str,
    #                     default='/data_8T1/wangcong/dataset/LOLdataset/eval15/low',
    #                     help='Input image or folder')
    # parser.add_argument('-g', '--gt', type=str,
    #                     default='/data_8T1/wangcong/dataset/LOLdataset/eval15/high',
    #                     help='groundtruth image')
    # parser.add_argument('-w_vqgan', '--weight_vqgan', type=str,
    #                     default='/data_8T1/wangcong/net_g_260000.pth',
    #                     help='path for model weights')
    parser.add_argument('-w', '--weight', type=str,
                        default='/home/zhuqi/lyd/UHDformer-main/experiments/SAF_adapter_kl8/models/net_g_latest.pth',
                        help='path for model weights')
    parser.add_argument('-o', '--output', type=str, default='/data/liuyidi/nitre_2023_dehaze/data_dehaze/UHD_noise/test_noise_50', help='Output folder')
    parser.add_argument('-s', '--out_scale', type=int, default=1, help='The final upsampling scale of the image')
    parser.add_argument('-c', '--config', type=str,default='/code/UHDformer-main/options/adapter/dwt_kl8.yml', help='path to the config file')
    parser.add_argument('--suffix', type=str, default='', help='Suffix of the restored image')
    parser.add_argument('--max_size', type=int, default=600,
                        help='Max image size for whole image inference, otherwise use tiled_test')
    parser.add_argument('--add_gaussian', type=int,default=0, help='Add Gaussian noise')
    
    parser.add_argument('--test_tile', action='store_true', help='Test with tile')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if args.weight is None:
    #     weight_path_vqgan = load_file_from_url(pretrain_model_url[f'x{args.out_scale}'])
    # else:
    #     weight_path_vqgan = args.weight_vqgan
    enhance_weight_path = args.weight
    # print('weight_path', weight_path_vqgan)
    # set up the model
    # VQGAN = FeMaSRNet(codebook_params=[[16, 1024, 256], [32, 1024, 128], [64, 1024, 64], [128, 1024, 32]], LQ_stage=False, scale_factor=args.out_scale).to(device)
    # VQGAN.load_state_dict(torch.load(weight_path_vqgan)['params'], strict=False)
    # VQGAN.eval()

    # EnhanceNet = SAFMN_ori(dim=48, n_blocks=8, ffn_scale=2.0, upscaling_factor=4).to(device)
    with open (args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    EnhanceNet = build_network(config["network_g"]).to(device)
    if config['train'].get('ema_decay', None):
        a = torch.load(enhance_weight_path)
        print(a.keys())
        EnhanceNet.load_state_dict(torch.load(enhance_weight_path)['params'], strict=True)
    else:
        EnhanceNet.load_state_dict(torch.load(enhance_weight_path)['params'], strict=True)
    EnhanceNet.eval()
    print_network(EnhanceNet)
    os.makedirs(args.output, exist_ok=True)
    if os.path.isfile(args.input):
        paths = [args.input]
        
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))
    ssim_all = 0
    psnr_all = 0
    lpips_all = 0
    num_img = 0
    pbar = tqdm(total=len(paths), unit='image')
    for idx, path in enumerate(paths):
        img_name = os.path.basename(path)
        pbar.set_description(f'Test {img_name}')

        # gt_path = args.gt
        # file_name = path.split('/')[-1]

        # gt_img = cv2.imread(os.path.join(gt_path, file_name), cv2.IMREAD_UNCHANGED)
        print('image name', path)
        # print(gt_img)
        if args.add_gaussian:
            print('add_gaussian')
            img = np.array(Image.open(path))
            img, _ = _add_gaussian_noise(img, args.add_gaussian)
            img_tensor = ToTensor()(img).to(device)
        else:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            img_tensor = img2tensor(img).to(device) / 255.

        img_tensor = img_tensor.unsqueeze(0)
        b, c, h, w = img_tensor.size()
        print('b, c, h, w = img_tensor.size()', img_tensor.size())
        img_tensor = check_image_size(img_tensor)
        # self.gt_rec, feature_degradation, restoration
        # with torch.no_grad():
        #     _, feature_degradation = VQGAN.VQGAN(img_tensor)

        if args.test_tile:
            with torch.no_grad():
                import time
                t0 = time.time()
                output = test_tile(EnhanceNet, img_tensor,1280)
                # output = test_tile(EnhanceNet, img_tensor,128)
                t1 = time.time()
                print('time:', t1-t0)
        else:
            with torch.no_grad():
                import time
                t0 = time.time()
                output = EnhanceNet(img_tensor)
                # output = img_tensor
                if len(output) >1:
                    output = output[0]
                t1 = time.time()
                print('time:', t1-t0)
        output = output
        # output = sr_model.test(img_tensor, rain = img_tensor-output)
        # else:
        #     output = sr_model.test_tile(img_tensor)
        # output_img = output['out_final']

        # [2, 1, 0]
        # output_first = tensor2img(output_first)
        output = output[:, :, :h, :w]
        output_img = tensor2img(output)
        gray = True
        # ssim = ssim_gray(output_img, gt_img, gray_scale=gray)
        # psnr = psnr_gray(output_img, gt_img, gray_scale=gray)
        # ssim = ssim_gray(output_img, gt_img)
        # psnr = psnr_gray(output_img, gt_img)
        # lpips_value = lpips(2 * torch.clip(img2tensor(output_img).unsqueeze(0) / 255.0, 0, 1) - 1,
        #                     2 * img2tensor(gt_img).unsqueeze(0) / 255.0 - 1).data.cpu().numpy()
        # ssim_all += ssim
        # psnr_all += psnr
        # lpips_all += lpips_value
        num_img += 1
        print('num_img', num_img)
        # print('ssim', ssim)
        # print('psnr', psnr)
        # print('lpips_value', lpips_value)
        save_path = os.path.join(args.output, f'{img_name}')
        # save_path_first = os.path.join(args.output + 'first/', f'{img_name}')
        imwrite(output_img, save_path)

        pbar.update(1)
    pbar.close()
    # print('avg_ssim:%f' % (ssim_all / num_img))
    # print('avg_psnr:%f' % (psnr_all / num_img))
    # print('avg_lpips:%f' % (lpips_all / num_img))


if __name__ == '__main__':
    main()
