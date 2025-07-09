import os
from metrics import calculate_psnr, calculate_ssim
import torch
import cv2
import lpips
import numpy as np
from torch.utils.tensorboard import SummaryWriter 
device = torch.device('cuda')
import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"]='0'
parser = argparse.ArgumentParser(description='Calculate PSNR, SSIM, and LPIPS for image comparison.')
parser.add_argument('--gt_path',default='/data/liuyidi/nitre_2023_dehaze/data_dehaze/UHD_LL/testing_set/gt', type=str, help='Path to the ground truth images.')
parser.add_argument('--results_path', type=str, help='Path to the generated results images.')
parser.add_argument('--step', type=int, default=0, help='step.')
parser.add_argument('--test_log', type=str, default='/test.log', help='model.')
args = parser.parse_args()
import sys

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

type = sys.getfilesystemencoding()
gt_path = args.gt_path
results_path = args.results_path
sys.stdout = Logger(os.path.dirname(results_path) + args.test_log)
#构建tensorboard
tb_path = os.path.join(os.path.dirname(results_path), 'tb_test')
# summary_writer = SummaryWriter(tb_path)

lpips_fn = lpips.LPIPS(net='alex').to(device)   ###########LPIPS

imgsName = sorted(os.listdir(results_path))
gtsName = sorted(os.listdir(gt_path))

assert len(imgsName) == len(gtsName)

cumulative_psnr, cumulative_ssim, cumulative_lpips = 0, 0, 0

for i in range(len(imgsName)):
    print('Processing image: %s' % (imgsName[i]))
    res = cv2.imread(os.path.join(results_path, imgsName[i]), cv2.IMREAD_COLOR)
    gt = cv2.imread(os.path.join(gt_path, gtsName[i]), cv2.IMREAD_COLOR)
    cur_psnr = calculate_psnr(res, gt, test_y_channel=True)
    cur_ssim = calculate_ssim(res, gt, test_y_channel=True)

    gt_tensor = torch.tensor(np.array(gt)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    res_tensor = torch.tensor(np.array(res)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    lpips = lpips_fn(gt_tensor.to(device) * 2 - 1, res_tensor.to(device) * 2 - 1).squeeze().item()

    print('PSNR is %.4f and SSIM is %.4f' % (cur_psnr, cur_ssim))
    print('LPIPS is %.4f ' % (lpips))

    cumulative_psnr += cur_psnr
    cumulative_ssim += cur_ssim
    cumulative_lpips += lpips
print(args.step)
print('Testing set, PSNR is %.4f and SSIM is %.4f' % (cumulative_psnr / len(imgsName), cumulative_ssim / len(imgsName)))
print('Testing set, LPIPS is %.4f' % (cumulative_lpips / len(imgsName)))
# summary_writer.add_scalar('PSNR', cumulative_psnr / len(imgsName), int(args.step))
# summary_writer.add_scalar('SSIM', cumulative_ssim / len(imgsName), int(args.step))
# summary_writer.add_scalar('LPIPS', cumulative_lpips / len(imgsName), int(args.step))
print(results_path)
