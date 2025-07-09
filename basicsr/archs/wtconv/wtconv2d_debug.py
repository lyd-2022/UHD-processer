import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/code/UHD-allinone')
from basicsr.archs.wtconv.util import wavelet
import cv2
from PIL import Image
import numpy as np
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor, Grayscale
class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = wavelet.create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        
        self.wt_function = wavelet.wavelet_transform_init(self.wt_filter)
        self.iwt_function = wavelet.inverse_wavelet_transform_init(self.iwt_filter)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1,in_channels,1,1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels*4, in_channels*4, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels*4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1,in_channels*4,1,1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride, groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:,:,0,:,:]
            
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            # curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:,:,0,:,:])
            x_h_in_levels.append(curr_x_tag[:,:,1:4,:,:])

        # next_x_ll = 0

        # for i in range(self.wt_levels-1, -1, -1):
        #     curr_x_ll = x_ll_in_levels.pop()
        #     curr_x_h = x_h_in_levels.pop()
        #     curr_shape = shapes_in_levels.pop()

        #     curr_x_ll = curr_x_ll + next_x_ll

        #     curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
        #     next_x_ll = self.iwt_function(curr_x)

        #     next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        # x_tag = next_x_ll
        # assert len(x_ll_in_levels) == 0
        
        # x = self.base_scale(self.base_conv(x))
        # x = x + x_tag
        
        # if self.do_stride is not None:
        #     x = self.do_stride(x)

        return x_ll_in_levels,x_h_in_levels

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None
    
    def forward(self, x):
        return torch.mul(self.weight, x)
    

# 测试模块
if __name__ == '__main__':
    wtconv = WTConv2d(3, 3, wt_levels=4,stride=2)
    #读取图片转换为tensor
    # image_path = '/model/liuyidi/VAE/UHD-allinone/vis/ori/ft_002_gt.png' 
    # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = image / 255.0
    # image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
    
    # image_path2 = '/model/liuyidi/VAE/UHD-allinone/vis/ori/ft_002.png'
    # image2 = cv2.imread(image_path2, cv2.IMREAD_COLOR)
    # image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    # image2 = image2 / 255.0
    # image2 = torch.tensor(image2, dtype=torch.float32).permute(2, 0, 1)
    # x = torch.stack([image,image2],dim=0)

    image_path = '/model/liuyidi/VAE/UHD-allinone/vis/ori/ft_001_gt.png' 
    image = np.array(Image.open(image_path).convert('RGB'))
    image = ToTensor()(image).unsqueeze(0)[0].cpu().numpy()
    np1 = np.load('/model/liuyidi/VAE/UHD-allinone/vis/ori/ft_001_gt.npy')
    np1 =  np.transpose(np1, (1, 2, 0))
    np1 = ToTensor()(np1).unsqueeze(0)

    #判断np1和image是否相等


    image_path2 = '/model/liuyidi/VAE/UHD-allinone/vis/ori/ft_001.png'
    image2 = np.array(Image.open(image_path2).convert('RGB'))
    image2 = ToTensor()(image2).unsqueeze(0)
    np2 = np.load('/model/liuyidi/VAE/UHD-allinone/vis/ori/ft_001.npy')
    np2 =  np.transpose(np2, (1, 2, 0))

    np2 = ToTensor()(np2).unsqueeze(0)
    # x = torch.cat([image,image2],dim=0)
    x = torch.cat([np1,np2],dim=0)



    np3 = np.load('/model/liuyidi/VAE/UHD-allinone/vis/level_0/ft_1_0_gt.npy')
    np3 =  np.transpose(np3, (1, 2, 0))
    with torch.no_grad():
        y,y2 = wtconv(x)
    #将tensor保存为图片
    for i in range(len(y)):
       for j in range(len(y[i])):
           m = y[i][j].permute(1,2,0).detach().numpy()
           m = cv2.cvtColor(m, cv2.COLOR_RGB2BGR)
           cv2.imwrite(f'/model/liuyidi/VAE/UHD-allinone/vis/28_UHD_LL_{i}_{j}.JPG',m*255) 