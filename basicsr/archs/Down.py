import torch
from torch import nn as nn
import torch.nn.functional as F
import numpy as np
import time


def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer

def ComplexResize(x, size, mode='bilinear', align_corners=False):
    # Split the tensor into its real and imaginary parts
    real_part = torch.real(x)
    imag_part = torch.imag(x)
    real_part_resized = F.interpolate(real_part, size=size, mode='bilinear', align_corners=False)
    imag_part_resized = F.interpolate(imag_part, size=size, mode='bilinear', align_corners=False)
    tensor_resized = torch.complex(real_part_resized, imag_part_resized)

    return tensor_resized

def pad_tensor_to_even(t):
    b, c, h, w = t.size()
    pad_h = 0
    pad_w = 0
    if h % 2 != 0:
        pad_h = 1
    if w % 2 != 0:
        pad_w = 1
    if pad_h != 0 or pad_w != 0:
        t = F.pad(t, (0, pad_w, 0, pad_h))
    return t


class FouriDown(nn.Module):

    def __init__(self, in_channel, base_channel):
        super(FouriDown, self).__init__()
        self.real_fuse = nn.Sequential(nn.Conv2d(in_channel*4,in_channel*4,1,1,0,groups=in_channel),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(in_channel*4,in_channel*4,1,1,0,groups=in_channel))
        self.imag_fuse = nn.Sequential(nn.Conv2d(in_channel*4,in_channel*4,1,1,0,groups=in_channel),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(in_channel*4,in_channel*4,1,1,0,groups=in_channel))
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)                                      
        self.Downsample = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)
        self.channel2x = nn.Conv2d(in_channel, base_channel, 1, 1)
        print('FouriDown')

    def forward(self, x):
       
        B, C, H, W = x.shape
        img_fft = torch.fft.fft2(x)
        real = img_fft.real
        imag = img_fft.imag
        mid_row, mid_col = img_fft.shape[2] // 4, img_fft.shape[3] // 4

        # Split into 16 patches
        img_fft_A = img_fft[:, :, :mid_row, :mid_col]
        img_fft_2 = img_fft[:, :, :mid_row, mid_col:mid_col*2]
        img_fft_1 = img_fft[:, :, :mid_row, mid_col*2:mid_col*3]
        img_fft_B = img_fft[:, :, :mid_row, mid_col*3:]
        
        img_fft_5 = img_fft[:, :, mid_row:mid_row*2, :mid_col]
        img_fft_6 = img_fft[:, :, mid_row:mid_row*2, mid_col:mid_col*2]
        img_fft_7 = img_fft[:, :, mid_row:mid_row*2, mid_col*2:mid_col*3]
        img_fft_8 = img_fft[:, :, mid_row:mid_row*2, mid_col*3:]

        img_fft_9 = img_fft[:, :, mid_row*2:mid_row*3, :mid_col]
        img_fft_10 = img_fft[:, :, mid_row*2:mid_row*3, mid_col:mid_col*2]
        img_fft_11 = img_fft[:, :, mid_row*2:mid_row*3, mid_col*2:mid_col*3]
        img_fft_12 = img_fft[:, :, mid_row*2:mid_row*3, mid_col*3:]

        img_fft_C = img_fft[:, :, mid_row*3:, :mid_col]
        img_fft_3 = img_fft[:, :, mid_row*3:, mid_col:mid_col*2]
        img_fft_4 = img_fft[:, :, mid_row*3:, mid_col*2:mid_col*3]
        img_fft_D = img_fft[:, :, mid_row*3:, mid_col*3:]

        # Cluster superposing groups and spectral shuffle
        fuse_A = torch.cat((torch.cat((img_fft_A, img_fft_B), dim=-1), torch.cat((img_fft_C, img_fft_D), dim=-1)), dim=-2)
        fuse_B = torch.cat((torch.cat((img_fft_1, img_fft_2), dim=-1), torch.cat((img_fft_4, img_fft_3), dim=-1)), dim=-2)
        fuse_C = torch.cat((torch.cat((img_fft_9, img_fft_12), dim=-1), torch.cat((img_fft_5, img_fft_8), dim=-1)), dim=-2)
        fuse_D = torch.cat((torch.cat((img_fft_11, img_fft_10), dim=-1), torch.cat((img_fft_7, img_fft_6), dim=-1)), dim=-2)

        tensors = [fuse_A, fuse_B, fuse_C, fuse_D]
        heights = [tensor.shape[2] for tensor in tensors]
        widths = [tensor.shape[3] for tensor in tensors]
        if len(set(heights)) == 1 and len(set(widths)) == 1:
            fuse = torch.stack(tensors, dim=2)
        else:
            for t in tensors:
                resized_tensors = [ComplexResize(t, size=(H//2, W//2), mode='bilinear', align_corners=False) for tensor in tensors]
                fuse = torch.stack(resized_tensors, dim=2)
                
        # Adaptive attention
        fuse = fuse.view(B, 4 * C, H//2, W//2)
        real = fuse.real
        imag = fuse.imag
        real_weight = self.real_fuse(real)
        imag_weight = self.imag_fuse(imag)
        fuse_weight = torch.complex(real_weight, imag_weight)
        fuse_weight = fuse_weight.view(B, C, 4, H // 2, W // 2)
        real_sigmoid = F.softmax(fuse_weight.real+0.25, dim=2)
        imag_sigmoid = F.softmax(fuse_weight.imag+0.25, dim=2)
        fuse_weight =  torch.complex(real_sigmoid, imag_sigmoid)
        fuse = torch.complex(real, imag)
        fuse = fuse.view(B, C, 4, H // 2, W // 2)
        fuse = fuse * fuse_weight

        # Superposing
        fuse = fuse.sum(dim=2)
        img = torch.abs(torch.fft.ifft2(fuse))
        img = img  + self.Downsample(x)
        img =  self.lrelu(self.channel2x(img))
    
        return img


class CutDown(nn.Module):

    def __init__(self, in_channel, base_channel):
        super(CutDown, self).__init__()
        self.channel2x = nn.Conv2d(in_channel, base_channel, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x =  self.lrelu(self.channel2x(x))
        _, _, H, W = x.shape
        img_fft = torch.fft.fft2(x)
        real = img_fft.real
        imag = img_fft.imag

        mid_row, mid_col = imag.shape[2] // 2, imag.shape[3] // 2
        imag_cut = torch.div(imag[:,:, mid_row//2:3*mid_row//2, mid_col//2:3*mid_col//2], 4)
        real_cut = torch.div(real[:,:, mid_row//2:3*mid_row//2, mid_col//2:3*mid_col//2], 4) 
        img = torch.abs(torch.fft.ifft2(torch.complex(real_cut, imag_cut)))

        return img


class BlurDown(nn.Module):
    def __init__(self, in_channel, channels, pad_type='reflect', filt_size=4, stride=2, pad_off=0):
        super(BlurDown, self).__init__()
        self.channel2x = nn.Conv2d(in_channel, channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, x):
        x =  self.lrelu(self.channel2x(x))
        if(self.filt_size==1):
            if(self.pad_off==0):
                return x[:,:,::self.stride,::self.stride]
            else:
                return self.pad(x)[:,:,::self.stride,::self.stride]
        else:
            return F.conv2d(self.pad(x), self.filt, stride=self.stride, groups=x.shape[1])


class unPixelShuffle(nn.Module):
    def __init__(self, downscale_factor=2):
        super(unPixelShuffle, self).__init__()
        self.down_s = downscale_factor

    def forward(self, x):

        x = pad_tensor_to_even(x)
        batch_size, channels, height, width = x.size()
        out_channels = channels * self.down_s * self.down_s

        out_height = height // self.down_s
        out_width = width // self.down_s

        input_view = x.contiguous().view(batch_size, channels, out_height, self.down_s, out_width, self.down_s)

        shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()

        return shuffle_out.view(batch_size, out_channels, out_height, out_width)



if __name__ == '__main__':
    # Test the module
    x = torch.randn(1, 3, 256, 256)
    model = FouriDown(3, 64)
    y = model(x)
    print(y.shape)