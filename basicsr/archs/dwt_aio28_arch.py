import torch
import torch.nn as nn
import torch.nn.functional as F
import sys 
sys.path.append("/fs-computility/ai4sData/liuyidi/code/UHD-processer")
# from basicsr.archs.VAE_arch import AutoencoderKL
import time
import yaml
from basicsr.utils.vae_util import instantiate_from_config
# from utils.distributions.distributions import DiagonalGaussianDistribution
from basicsr.utils.registry import ARCH_REGISTRY
import math
from basicsr.utils.distributions.distributions import DiagonalGaussianDistribution
# from basicsr.archs.encoder import nonlinearity, Normalize, ResnetBlock, make_attn, Downsample, Upsample
from basicsr.archs.wtconv import WTConv2d
from einops import rearrange
# from basicsr.archs.Fourier_Upsampling import freup_Areadinterpolation,freup_AreadinterpolationV2,freup_Cornerdinterpolation,freup_Periodicpadding
from basicsr.archs.wtconv.util import wavelet
from basicsr.archs.merge.gate import GatedFeatureEnhancement
from basicsr.archs.Resblock.Res_four import Res_four,Res_four2,Res_four3,Res_four4,Res_four5,Res_four6,Res_four7,Res_four8,Res_four9,Res_four10,Res_four11,Res_four12

import numbers
import numpy as np  

import torch.fft as fft

# Layer Norm
class fresadd(nn.Module):
    def __init__(self, channels=32,freup_type='pad'):
        super(fresadd, self).__init__()
        if freup_type == 'pad':
            self.Fup = freup_Periodicpadding(channels)
        elif freup_type == 'corner':
            self.Fup = freup_Cornerdinterpolation(channels)
        elif freup_type == 'area':
            self.Fup = freup_Areadinterpolation(channels)
        elif freup_type == 'areaV2':
            self.Fup = freup_AreadinterpolationV2(channels)
        print('freup_type is',freup_type)
        self.fuse = nn.Conv2d(channels, channels,1,1,0)

    def forward(self,x):

        x1 = x
        
        x2 = F.interpolate(x1,scale_factor=2,mode='bilinear')
      
        x3 = self.Fup(x1)
     


        xm = x2 + x3
        xn = self.fuse(xm)

        return xn

def make_res(in_channels, out_channels,temb_channels,dropout, res_type="vanilla"):
    assert res_type in ["vanilla", "Fourmer","MAB","Res_four","Res_four2","Res_four3","Res_four4","Res_four5","Res_four6","Res_four7","Res_four8","Res_four9","Res_four10","Res_four11","Res_four12","none"], f'res_type {res_type} unknown'
    print(f"making res of type '{res_type}' with {in_channels} in_channels")
    if res_type == "vanilla":
        return ResnetBlock(in_channels=in_channels,
                                         out_channels=out_channels,
                                         temb_channels=temb_channels,
                                         dropout=dropout)
    elif res_type == "Fourmer":
        return ProcessBlock(in_channels,out_channels)
    elif res_type == "Res_four":
        return Res_four(in_channels=in_channels,
                                         out_channels=out_channels,
                                         dropout=dropout)
    elif res_type == "Res_four2":
        return Res_four2(in_channels=in_channels,
                                         out_channels=out_channels,
                                         dropout=dropout)
    elif res_type == "Res_four3":
        return Res_four3(in_channels=in_channels,
                                         out_channels=out_channels,
                                         dropout=dropout)
    elif res_type == "Res_four4":
        return Res_four4(in_channels=in_channels,
                                         out_channels=out_channels,
                                         dropout=dropout)
    elif res_type == "Res_four5":
        return Res_four5(in_channels=in_channels,
                                         out_channels=out_channels,
                                         dropout=dropout)
    elif res_type == "Res_four6":
        return Res_four6(in_channels=in_channels,
                                         out_channels=out_channels,
                                         dropout=dropout)
    elif res_type == "Res_four7":
        return Res_four7(in_channels=in_channels,
                                         out_channels=out_channels,
                                         dropout=dropout)
    elif res_type == "Res_four8":
        return Res_four8(in_channels=in_channels,
                                         out_channels=out_channels,
                                         dropout=dropout)
    elif res_type == "Res_four9":
        return Res_four9(in_channels=in_channels,
                                         out_channels=out_channels,
                                         dropout=dropout)
    elif res_type == "Res_four10":
        return Res_four10(in_channels=in_channels,
                                         out_channels=out_channels,
                                         dropout=dropout)
    elif res_type == "Res_four11":
        return Res_four11(in_channels=in_channels,
                                         out_channels=out_channels,
                                         dropout=dropout)
    elif res_type == "Res_four12":
        return Res_four12(in_channels=in_channels,
                                         out_channels=out_channels,
                                         dropout=dropout)
    elif res_type == "MAB":
        return MAB(in_channels, out_channels)

    else:
        return nn.Identity(in_channels)

# Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
    

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None
    



class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# FC
class FC(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.fc = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(), 
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.fc(x)


# Local feature
class Local(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        hidden_dim = int(dim // growth_rate)

        self.weight = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.weight(y)
        return x*y


# Gobal feature
class Gobal(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act2 = nn.GELU()
        self.conv3 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act3 = nn.Sigmoid()

    def forward(self, x):
        _, C, H, W = x.shape
        y = F.interpolate(x, size=[C, C], mode='bilinear', align_corners=True) #[1, 64, 64, 64]
        # b c w h -> b c h w
        y = self.act1(self.conv1(y)).permute(0, 1, 3, 2)
        # b c h w -> b w h c
        y = self.act2(self.conv2(y)).permute(0, 3, 2, 1)
        # b w h c -> b c w h
        y = self.act3(self.conv3(y)).permute(0, 3, 1, 2)
        y = F.interpolate(y, size=[H, W], mode='bilinear', align_corners=True)
        return x*y
    

class AttBlock(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()

        self.norm1 = LayerNorm(dim) 
        self.norm2 = LayerNorm(dim) 

        self.local = Local(dim, ffn_scale)
        self.gobal = Gobal(dim)
        self.conv = nn.Conv2d(2*dim, dim, 1, 1, 0)
        # Feedforward layer
        self.fc = FC(dim, ffn_scale) 

    def forward(self, x):
        y = self.norm1(x)
        y_l = self.local(y)
        y_g = self.gobal(y)
        y = self.conv(torch.cat([y_l, y_g], dim=1)) + x

        y = self.fc(self.norm2(y)) + y
        return y
    

class ResBlock(nn.Module):
    def __init__(self, dim, k=3, s=1, p=1, b=True):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=k, stride=s, padding=p, bias=b)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=k, stride=s, padding=p, bias=b)

    def forward(self, x):
        res = self.conv2(self.act(self.conv1(x)))
        return res + x


class LowRankConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, prompt_size, rank,rank_scale=None,prompt_len=3, stride=1, padding=0, groups=1, bias=False):
        super(LowRankConv2d, self).__init__()
        for i in range(prompt_len):
            if rank_scale:
                setattr(self, f'A{i+1}', nn.Parameter(torch.randn(out_channels, 1, prompt_size, int(rank*rank_scale[i]))))
                setattr(self, f'B{i+1}', nn.Parameter(torch.randn(out_channels, 1, int(rank*rank_scale[i]), prompt_size)))
            else:
                setattr(self, f'A{i+1}', nn.Parameter(torch.randn(out_channels, 1, prompt_size, rank)))
                setattr(self, f'B{i+1}', nn.Parameter(torch.randn(out_channels, 1, rank, prompt_size)))


        # bias
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None
        self.prompt_len = prompt_len
        self.stride = stride
        self.padding = padding
        self.groups = groups
        kernel_size = prompt_size
        # 定义不同的kernel_size和对应的dilation组合
        if kernel_size == 65:
            self.kernel_sizes = [5, 11, 9, 9, 7, 3, 3, 3, 3, 3, 3, 3]
            self.dilates = [1, 2, 4, 3, 2, 5, 6, 7, 8, 9, 12, 16]
        elif kernel_size == 33:
            self.kernel_sizes = [5,5,17,11,9, 9, 7,7,3,3,3,3, 3,3]
            self.dilates =      [1,2,4,5,8,6, 3, 5,2,6,12,16,24,32]
        elif kernel_size == 17:
            self.kernel_sizes = [5, 9, 3, 3, 3]
            self.dilates = [1, 2, 4, 5, 7]
        
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 5, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 5, 3, 3]
            self.dilates = [1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3]
            self.dilates = [1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]

        # 去重kernel_size，并记录dilation对应的索引
        self.unique_kernel_sizes = sorted(set(self.kernel_sizes))
        self.kernel_to_dilates = {}
        self.kernel_to_scale = {}
        for k_size in self.unique_kernel_sizes:
            self.kernel_to_dilates[k_size] = [self.dilates[i] for i in range(len(self.kernel_sizes)) if self.kernel_sizes[i] == k_size]
            if (kernel_size // k_size)// 2 * 2 != 0:
                self.kernel_to_scale[k_size] = (kernel_size // k_size)// 2 * 2
            else:
                self.kernel_to_scale[k_size] = 1

    
        self.mlps = nn.ModuleDict({
            str(k_size): nn.ModuleDict({
                'mlp': nn.Sequential(
                    nn.Conv2d((self.kernel_to_scale[k_size] ** 2*self.prompt_len),(self.kernel_to_scale[k_size] ** 2*self.prompt_len),kernel_size=3, padding=1,groups=(self.kernel_to_scale[k_size] ** 2*self.prompt_len)),
                    nn.Conv2d(self.prompt_len * (self.kernel_to_scale[k_size] ** 2), self.prompt_len*4* (self.kernel_to_scale[k_size] ** 2) if self.kernel_to_scale[k_size] <= 2 else 64, kernel_size=1),  # 输入通道数要考虑到pixel_unshuffle后的变化
                    nn.ReLU(),
                    nn.Conv2d(self.prompt_len*4* (self.kernel_to_scale[k_size] ** 2) if self.kernel_to_scale[k_size] <= 2 else 64, self.prompt_len*self.kernel_to_scale[k_size], kernel_size=1)
                ),
                'dilation_operations': nn.ModuleList([
                    nn.Conv2d(self.prompt_len*self.kernel_to_scale[k_size], self.prompt_len, kernel_size=1, bias=False)
                    for _ in self.kernel_to_dilates[k_size]  # 根据dilates为当前kernel_size设置对应的卷积操作
                ])
            })
            for k_size in self.unique_kernel_sizes
        })

    def convert_dilated_to_nondilated(self, kernel, dilate_rate):
        identity_kernel = torch.ones((1, 1, 1, 1)).to(kernel.device)
        if kernel.size(1) == 1:
            # 这是一个深度卷积核
            dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
            return dilated
        else:
            # 这是一个稠密卷积核或分组卷积核
            slices = []
            for i in range(kernel.size(1)):
                dilated = F.conv_transpose2d(kernel[:, i:i + 1, :, :], identity_kernel, stride=dilate_rate)
                slices.append(dilated)
            return torch.cat(slices, dim=1)

    def merge_dilated_into_large_kernel(self, large_kernel, dilated_kernel, dilated_r):
        large_k = large_kernel.size(2)
        dilated_k = dilated_kernel.size(2)
        equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
        equivalent_kernel = self.convert_dilated_to_nondilated(dilated_kernel, dilated_r)
        rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
        merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
        return merged_kernel



    def pad_input(self, x, scale_factor):
        """根据scale_factor计算所需的填充"""
        _, _, h, w = x.size()
        pad_h = (scale_factor - h % scale_factor) % scale_factor  # 需要填充的高度
        pad_w = (scale_factor - w % scale_factor) % scale_factor  # 需要填充的宽度

        # 如果需要填充，则使用F.pad进行填充，分别在高和宽两侧进行对称填充
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)
        
        return x
    
    def merge_kernels(self, x0):
        # 合并所有卷积核为一个大的卷积核
        # weight0 = torch.einsum('oikj,oijl->oikl', self.A, self.B)
        weight0 = torch.cat([torch.einsum('oikj,oijl->oikl', getattr(self, f'A{i+1}'), getattr(self, f'B{i+1}')) for i in range(self.prompt_len)],dim=1)
        # merged_weight = weight0.clone()
        merged_weight = torch.zeros_like(weight0)
        out_channel,groups,kh,kw = weight0.size()
        for k_size in self.unique_kernel_sizes:
            scale_factor = self.kernel_to_scale[k_size]
            if scale_factor > 1:
                weight = self.pad_input(weight0, scale_factor)
                weight = F.pixel_unshuffle(weight, downscale_factor=scale_factor)
            else:
                weight = weight0
            
            mlp_dict = self.mlps[str(k_size)]
            mlp = mlp_dict['mlp']
            kernel_weight = mlp(weight)
            kernel_weight = F.interpolate(kernel_weight, (k_size, k_size), mode="bilinear")

            # 进行卷积核合并
            for idx, dilation in enumerate(self.kernel_to_dilates[k_size]):
                dilation_weight = mlp_dict['dilation_operations'][idx](kernel_weight)
                merged_weight = self.merge_dilated_into_large_kernel(merged_weight, dilation_weight, dilation)
        merged_weight = merged_weight.reshape(out_channel*groups,1,kh,kw)
        merged_weight = merged_weight.reshape(out_channel*groups,1,kh,kw)
        return merged_weight
    

    def forward(self, x):
        # 计算低秩分解卷积核的权重
        # weight0 = torch.concatenate([torch.einsum('oikj,oijl->oikl', self.A1, self.B1),torch.einsum('oikj,oijl->oikl', self.A2, self.B2),torch.einsum('oikj,oijl->oikl', self.A3, self.B3)],dim=1)
        weight0 = torch.cat([torch.einsum('oikj,oijl->oikl', getattr(self, f'A{i+1}'), getattr(self, f'B{i+1}')) for i in range(self.prompt_len)],dim=1)


        # 初始化输入 x0
        x0 = x
        x = x.repeat(1, self.prompt_len, 1, 1)
        #判断是否训练
        if self.training:
        
            for k_size in self.unique_kernel_sizes:
                # 使用 MLP 生成初始卷积核
                scale_factor = self.kernel_to_scale[k_size]
                if scale_factor > 1:
                    weight = self.pad_input(weight0, scale_factor)  # 使用pad确保输入维度是scale_factor的整数倍
                    weight = F.pixel_unshuffle(weight, downscale_factor=scale_factor)  # 在 MLP 前使用pixel_unshuffle
                else:
                    weight = weight0
                
                mlp_dict = self.mlps[str(k_size)]
                mlp = mlp_dict['mlp']
                kernel_weight = mlp(weight)

                # 插值到对应的 kernel_size
                kernel_weight = F.interpolate(kernel_weight, (k_size, k_size), mode="bilinear")

                # 第二层循环：遍历相同 kernel_size 但不同 dilation 的卷积操作
                for idx, dilation in enumerate(self.kernel_to_dilates[k_size]):
                    # 查找对应 dilation 的卷积操作
                    dilation_weight = mlp_dict['dilation_operations'][idx](kernel_weight).reshape(-1,1,k_size,k_size).contiguous()   

                    # 应用卷积操作
                    x = x + F.conv2d(
                        x0, dilation_weight, bias=self.bias, stride=self.stride,
                        padding=dilation * (k_size - 1) // 2,
                        dilation=dilation, groups=self.groups
                    )
        else:
            # print('merge')
            merged_kernel = self.merge_kernels(x0)  # 合并权重
            x = F.conv2d(x0, merged_kernel, bias=self.bias, stride=self.stride, padding=(merged_kernel.shape[-1]-1)//2, groups=self.groups)


        return x


class PromptGenBlock(nn.Module):
    def __init__(self, prompt_dim=128, prompt_len=5, prompt_size=96,rank=2,rank_scale =[1,1.2,0.8]):
        super(PromptGenBlock, self).__init__()
        # self.promt_A = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size, rank))
        # self.promt_B = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, rank, prompt_size))
        # self.prompt_param = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size))


        # self.conv = nn.Conv2d(12, prompt_dim, kernel_size=1, stride=1,bias=False)
        self.shuffler = nn.PixelUnshuffle(2)
        # Create a list of convolutional layers

        self.convolutions_cls = LowRankConv2d(prompt_dim, prompt_dim,prompt_size=prompt_size,rank=rank,rank_scale=rank_scale,prompt_len=prompt_len,  stride=1, groups=prompt_dim,bias=False) 

        self.convolutions_deg = nn.Conv2d(prompt_dim, prompt_dim*2, kernel_size=1, stride=1,bias=False)

    def forward(self, x, prompt_weights):
        B, C, H, W = x.shape
        cls_prompt_weight = F.softmax(prompt_weights[:,:-1],dim=1)
        deg_prompt_weight = torch.stack((prompt_weights[:, :3].sum(dim=1),prompt_weights[:, 3]),dim=1)
        

        # Apply each convolution to the prompt
        combined_prompt = 0
        convolved_prompts = self.convolutions_deg(x).reshape(B, -1, C, H, W).contiguous()
        convolved_prompts = torch.einsum('bc,bcxyz->bxyz', deg_prompt_weight, convolved_prompts)
        convolved_prompts = self.convolutions_cls(convolved_prompts).reshape(B, -1, C, H, W).contiguous()
        convolved_prompts = torch.einsum('bc,bcxyz->bxyz', cls_prompt_weight, convolved_prompts)

        
        

        return convolved_prompts
    
class PromptGenBlock_fre(nn.Module):
    def __init__(self, prompt_dim=128,ada_dim=12, prompt_size=96,rank=2):
        super(PromptGenBlock_fre, self).__init__()
        # self.promt_A = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size, rank))
        # self.promt_B = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, rank, prompt_size))
        # self.prompt_param = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size))

        self.conv_fre = LowRankConv2d(prompt_dim, prompt_dim,prompt_size=prompt_size,rank=rank,  stride=1, groups=prompt_dim,bias=False)

        self.pconv = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=1, stride=1,bias=False)
        
        self.shuffler = nn.PixelUnshuffle(2)
        self.conv = nn.Conv2d(ada_dim, prompt_dim, kernel_size=1, stride=1,bias=False)

        # Create a list of convolutional layers


    def forward(self, x,ada):
        B, C, H, W = x.shape
        ada = self.shuffler(ada)
        ada = self.conv(ada)
        # cls_prompt_weight = F.softmax(prompt_weights[:,:-1],dim=1)
        # deg_prompt_weight = torch.stack((prompt_weights[:, :3].sum(dim=1),prompt_weights[:, 3]),dim=1)
        

        # # Apply each convolution to the prompt
        # combined_prompt = 0
        # convolved_prompts = [self.convolutions_cls[i](x) for i in range(len(self.convolutions_cls))]
        # for k in range(len(convolved_prompts)):
        #     conv_deg = [self.convolutions_deg[i](convolved_prompts[k])*deg_prompt_weight[:,i].unsqueeze(1).unsqueeze(2).unsqueeze(3) for i in range(len(self.convolutions_deg))]
        #     conv_deg = sum(conv_deg)*cls_prompt_weight[:,k].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        #     combined_prompt += conv_deg
        combined_prompt = self.conv_fre(x,ada)
        combined_prompt = self.pconv(combined_prompt)
        

        return combined_prompt


class Classifier(nn.Module):
    def __init__(self,class_num):
        super(Classifier, self).__init__()
        self.lastOut = nn.Linear(32, class_num)

        # Condtion network
        self.CondNet = nn.Sequential(nn.Conv2d(3, 128, 3, 3), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(128, 128, 3,3), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(128, 128, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(128, 32, 1))

    def forward(self, x):
        out = self.CondNet(x)
        out = nn.AdaptiveAvgPool2d(1)(out)

        # if out.size()[2] > out.size()[3]:
        #     out = nn.AvgPool2d(out.size()[3])(out)
        # else:
        #     out = nn.AvgPool2d(out.size()[2])(out)
        out = out.view(out.size(0), -1)
        out = self.lastOut(out)
        out = F.softmax(out, dim=1)
        return out
    
    
# @ARCH_REGISTRY.register()
class dwt_aio28(nn.Module):
    def __init__(self, dim, n_blocks=8, ffn_scale=2.0,dwt_levels=3,only_deg=1, upscaling_factor=4,vae_weight=None,cls_weight=None,deg_num= 7,prompt_scale=2,rank=[4,2,1],rank_scale= None,config=None,sample= True,dwt_dim = 16,num_heads=3,param_key = 'params',out_dim= 64):
        super().__init__()
        with open(config) as f:
            # config = yaml.load(f, Loader=yaml.FullLoader)
            config = yaml.load(f, Loader=yaml.FullLoader)["network_g"]
            config.pop('type')
            self.vae = AutoencoderKL(**config,dwt_dim=dwt_dim,num_heads=num_heads,prompt_scale=prompt_scale,rank=rank,rank_scale=rank_scale,prompt_len=deg_num)
            self.sample = sample
            self.only_deg = only_deg
            self.dwt_levels = dwt_levels
            self.deg_num = deg_num
            if vae_weight:
                print(torch.load(vae_weight,map_location='cpu').keys())
                msg = self.vae.load_state_dict(torch.load(vae_weight,map_location='cpu')[param_key],strict=False)
                print(f"load vae weight from{param_key}")
                print(f"load vae weight from {vae_weight}")
                print('missing keys:',len(msg.missing_keys),'unexpected keys:',len(msg.unexpected_keys))
            self.wt_filter, self.iwt_filter = wavelet.create_wavelet_filter('db1', dim, dim, torch.float)
            self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
            self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
            self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1,dim*4,1,1], init_scale=0.1) for _ in range(self.dwt_levels)]
            )
            self.wt_function = wavelet.wavelet_transform_init(self.wt_filter)
            self.iwt_function = wavelet.inverse_wavelet_transform_init(self.iwt_filter)
            self.cls = Classifier(self.deg_num+1)
            if cls_weight:
                print(torch.load(cls_weight,map_location='cpu').keys())
                msg2 = self.cls.load_state_dict(torch.load(cls_weight,map_location='cpu')['params_ema'],strict=True)
                print(f"load cls weight from {cls_weight}")
                print('missing keys:',len(msg2.missing_keys),'unexpected keys:',len(msg2.unexpected_keys))



            # self.prompt_scale = prompt_scale
        

            a =0
            m = []
            for name, param in self.vae.named_parameters():
                if name in msg.missing_keys and 'wt_filter' not in name and 'iwt_filter' not in name:
                    param.requires_grad = True
                    a +=1
                    m.append(name)
                    

                else:
                    param.requires_grad = False
            print(f"adapter num is {a}")
            # assert len(msg.missing_keys) == a
            for i in msg.missing_keys:
                if i not in m:
                    print(i)


            #冻结cls的参数
            for name, param in self.cls.named_parameters():
                param.requires_grad = False


        # self.to_feat = nn.Sequential(
        #     nn.Conv2d(3, dim // upscaling_factor, 3, 1, 1),
        #     nn.PixelUnshuffle(upscaling_factor),
        #     nn.Conv2d(dim*upscaling_factor, dim, 1, 1, 0),
        # )
        
        # self.feats = nn.ModuleList([AttBlock(dim, ffn_scale) for _ in range(self.dwt_levels+1)])
        self.feats = nn.ModuleList(nn.Sequential(*[AttBlock(dim, ffn_scale) for _ in range(n_blocks[i])]) for i in range(self.dwt_levels+1))
        # self.feats = nn.Sequential(*[AttBlock(dim, ffn_scale) for _ in range(n_blocks)])

        # self.to_img = nn.Sequential(
        #     nn.Conv2d(out_dim, dim, 3, 1, 1),
        #     nn.PixelShuffle(upscaling_factor)
        # )
        # self.merge =  nn.Sequential(  # 这里考虑加一个MOE
        #     nn.Conv2d(out_dim, out_dim-dim, 3, 1, 1),
        # )
        self.rec_block = nn.Sequential(WTConv2d(3, 3, kernel_size=5, stride=1, bias=True, wt_levels=3, wt_type='db1'),
                                       nn.Conv2d(3, 16, 1, 1, 0),
                                       nn.Conv2d(16, 3, 1, 1, 0),
                                       WTConv2d(3, 3, kernel_size=5, stride=1, bias=True, wt_levels=3, wt_type='db1')
                                       )
        

        

    def forward(self, input,gt=None,stage=None):
        prompt_weight = self.cls(input)
        
        x,add,high_list = self.vae.encode(input,prompt_weight)
        # x = self.vae.encode(input,use_adapter=False)
        posterior = DiagonalGaussianDistribution(x)
        x = posterior.sample()

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []
        curr_x_ll = x

        if gt is not None:
            
            gt = self.vae.encode(gt,use_adapter=False)
            gt = DiagonalGaussianDistribution(gt)
            gt = gt.sample()
            curr_gt_ll = gt
            gt_h_in_levels = []
            for i in range(self.dwt_levels):
                curr_gt = self.wt_function(curr_gt_ll)
                curr_gt_ll = curr_gt[:,:,0,:,:]
                shape_gt = curr_gt.shape
                curr_gt_tag = curr_gt.reshape(shape_gt[0], shape_gt[1]* 4 , shape_gt[3], shape_gt[4])
                curr_gt_tag = self.wavelet_scale[i](curr_gt_tag)

                
                
                gt_h_in_levels.append(curr_gt[:,:,1:4,:,:])

            del curr_gt,curr_gt_ll,curr_gt_tag,shape_gt
        
        for i in range(self.dwt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:,:,0,:,:]
            
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1]* 4 , shape_x[3], shape_x[4])
            

            curr_x_tag = self.wavelet_scale[i](curr_x_tag)
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:,:,0,:,:])

            x_h_in_levels.append(curr_x_tag[:,:,1:4,:,:])
        
        next_x_ll = 0
        if self.dwt_levels > 0:
            del curr_x_tag,curr_x
        # self.feats(curr_x_tag)+curr_x_tag
        for i in range(self.dwt_levels-1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_ll = self.feats[self.dwt_levels-1-i](curr_x_ll)+curr_x_ll
            # curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll
            if stage is not None:
                if i <= self.dwt_levels - stage-1 and i > 0 :
                    for j in range(i,-1,-1):
                        curr_x_h  = gt_h_in_levels[j]
                        curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
                        curr_x_ll = self.iwt_function(curr_x)
                    next_x_ll = curr_x_ll 
                    break

                else:
                    curr_x_h = x_h_in_levels.pop() 
            else:
                curr_x_h = x_h_in_levels.pop()

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]
        if self.dwt_levels > 0:
            del x_ll_in_levels,x_h_in_levels,curr_x,shape_x,shapes_in_levels,curr_shape,curr_x_h,curr_x_ll
        x = self.feats[-1](x)
        
        
        
        x = next_x_ll+ x
        
        input = input + self.rec_block(input)
        
        x = self.vae.decode(x,add,high_list,prompt_weight) + input

        

        return x
        # return x, posterior


        

    @torch.no_grad()
    def test_tile(self, input, tile_size=512, tile_pad=16):
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
                output_tile = self(input_tile)

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
    



class AutoencoderKL(nn.Module):
    def __init__(self,
                 ddconfig,    
                 embed_dim,
                 optim,
                 dwt_dim,
                 prompt_len,
                 prompt_scale,
                 rank,
                 rank_scale,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 num_heads=3,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig,dwt_dim=dwt_dim,prompt_scale=prompt_scale,rank=rank,rank_scale=rank_scale,prompt_len=prompt_len)
        self.decoder = Decoder(**ddconfig,dwt_dim=dwt_dim,prompt_scale=prompt_scale,rank=rank,prompt_len=prompt_len)
        # self.loss = instantiate_from_config(lossconfig)
        self.adapter_mid = nn.Sequential(make_attn(dwt_dim, attn_type="restormer",num_heads=num_heads),
                                         nn.Conv2d(dwt_dim, dwt_dim, 1),
                                     make_attn(dwt_dim, attn_type="restormer",num_heads=num_heads))
        self.learning_rate = optim["lr"]
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x,prompt_weight=None,use_adapter = True):

        h,ll ,high_list = self.encoder(x,prompt_weight,use_adapter)
        
        moments = self.quant_conv(h)
        if use_adapter:
            ll = self.adapter_mid(ll)
            return moments,ll,high_list
    
        else:               
            
            return moments  
        
        

    def decode(self, z,ll,high_list,prompt_weight):
        z = self.post_quant_conv(z)
        dec = self.decoder(z,ll,high_list,prompt_weight)
        return dec

    def forward(self, input):
        #采样过程合并到encode中
        z,posterior = self.encode(input)
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        # x = batch[k]
        x = batch
        # print(x.shape)
        if len(x.shape) == 3:
            x = x[..., None]
        # x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x
        
        

        
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,prompt_scale,rank,prompt_len,
                 attn_resolutions, dropout=0.0, freup_type='pad', in_channels,
                 resolution, z_channels, dwt_dim = 16,give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="restormer",res_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = make_res(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,res_type=res_type)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = make_res(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,res_type=res_type)

        # upsampling
        self.up = nn.ModuleList()
        self.adapters = nn.ModuleList()

        # ch_mult = (1,) + tuple(ch_mult)
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(make_res(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout,res_type=res_type))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = fresadd(block_in, freup_type)
                curr_res = curr_res * 2
                self.adapters.insert(0, dwt_revadapter(dwt_dim,dwt_dim,block_out,block_out,prompt_len,curr_res//(prompt_scale*2)+1,rank[i_level-1]))
            self.up.insert(0, up) # prepend to get consistent order
            

        # self.adapters.append(dwt_adapter(dwt_dim,dwt_dim,block_out,block_out,curr_res//prompt_scale+1,rank[i_level]))

        # end
        self.norm_out = Normalize(block_in)
        if dwt_dim > 4:
            
            self.norm_out2 = Normalize(dwt_dim)
        else:
            self.norm_out2 = Normalize(dwt_dim,num_groups=dwt_dim)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        
        self.conv_out2 = torch.nn.Conv2d(dwt_dim,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

        # self.shuffler = nn.PixelShuffle(2)

    def forward(self, z,ll,high_list,prompt_weight):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)
        add_in = ll
        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            # h1 = self.shuffler(h)
            # h1 = torch.cat([h1, h1], dim=1)
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
                add,add_in = self.adapters[i_level-1](add_in,high_list[i_level-1],h,prompt_weight)
                h = h + add

                

        # end
        if self.give_pre_end:
            return h
        det_out = self.conv_out2(nonlinearity(self.norm_out2(add_in)))
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
            det_out = torch.tanh(det_out)
        return h+det_out

class ZeroConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ZeroConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)

class Adapter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Adapter, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels//4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels//4, out_channels)
        self.out_channels = out_channels
        self.init_weights()

    def init_weights(self):
        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        x_flat = x.view(batch_size, channels, -1).permute(0, 2, 1)  # Reshape to (batch_size, height*width, channels)
        x_flat = self.fc1(x_flat)
        x_flat = self.relu(x_flat)
        x_flat = self.fc2(x_flat)
        x_flat = x_flat.permute(0, 2, 1).view(batch_size, self.out_channels, height, width)  # Reshape back
        return  x_flat
    
class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None
    
    def forward(self, x):
        return torch.mul(self.weight, x)

class FFT_filter(nn.Module):
    def __init__(self, C1, C2):
        super(FFT_filter, self).__init__()
        
        self.C1 = C1
        self.C2 = C2
        
        # 滤波器生成模块
        self.filter_generators = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(C1, C2, kernel_size=1),
                nn.Sigmoid()  # 使用Sigmoid激活函数进行归一化
            ) for _ in range(4)
        ])
        
        # 通道权重生成模块
        self.channel_weight_generator = nn.Sequential(
            nn.Conv2d(C2, 4*C1, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )
        self.fuse = nn.Conv2d(4*C2, C2, 1)
        # 输出模块
        # self.output1_conv = nn.Conv2d(4 * C1 + C2, C2, kernel_size=1)
        # self.output2_conv = nn.Conv2d(4 * C1 + C2, 1 * C1, kernel_size=1)
        # self.output3_conv = nn.Conv2d(4 * C1 + C2, 3 * C1, kernel_size=1)

    def forward(self, x1, x2):
        # x1: (B, 4*C1, H, W)
        # x2: (B, C2, H, W)
        B, _, H, W = x1.shape
        
        # 1. 特征划分
        x1_splits = torch.split(x1, self.C1, dim=1)  # 划分为4个(C1, H, W)的块
        
        # 2. 生成滤波器并应用于RFFT频谱
        x2_rfft = fft.rfft2(x2)
        x2_rfft_shifted = fft.fftshift(x2_rfft)  # 将频谱中心移到图像中心
        
        outputs2 = []
        for i in range(4):
            filters = self.filter_generators[i](x1_splits[i])
            
            half_width = W // 2 +1
            filters_first_half = filters[..., :half_width]
            filters_second_half = torch.flip(filters[..., half_width-2:],dims=[-1])

            # 平均前后两部分滤波器
            filters_avg = (filters_first_half + filters_second_half) / 2
            
            filtered_rfft = x2_rfft_shifted * filters_avg
            output_irfft = fft.irfft2(fft.ifftshift(filtered_rfft), s=(H, W))
            output_irfft = output_irfft + x2
            outputs2.append(output_irfft)

        # 将输出列表的张量在通道维度上拼接
        output2 = torch.cat(outputs2, dim=1)
        output2 = self.fuse(output2)
        
        # 3. 通道权重生成和调制
        channel_weight = self.channel_weight_generator(x2)
        output1 = x1 * channel_weight+x1

        # 4. 特征融合
        fused_feature = torch.cat([output1, output2], dim=1)

        return fused_feature

class WTblock(nn.Module):
    def __init__(self, in_channels, out_channels,enc_channel, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTblock, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        # self.wt_filter = 
        self.wt_filter = nn.Parameter(wavelet.create_wavelet_decfilter(wt_type, in_channels, in_channels, torch.float), requires_grad=False)
        # if self.wt_filter.requires_grad:
        #     a = 0
        self.wt_function = wavelet.wavelet_transform_init(self.wt_filter)
        
        # self.merge = nn.Conv2d(in_channels*4+enc_channel, in_channels*4, kernel_size=3, padding=1)
        # self.merge = GatedFeatureEnhancement(in_channels*4,enc_channel)
        self.naf = NAFBlock(in_channels*4)
        self.wavelet_convs = nn.Conv2d(in_channels*4, in_channels*4, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels*4, bias=False) 
        self.wavelet_scale = _ScaleModule([1,in_channels*4,1,1], init_scale=0.1) 
        self.merge = FFT_filter(in_channels,enc_channel)
        
        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride, groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x,enc): 
        curr_x_ll = x
        curr_shape = curr_x_ll.shape
        if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)
    
        curr_x = self.wt_function(curr_x_ll)
            
        shape_x = curr_x.shape
        curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
        # curr_x_tag = torch.cat((curr_x_tag,enc),dim=1)
        # curr_x_tag = self.merge(curr_x_tag,enc)
        curr_x_tag = self.naf(curr_x_tag)
        curr_x_tag = self.wavelet_convs(curr_x_tag)

        curr_x_tag = self.wavelet_scale(curr_x_tag)
        curr_x_tag = self.merge(curr_x_tag,enc)

        

        return curr_x_tag
    
class WT_revblock(nn.Module):
    def __init__(self, in_channels, out_channels,enc_channel, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WT_revblock, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.iwt_filter = wavelet.create_wavelet_recfilter(wt_type, in_channels, in_channels, torch.float)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        
        self.iwt_function = wavelet.inverse_wavelet_transform_init(self.iwt_filter)
        self.merge_conv  = nn.Conv2d(in_channels+enc_channel, in_channels, kernel_size=3, padding=1)
        # self.wavelet_scale = _ScaleModule([1,in_channels*4,1,1], init_scale=0.1) 
        self.naf = NAFBlock(in_channels)
        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride, groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, ll,high ,enc,ll_before= None): 
        if ll_before:
            ll = ll+ll_before
        curr_x = torch.cat([ll.unsqueeze(2), high], dim=2)
        next_x_ll = self.iwt_function(curr_x)
        next_x_ll = torch.cat([next_x_ll,enc],dim=1)
        next_x_ll = self.merge_conv(next_x_ll)
        next_x_ll = self.naf(next_x_ll)
        

        return next_x_ll
    

class dwt_adapter (nn.Module):
    def __init__(self, in_channels, out_channels,enc_channel,prompt_dim,prompt_len,prompt_size,rank,rank_scale,stride=1):
        super(dwt_adapter, self).__init__()

        self.wtblock = WTblock(in_channels, out_channels,enc_channel, kernel_size=5, stride=1, bias=True, wt_levels=2, wt_type='db1')
        self.zeroconv = ZeroConv2d(enc_channel, enc_channel, 1)
        self.enc_conv = nn.Conv2d(in_channels*4+enc_channel, enc_channel, 3,padding=1, stride=stride)
        self.ll_conv = nn.Conv2d(in_channels*4+enc_channel,in_channels,1)
        self.high_conv = nn.Conv2d(in_channels*4+enc_channel,in_channels*3,1)
        self.prompt = PromptGenBlock(prompt_dim,prompt_len,prompt_size,rank,rank_scale)

    def forward(self, x,enc,prompt_weight):

        prompt = self.prompt(enc,prompt_weight)
        enc = enc+prompt
        x = self.wtblock(x,enc)
        b,c,h,w = x.shape
        n = 4
        c = int((c -enc.shape[1])/n)
        
        # x = x.view(b,-1,h,w).contiguous()
        ll = self.ll_conv(x)
        high = self.high_conv(x).view(b,c,n-1,h,w).contiguous()
        
        x = self.enc_conv(x)
        
        x = self.zeroconv(x)
        return x,ll,high

class dwt_revadapter (nn.Module):
    def __init__(self, in_channels, out_channels,enc_channel,prompt_dim,prompt_len,prompt_size,rank, kernel_size=5, stride=1):
        super(dwt_revadapter, self).__init__()

        self.wtblock = WT_revblock(in_channels, out_channels,enc_channel, kernel_size=5, stride=1, bias=True, wt_levels=2, wt_type='db1')
        self.zeroconv = ZeroConv2d(in_channels, enc_channel, 3,padding=1, stride=stride)
        # self.high_prompt = PromptGenBlock_fre(prompt_dim,36,prompt_size,rank)
        # self.ll_prompt = PromptGenBlock_fre(prompt_dim,12,prompt_size,rank)
        # self_Ph = nn.conv2d()

        self.low_conv_amp = nn.Conv2d(3, 1, 1)
        self.low_conv_pha = nn.Conv2d(3, 1, 1)
        self.high_conv_amp = nn.Conv2d(3, 1, 1)
        self.high_conv_pha = nn.Conv2d(3, 1, 1)
        self.mid_conv_amp = nn.Conv2d(6, 2, 1)
        self.mid_conv_pha = nn.Conv2d(6, 2, 1)
        self.shuffler = nn.PixelShuffle(2)
        self.fre_prompt = nn.Conv2d(4,4*(prompt_len+1),1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, ll,high,enc,prompt_weight):
        # x1= torch.fft.rfft2(x1, norm='backward')
        # amp, phase = torch.abs(x1), torch.angle(x1)
        # amp = self.ampconv(amp)
        # phase = self.phaconv(phase)
        # x1 = torch.complex(amp*torch.cos(phase),amp*torch.sin(phase))
        # x1 = torch.fft.irfft2(x1, s=(h.shape[2],h.shape[3]),norm='backward')
        ll1 = torch.fft.rfft2(ll, norm='backward')
        high1 = torch.fft.rfft2(high[:,:,-1], norm='backward')
        mid1 = torch.fft.rfft2(high[:,:,:-1].reshape(high.shape[0],-1,high.shape[3],high.shape[4]).contiguous(), norm='backward')
        ll_amp, ll_pha = torch.abs(ll1), torch.angle(ll1)
        high_amp, high_pha = torch.abs(high1), torch.angle(high1)
        mid_amp, mid_pha = torch.abs(mid1), torch.angle(mid1)
        ll_amp = self.low_conv_amp(ll_amp)
        ll_pha = self.low_conv_pha(ll_pha)
        high_amp = self.high_conv_amp(high_amp)
        high_pha = self.high_conv_pha(high_pha)
        mid_amp = self.mid_conv_amp(mid_amp)
        mid_pha = self.mid_conv_pha(mid_pha)
        ll1 = torch.complex(ll_amp*torch.cos(ll_pha),ll_amp*torch.sin(ll_pha))
        high1 = torch.complex(high_amp*torch.cos(high_pha),high_amp*torch.sin(high_pha))
        mid1 = torch.complex(mid_amp*torch.cos(mid_pha),mid_amp*torch.sin(mid_pha))
        prompt = torch.cat([ll1,high1,mid1],dim=1)
        B,C,H,W = prompt.shape
        amp = self.fre_prompt(torch.abs(prompt)).reshape(B,-1,4,H,W).contiguous()
        pha = self.fre_prompt(torch.angle(prompt)).reshape(B,-1,4,H,W).contiguous()
        amp = torch.einsum('bijkm,bi->bjkm', amp, prompt_weight)
        pha = torch.einsum('bijkm,bi->bjkm', pha, prompt_weight)
        prompt = torch.complex(amp*torch.cos(pha),amp*torch.sin(pha))
        # prompt = torch.einsum('bijkm,bi->bjkm', prompt.reshape(B,-1,4,H,W), prompt_weight)
        prompt = torch.fft.irfft2(prompt, s=(ll.shape[2],ll.shape[3]),norm='backward')
        
        # low_pro = torch.fft.irfft2(ll1, s=(ll.shape[2],ll.shape[3]),norm='backward')
        # high_pro = torch.fft.irfft2(high1, s=(ll.shape[2],ll.shape[3]),norm='backward')
        # mid_pro = torch.fft.irfft2(mid1, s=(ll.shape[2],ll.shape[3]),norm='backward')


       
        # prompt = 
        prompt = self.shuffler(prompt)
        half_width = prompt.shape[-1] // 2 +1
        filters_first_half = prompt[..., :half_width]
        filters_second_half = torch.flip(prompt[..., half_width-2:],dims=[-1])
        filters_avg = (filters_first_half + filters_second_half) / 2
        filters_avg = self.sigmoid(filters_avg)
        enc_fft = torch.fft.rfft2(enc, norm='backward')
        enc_fft = torch.fft.fftshift(enc_fft)
        enc_fft = enc_fft * filters_avg +enc_fft
        enc_fft = torch.fft.irfft2(enc_fft, s=(enc.shape[2],enc.shape[3]),norm='backward')

        enc = enc+enc_fft
        ll = self.wtblock(ll,high,enc)
        add = self.zeroconv(ll)
        return add,ll

                
class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,prompt_scale,rank,rank_scale,prompt_len,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels,dwt_dim = 16 ,double_z=True, use_linear_attn=False, attn_type="vanilla",res_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        self.conv_in2 = torch.nn.Conv2d(in_channels,
                                       dwt_dim,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult

        self.adapters  = nn.ModuleList()

        self.down = nn.ModuleList()
        # self.shuffler = nn.PixelUnshuffle(2)

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(make_res(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout,res_type=res_type))
                
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            
            down = nn.Module()
            down.block = block
            down.attn = attn
            
            # down.adapters = adapters  # Add adapters to down
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2

                self.adapters.append(dwt_adapter(dwt_dim,dwt_dim,block_out,block_out,prompt_len,curr_res//prompt_scale+1,rank[i_level],rank_scale,))
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = make_res(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,res_type=res_type)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = make_res(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,res_type=res_type)
        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2 * z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

        

    def forward(self, x,prompt_weight,use_adapter=True):
        # timestep embedding
        temb = None
       

        # downsampling
        hs = [self.conv_in(x)]
        if use_adapter:
            adapter_in =[self.conv_in2(x)]
            high_list = []
        else:
            ll = None
            high_list = None

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                
                # h = h + self.down[i_level].adapters[i_block](torch.cat([adapter_in[i_level],h], dim=1))  # Apply Adapter
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
                
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(hs[-1])
                if use_adapter:
                    
                    add,ll,high  = self.adapters[i_level](adapter_in[i_level],h,prompt_weight)
                    h = h + add
                    adapter_in.append(ll)
                    high_list.append(high)
                hs.append(h)



        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        
        return h,ll,high_list

# for LOL dataset
# class SAFMN(nn.Module):
#     def __init__(self, dim, n_blocks=8, ffn_scale=2.0, upscaling_factor=4):
#         super().__init__()
#         self.to_feat = nn.Sequential(
#             nn.Conv2d(3, dim, 3, 1, 1),
#             ResBlock(dim, 3, 1, 1)
#         )

#         self.feats = nn.Sequential(*[AttBlock(dim, ffn_scale) for _ in range(n_blocks)])

#         self.to_img = nn.Sequential(
#             ResBlock(dim, 3, 1, 1),
#             nn.Conv2d(dim, 3, 3, 1, 1)
#         )

#     def forward(self, x):
#         x = self.to_feat(x)
#         x = self.feats(x) + x
#         x = self.to_img(x)
#         return x

if __name__== '__main__': 
    # x = torch.randn(2, 3, 512, 512).to('cuda')
    import yaml
    config = yaml.load(open('/fs-computility/ai4sData/liuyidi/code/LatentGen/options/all-in-one/6dre/dwt_aio28_6d.yml', 'r'), Loader=yaml.FullLoader)['network_g']
    config.pop('type')
    model = dwt_aio28(**config).to('cuda')

    HF = torch.randn(1, 3, 3280, 2160).cuda()
    from thop.profile import profile
    #计算模型可训练参数量
    param = sum(p.numel() for p in model.parameters())
    print('Model Param: ',param)
    name = "our"
    total_ops, total_params = profile(model, (HF,))
    print(
        "%s         | %.4f(M)      | %.4f(G)         |"
        % (name, total_params / (1000 ** 2), total_ops / (1000 ** 3))
    )