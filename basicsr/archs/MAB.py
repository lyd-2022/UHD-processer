#coding:utf-8
import torch
import torch.nn as nn


import torch
import torch.nn as nn
import torch.nn.functional as F
# Dummy definitions for the missing utilities
class Conv2d1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        return self.conv(x)

class LayerNorm4D(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.layer_norm = nn.GroupNorm(1, num_features)

    def forward(self, x):
        return self.layer_norm(x)

class GSAU(nn.Module):
    r"""Gated Spatial Attention Unit.

    Args:
        n_feats: Number of input channels

    """

    def __init__(self, n_feats: int) -> None:
        super().__init__()
        i_feats = n_feats * 2

        self.Conv1 = Conv2d1x1(n_feats, i_feats)
        self.DWConv1 = nn.Conv2d(
            n_feats, n_feats, 7, 1, 7 // 2, groups=n_feats)
        self.Conv2 = Conv2d1x1(n_feats, n_feats)

        self.norm = LayerNorm4D(n_feats)
        self.scale = nn.Parameter(torch.zeros(
            (1, n_feats, 1, 1)), requires_grad=True)

    def forward(self, x) -> torch.Tensor:
        shortcut = x.clone()

        x = self.Conv1(self.norm(x))
        a, x = torch.chunk(x, 2, dim=1)
        x = x * self.DWConv1(a)
        x = self.Conv2(x)

        return x * self.scale + shortcut

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
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

    
class GroupGLKA(nn.Module):
    def __init__(self, n_feats, k=2, squeeze_factor=15):
        super().__init__()
        i_feats = 2*n_feats
        
        self.n_feats= n_feats
        self.i_feats = i_feats
        
        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)
        
        #Multiscale Large Kernel Attention
        self.LKA7 = nn.Sequential(
            nn.Conv2d(n_feats//3, n_feats//3, 7, 1, 7//2, groups= n_feats//3),  
            nn.Conv2d(n_feats//3, n_feats//3, 9, stride=1, padding=(9//2)*4, groups=n_feats//3, dilation=4),
            nn.Conv2d(n_feats//3, n_feats//3, 1, 1, 0))
        self.LKA5 = nn.Sequential(
            nn.Conv2d(n_feats//3, n_feats//3, 5, 1, 5//2, groups= n_feats//3),  
            nn.Conv2d(n_feats//3, n_feats//3, 7, stride=1, padding=(7//2)*3, groups=n_feats//3, dilation=3),
            nn.Conv2d(n_feats//3, n_feats//3, 1, 1, 0))
        self.LKA3 = nn.Sequential(
            nn.Conv2d(n_feats//3, n_feats//3, 3, 1, 1, groups= n_feats//3),  
            nn.Conv2d(n_feats//3, n_feats//3, 5, stride=1, padding=(5//2)*2, groups=n_feats//3, dilation=2),
            nn.Conv2d(n_feats//3, n_feats//3, 1, 1, 0))
        
        self.X3 = nn.Conv2d(n_feats//3, n_feats//3, 3, 1, 1, groups= n_feats//3)
        self.X5 = nn.Conv2d(n_feats//3, n_feats//3, 5, 1, 5//2, groups= n_feats//3)
        self.X7 = nn.Conv2d(n_feats//3, n_feats//3, 7, 1, 7//2, groups= n_feats//3)
        
        self.proj_first = nn.Sequential(
            nn.Conv2d(n_feats, i_feats, 1, 1, 0))

        self.proj_last = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0))

        
    def forward(self, x, pre_attn=None, RAA=None):
        shortcut = x.clone()
        
        x = self.norm(x)
        
        x = self.proj_first(x)
        
        a, x = torch.chunk(x, 2, dim=1) 
        
        a_1, a_2, a_3= torch.chunk(a, 3, dim=1)
        
        a = torch.cat([self.LKA3(a_1)*self.X3(a_1), self.LKA5(a_2)*self.X5(a_2), self.LKA7(a_3)*self.X7(a_3)], dim=1)
        
        x = self.proj_last(x*a)*self.scale + shortcut
        
        return x 



class SGAB(nn.Module):
    def __init__(self, n_feats, drop=0.0, k=2, squeeze_factor= 15, attn ='GLKA'):   
        super().__init__()
        i_feats =n_feats*2
        
        self.Conv1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0) 
        self.DWConv1 = nn.Conv2d(n_feats, n_feats, 7, 1, 7//2, groups= n_feats)     
        self.Conv2 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)
        
        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)
        
    def forward(self, x):      
        shortcut = x.clone()
        
        #Ghost Expand      
        x = self.Conv1(self.norm(x))
        a, x = torch.chunk(x, 2, dim=1) 
        x = x*self.DWConv1(a)
        x = self.Conv2(x)
        
        return  x*self.scale + shortcut
    


class MAB(nn.Module):
    def __init__(
        self,in_channel, n_feats):   
        super().__init__()

        self.conv_in = nn.Conv2d(in_channel, (in_channel//3)*3, 1, 1, 0)
        
        self.LKA = MLKA((in_channel//3)*3) 
        
        self.LFE = GSAU( (in_channel//3)*3)
        self.conv_out = nn.Conv2d((in_channel//3)*3 ,n_feats, 1, 1, 0)
        
    def forward(self, x, pre_attn=None, RAA=None): 
        #large kernel attention
        x = self.conv_in(x)
        x  = self.LKA(x)  
        
        #local feature extraction
        x = self.LFE(x)  
        x = self.conv_out(x)
        
        return x   

class MLKA(nn.Module):
    r"""Multi-scale Large Kernel Attention.

    Args:
        n_feats: Number of input channels

    """

    def __init__(self, n_feats: int) -> None:
        super().__init__()
        i_feats = 2 * n_feats

        self.n_feats = n_feats
        self.i_feats = i_feats
        self.norm = LayerNorm4D(n_feats)
        self.scale = nn.Parameter(torch.zeros(
            (1, n_feats, 1, 1)), requires_grad=True)

        self.LKA7 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 7,
                      1, 7 // 2, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 9, 1, (9 // 2)
                      * 4, groups=n_feats // 3, dilation=4),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))
        self.LKA5 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 5,
                      1, 5 // 2, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 7, 1, (7 // 2)
                      * 3, groups=n_feats // 3, dilation=3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))
        self.LKA3 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3,
                      3, 1, 1, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 5, 1, (5 // 2)
                      * 2, groups=n_feats // 3, dilation=2),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))

        self.X3 = nn.Conv2d(n_feats // 3, n_feats // 3,
                            3, 1, 1, groups=n_feats // 3)
        self.X5 = nn.Conv2d(n_feats // 3, n_feats // 3, 5,
                            1, 5 // 2, groups=n_feats // 3)
        self.X7 = nn.Conv2d(n_feats // 3, n_feats // 3, 7,
                            1, 7 // 2, groups=n_feats // 3)

        self.proj_first = Conv2d1x1(n_feats, i_feats)
        self.proj_last = Conv2d1x1(n_feats, n_feats)

    def forward(self, x) -> torch.Tensor:
        shortcut = x.clone()

        x = self.norm(x)
        x = self.proj_first(x)
        a, x = torch.chunk(x, 2, dim=1)
        a_1, a_2, a_3 = torch.chunk(a, 3, dim=1)
        a = torch.cat([self.LKA3(a_1) * self.X3(a_1), self.LKA5(a_2)
                      * self.X5(a_2), self.LKA7(a_3) * self.X7(a_3)], dim=1)
        x = self.proj_last(x * a)

        return x * self.scale + shortcut
    
class LKAT(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
       
        #self.norm = LayerNorm(n_feats, data_format='channels_first')
        #self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)
        
        self.conv0 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0),
            nn.GELU())
        
        self.att = nn.Sequential(
                nn.Conv2d(n_feats, n_feats, 7, 1, 7//2, groups= n_feats),  
                nn.Conv2d(n_feats, n_feats, 9, stride=1, padding=(9//2)*3, groups=n_feats, dilation=3),
                nn.Conv2d(n_feats, n_feats, 1, 1, 0))  

        self.conv1 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)
        
    def forward(self, x):
        x = self.conv0(x)
        x = x*self.att(x) 
        x = self.conv1(x) 
        return x    
    

if __name__ == '__main__':
    n_feats = 12
    block = MAB(n_feats)
    # block  = ResnetBlock(in_channels=n_feats,out_channels=n_feats,dropout=0.1,temb_channels=0)
    # block = LKAT(n_feats)
    from thop import profile
    from thop import clever_format
    # input = torch.rand(1, 3,3840, 2160)
    # input = torch.rand(1, 12, 256, 256)
    # input = torch.rand(1, 3, 1024, 1024)
    input = torch.rand(1, 12, 256, 256)
    flops,params = profile(block,inputs=(input,))
    flops,params = clever_format([flops,params], "%.3f")
    print(f"params:{params},flops:{flops}")
    output = block(input)
    print(input.size())
    print(output.size())
