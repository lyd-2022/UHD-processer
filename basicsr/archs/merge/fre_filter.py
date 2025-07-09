import torch
import torch.nn as nn
import torch.fft as fft

import torch.fft as fft

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
        self.output1_conv = nn.Conv2d(4 * C1 + C2, C2, kernel_size=1)
        self.output2_conv = nn.Conv2d(4 * C1 + C2, 1 * C1, kernel_size=1)
        self.output3_conv = nn.Conv2d(4 * C1 + C2, 3 * C1, kernel_size=1)

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


if __name__ == '__main__':
    x1 = torch.randn(2, 4*3, 8, 8)
    x2 = torch.randn(2, 3, 8, 8)
    model = FFT_filter(3, 3)
    out = model(x1, x2)
    print(out.shape)