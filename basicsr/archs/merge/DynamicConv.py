import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DynamicConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding

        # 生成卷积核的线性层
        self.kernel_generator = nn.Linear(in_channels, out_channels * kernel_size * kernel_size)

    def forward(self, x, dynamic_weights):
        B, C, H, W = x.shape

        # 生成卷积核
        kernels = self.kernel_generator(dynamic_weights)  # (B, out_channels * kernel_size * kernel_size)
        kernels = kernels.view(B, self.out_channels, C, self.kernel_size, self.kernel_size)

        # 扩展输入的维度，以匹配动态卷积核
        x = x.view(1, B * C, H, W)
        kernels = kernels.view(B * self.out_channels, C, self.kernel_size, self.kernel_size)

        # 应用卷积
        output = F.conv2d(x, kernels, stride=self.stride, padding=self.padding, groups=B)
        output = output.view(B, self.out_channels, H, W)

        return output

class ComplexFusionModule(nn.Module):
    def __init__(self, C1, C2):
        super(ComplexFusionModule, self).__init__()
        
        self.C1 = C1
        self.C2 = C2
        
        # 动态卷积层
        self.dynamic_convs = nn.ModuleList([
            DynamicConv2d(C2, C2) for _ in range(4)
        ])
        
        # 通道权重生成模块
        self.channel_weight_generator = nn.Sequential(
            nn.Conv2d(4 * C1, C2, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 输出模块
        self.output1_conv = nn.Conv2d(C2, C2, kernel_size=1)
        self.output2_conv = nn.Conv2d(4 * C1, 1 * C1, kernel_size=1)
        self.output3_conv = nn.Conv2d(4 * C1, 3 * C1, kernel_size=1)

    def forward(self, x1, x2):
        # x1: (B, 4*C1, H, W)
        # x2: (B, C2, H, W)
        B, _, H, W = x1.shape
        
        # 1. 特征划分
        x1_splits = torch.split(x1, self.C1, dim=1)  # 划分为4个(C1, H, W)的块
        
        # 2. 生成卷积核并应用卷积
        outputs1 = [self.dynamic_convs[i](x2, x1_splits[i].view(B, -1)) for i in range(4)]
        output1 = sum(outputs1)  # 将4个输出累加

        # 3. 通道权重生成和调制
        channel_weight = self.channel_weight_generator(x1)
        output2 = x2 * channel_weight  # 调制输入2

        # 4. 特征融合
        fused_feature = output1 + output2

        # 5. 多输出生成
        output1_final = self.output1_conv(fused_feature)
        output2_final = self.output2_conv(x1)
        output3_final = self.output3_conv(x1)

        return output1_final, output2_final, output3_final

# 示例用法
B = 8
C1 = 16
C2 = 32
H = 64
W = 64

model = ComplexFusionModule(C1, C2)
x1 = torch.randn(B, 4 * C1, H, W)
x2 = torch.randn(B, C2, H, W)

output1, output2, output3 = model(x1, x2)
print(output1.shape)  # (B, C2, H, W)
print(output2.shape)  # (B, 1*C1, H, W)
print(output3.shape)  # (B, 3*C1, H, W)
