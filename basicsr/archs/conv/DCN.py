import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d

# 定义输入特征图
batch_size = 1
in_channels = 3
out_channels = 64
kernel_size = 3
height, width = 32, 32

input = torch.randn(batch_size, in_channels, height, width)

# 定义可变形卷积层
deform_conv = DeformConv2d(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    stride=1,
    padding=1,
    dilation=1,
    groups=1,
    bias=True
)

# 预定义的权重
weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

# 预定义的偏移量
offset = nn.Parameter(torch.randn(batch_size, 2 * kernel_size * kernel_size, height, width))

# 将预定义的权重和偏移量传递给可变形卷积层
output = deform_conv(input, offset, weight=weight)

print(output.shape)  # 输出卷积结果的形状