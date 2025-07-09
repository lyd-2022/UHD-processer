import torch
from pytorch_wavelets import DWTForward, DWTInverse

# 初始化小波变换和逆变换
dwt = DWTForward(J=3, wave='db1')
idwt = DWTInverse(wave='db1')

# 生成一个随机信号
x = torch.randn(1, 3, 64, 64)

# 进行正变换，获取分解系数
coeffs = dwt(x)

# 逐层逆变换
# 从最深层开始，逐层进行逆变换
current_coeffs = coeffs
for j in range(3, 0, -1):
    # 获取当前层的低频分量
    LL = current_coeffs[0]
    
    # 获取当前层的高频分量
    H = current_coeffs[1][j-1]

    
    # 进行逆变换
    reconstructed = idwt([LL, H])
    
    # 更新当前层的分解系数
    if j > 1:
        current_coeffs = dwt(reconstructed)

# 最终重建的信号
y = reconstructed

# 计算误差
error = torch.norm(x - y)
print(f"Error: {error.item()}")