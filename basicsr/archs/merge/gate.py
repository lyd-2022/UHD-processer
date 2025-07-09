import torch
import torch.nn as nn

class FeatureFusion(nn.Module):
    def __init__(self, channel1,channel2):
        super(FeatureFusion, self).__init__()
        self.conv = nn.Conv2d(channel1+channel2, channel1, kernel_size=1)

    def forward(self, x1, x2):
        h = torch.cat([x1, x2], dim=1)  # 在通道维度上拼接
        h = self.conv(h)  # 1x1卷积融合特征
        return h

class GatingMechanism(nn.Module):
    def __init__(self, channel1,channel2):
        super(GatingMechanism, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv2d( channel1+channel2, channel1, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channel1, channel1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        h = torch.cat([x1, x2], dim=1)  # 在通道维度上拼接
        g = self.gate(h)  # 生成门控向量
        return g

class FeatureEnhancement(nn.Module):
    def __init__(self, channel1,channel2):
        super(FeatureEnhancement, self).__init__()
        self.feature_fusion = FeatureFusion(channel1,channel2)
        self.gating_mechanism = GatingMechanism(channel1,channel2)

    def forward(self, x1, x2):
        fused_feature = self.feature_fusion(x1, x2)
        gate = self.gating_mechanism(x1, x2)
        enhanced_feature = gate * fused_feature + (1 - gate) * x1  # 使用门控向量进行增强
        return enhanced_feature

class GatedFeatureEnhancement(nn.Module):
    def __init__(self, channel1,channel2):
        super(GatedFeatureEnhancement, self).__init__()
        self.feature_enhancement = FeatureEnhancement(channel1,channel2)

    def forward(self, x1, x2):
        enhanced_feature = self.feature_enhancement(x1, x2)
        return enhanced_feature


if __name__ == '__main__':
    x1 = torch.randn(2, 64, 32, 32)
    x2 = torch.randn(2, 32, 32, 32)
    gate_enhance = GatedFeatureEnhancement(64,32)
    out = gate_enhance(x1, x2)
    print(out.shape)  # torch.Size([2, 64, 32, 32])