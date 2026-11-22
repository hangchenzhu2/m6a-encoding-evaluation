# SE / ECA / CBAM channel attention, 1-D versions (after Hu 2018, Wang 2020, Woo 2018)
import torch
import torch.nn as nn
import torch.nn.functional as F


class CBAMBlock1d(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        # channel attention
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel)
        )
        # 1-D spatial attention
        self.conv_spatial = nn.Conv1d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, L = x.size()
        # channel attention branch
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        avg_att = self.mlp(avg_out)
        max_att = self.mlp(max_out)
        channel_att = self.sigmoid(avg_att + max_att).view(b, c, 1)
        x = x * channel_att

        # spatial branch
        avg_sp = torch.mean(x, dim=1, keepdim=True)
        max_sp, _ = torch.max(x, dim=1, keepdim=True)
        spatial_in = torch.cat([avg_sp, max_sp], dim=1)
        spatial_att = self.sigmoid(self.conv_spatial(spatial_in))
        x = x * spatial_att
        return x


class ECABlock1d(nn.Module):
    def __init__(self, channel, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # 1-D convolution instead of FC, no channel reduction
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, L]
        b, c, l = x.size()
        # Squeeze [B,C,L] -> [B,C,1]
        y = self.avg_pool(x)
        # reshape for Conv1d: [B,C,1] -> [B,1,C]
        y = y.transpose(-1, -2)
        # local channel interaction
        y = self.conv(y)
        y = self.sigmoid(y)
        # restore shape: [B,1,C] -> [B,C,1]
        y = y.transpose(-1, -2)
        # feature re-weighting
        return x * y.expand_as(x)

# 1-D squeeze-and-excitation block


class SEBlock1d(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        # squeeze: global pooling
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # excitation: two FC layers learn channel weights
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [batch, channels, length]
        b, c, _ = x.size()
        # squeeze: global average pooling
        weight = self.avg_pool(x).view(b, c)
        # excitation: per-channel weight in [0, 1]
        weight = self.fc(weight).view(b, c, 1)
        # scale the input by the channel weights
        return x * weight.expand_as(x)


