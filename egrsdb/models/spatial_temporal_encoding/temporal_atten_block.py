import torch
import torch.nn.functional as F
from torch import nn

from egrsdb.models.egvsr.weight import Weight


class _SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(_SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class _CALayer(nn.Module):
    def __init__(self, channel, cal_kernel_size, reduction=16):
        super(_CALayer, self).__init__()
        cks = cal_kernel_size
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, cks, padding=cks // 2, bias=True),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, cks, padding=cks // 2, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class _ResidualUnits(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3):
        super(_ResidualUnits, self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size, padding=kernel_size >> 1, stride=1)
        self.conv1 = nn.Conv2d(growth_rate, in_channels, kernel_size=kernel_size, padding=kernel_size >> 1, stride=1)
        self.relu = nn.PReLU(growth_rate)
        self.weight1 = Weight(1)
        self.weight2 = Weight(1)

    def forward(self, x):
        x1 = self.conv1(self.relu(self.conv(x)))
        output = self.weight1(x) + self.weight2(x1)
        return output


class _ConvRelu(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=1):
        super(_ConvRelu, self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class _AdaptiveResidualFeatureBlock(nn.Module):
    def __init__(self, n_feats, cal_kernel_size):
        super(_AdaptiveResidualFeatureBlock, self).__init__()
        self.layer1 = _ResidualUnits(n_feats, n_feats // 2, 3)
        self.layer2 = _ResidualUnits(n_feats, n_feats // 2, 3)
        self.alise = _ConvRelu(2 * n_feats, n_feats, 1, 1, 0)
        self.atten_cal = _CALayer(n_feats, cal_kernel_size)
        self.layer4 = _ConvRelu(n_feats, n_feats, 3, 1, 1)
        self.atten_se = _SELayer(n_feats)
        self.weight2 = Weight(1)
        self.weight3 = Weight(1)
        self.weight4 = Weight(1)
        self.weight5 = Weight(1)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = torch.cat([self.weight2(x2), self.weight3(x1)], 1)
        x4 = self.alise(x3)
        x5 = self.atten_cal(x4)
        x6 = self.layer4(x5)
        x7 = self.atten_se(x6)
        return self.weight4(x) + self.weight5(x7)


class _HighPreservingBlock(nn.Module):
    def __init__(self, n_feats, loop_count, cal_kernel_size):
        super(_HighPreservingBlock, self).__init__()
        self.loop_count = loop_count

        self.encoder = _AdaptiveResidualFeatureBlock(n_feats, cal_kernel_size)
        self.down = nn.AvgPool2d(kernel_size=2)
        self.decoder_low = _AdaptiveResidualFeatureBlock(n_feats, cal_kernel_size)
        self.decoder_high = _AdaptiveResidualFeatureBlock(n_feats, cal_kernel_size)
        self.alise2 = _ConvRelu(2 * n_feats, n_feats, 1, 1, 0)
        self.att = _CALayer(n_feats, cal_kernel_size)
        self.alise = _AdaptiveResidualFeatureBlock(n_feats, cal_kernel_size)

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.down(x1)
        high = x1 - F.interpolate(x2, size=x.size()[-2:], mode="bilinear", align_corners=True)
        for i in range(self.loop_count):
            x2 = self.decoder_low(x2)
        x3 = x2
        x4 = F.interpolate(x3, size=x.size()[-2:], mode="bilinear", align_corners=True)
        high1 = self.decoder_high(high)
        x5 = self.alise2(torch.cat([x4, high1], dim=1))
        x6 = self.att(x5)
        x7 = self.alise(x6) + x
        return x7


class TemporalAttentionBlock(nn.Module):
    def __init__(self, in_channels, depth, loop_count, cal_kernel_size):
        super(TemporalAttentionBlock, self).__init__()
        self.depth = depth
        self.hpbs = nn.ModuleList()
        for i in range(depth):
            self.hpbs.append(_HighPreservingBlock(in_channels, loop_count, cal_kernel_size))
        self.reduce = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        res = x
        for i in range(self.depth):
            x = self.hpbs[i](x)
            res = res + x
        out = self.reduce(res)
        return out
