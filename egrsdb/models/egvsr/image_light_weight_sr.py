from torch import nn

from egrsdb.models.egvsr.lcb import LightWeightCNNBackbone
from egrsdb.models.egvsr.ltb import MLABlock
from egrsdb.models.egvsr.patch_options import restore_patches
from egrsdb.models.egvsr.upsampler import Upsampler
from egrsdb.models.egvsr.weight import Weight


class LightCNNTransformer(nn.Module):
    def __init__(self, n_feats, depth):
        super(LightCNNTransformer, self).__init__()
        self.lcb = LightWeightCNNBackbone(in_channels=n_feats, depth=depth)
        self.ltb = MLABlock(dim=3 * 3 * n_feats)
        self.alise = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.weight1 = Weight(1)
        self.weight2 = Weight(1)

    def forward(self, feature):
        x = self.lcb(feature)
        x = self.ltb(x)
        x = x.permute(0, 2, 1)
        x = restore_patches(x, (128, 128), (3, 3), 1, 1)
        x = self.alise(x)
        x = self.weight1(feature) + self.weight2(x)
        return x


class ImageLightWeightSR(nn.Module):
    def __init__(self, scale, in_channels, n_feats, depth):
        super(ImageLightWeightSR, self).__init__()
        self.head = nn.Conv2d(in_channels, n_feats, 3, padding=1)

        self.body = LightCNNTransformer(n_feats, depth)

        self.body_to_tail = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            Upsampler(scale, n_feats),
            nn.Conv2d(n_feats, in_channels, 3, padding=1),
        )
        self.head_to_tail = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            Upsampler(scale, n_feats),
            nn.Conv2d(n_feats, in_channels, 3, padding=1),
        )

    def forward(self, image):
        x_head = self.head(image)
        x_body = self.body(x_head)
        body_tail = self.body_to_tail(x_body)
        head_tail = self.head_to_tail(x_head)
        sr = body_tail + head_tail
        return sr
