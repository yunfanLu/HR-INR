from absl.logging import info
from torch import nn

from egrsdb.models.rstt.layers import EncoderLayer


class EventPyramidRepresentationSwinTransformerEncoder(nn.Module):
    def __init__(
        self,
        pyramid_level,
        pyramid_moments,
        epre_channel,
        epr_out_channel,
        depth=3,
        num_heads=8,
        window_size=(4, 4),
    ):
        super(EventPyramidRepresentationSwinTransformerEncoder, self).__init__()
        self.pl = pyramid_level
        self.pm = pyramid_moments
        self.ec = epre_channel  # event pyramid representation encoding channel
        # encoder
        self.epr_head = nn.Conv2d(self.pl * self.pm, self.ec * self.pl, kernel_size=1, stride=1, padding=0)
        self.epr_encoder = EncoderLayer(
            dim=self.ec, depth=depth, num_heads=num_heads, num_frames=pyramid_level, window_size=window_size
        )
        self.to_temporal = nn.Conv2d(self.ec * self.pl, epr_out_channel, 1, padding=0)

    def forward(self, epr):
        # epr: [B, PL, PM, H, W]
        B, PL, PM, H, W = epr.shape
        # reshape, [B, PL, PM, H, W] -> [B, PL * PM, H, W]
        epr = epr.reshape(B, PL * PM, H, W)
        epr = self.epr_head(epr)
        epr = epr.reshape(B, PL, self.ec, H, W)
        epr_inr = self.epr_encoder(epr)
        epr_inr = epr_inr.reshape(B, PL * self.ec, H, W)
        epr_inr = self.to_temporal(epr_inr)
        return epr_inr
