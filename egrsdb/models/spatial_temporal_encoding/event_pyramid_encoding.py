from absl.logging import info
from torch import nn


class EventPyramidEncoding3xConv(nn.Module):
    def __init__(self, pyramid_level, pyramid_moments, epre_channel, epr_out_channel):
        super(EventPyramidEncoding3xConv, self).__init__()
        self.pyramid_level = pyramid_level
        self.pyramid_moments = pyramid_moments
        self.in_channel = pyramid_level * pyramid_moments
        self.epre_channel = epre_channel  # event pyramid representation encoding channel
        self.epr_out_channel = epr_out_channel
        # encoder
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(self.in_channel, epre_channel, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(epre_channel, epre_channel, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(epre_channel, epre_channel, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(epre_channel, epre_channel, 3, padding=1),
            nn.ReLU(),
        )
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(self.in_channel, epre_channel, 3, padding=1),
            nn.ReLU(),
        )
        self.to_temporal = nn.Conv2d(epre_channel, epr_out_channel, 1, padding=0)
        self._info()

    def forward(self, event_pyramid_representation_local):
        # epr: [B, PL, PM, H, W]
        B, PL, PM, H, W = event_pyramid_representation_local.shape
        # reshape, [B, PL, PM, H, W] -> [B, PL * PM, H, W]
        event_pyramid_representation_local = event_pyramid_representation_local.reshape(B, PL * PM, H, W)
        epr_inr_1 = self.encoder_1(event_pyramid_representation_local)
        epr_inr_2 = self.encoder_2(event_pyramid_representation_local)
        epr_inr = epr_inr_1 + epr_inr_2
        epr_inr = self.to_temporal(epr_inr)
        return epr_inr

    def _info(self):
        info("Init EventPyramidEncoding3xConv")
        info(f"  pyramid_level  : {self.pyramid_level}")
        info(f"  pyramid_moments: {self.pyramid_moments}")
        info(f"  in_channel     : {self.in_channel}")
        info(f"  epre_channel   : {self.epre_channel}")
        info(f"  epr_out_channel: {self.epr_out_channel}")
