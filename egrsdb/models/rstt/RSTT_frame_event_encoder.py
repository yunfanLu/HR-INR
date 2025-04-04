import numpy as np
import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F

from egrsdb.models.rstt.RSTT import RSTT_L, RSTT_M, RSTT_S, get_rstt_large, get_rstt_medium, get_rstt_small


class _SCN(nn.Module):
    def __init__(self, frames_channels, events_channels, hidden_channels, loop):
        super(_SCN, self).__init__()
        self.hidden_channels = hidden_channels
        self.loop = loop

        self.frames_channels_reduce = nn.Conv2d(frames_channels, hidden_channels, 1, 1, 0, bias=False)
        self.events_channels_reduce = nn.Sequential(
            nn.Conv2d(events_channels, hidden_channels, 1, 1, 0, bias=False), nn.Sigmoid()
        )

        self.W1 = nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, bias=False)
        self.S1 = nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, groups=1, bias=False)
        self.S2 = nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, groups=1, bias=False)
        self.shlu = nn.ReLU(True)
        self.frames_channels_restore = nn.Conv2d(hidden_channels, frames_channels, 1, 1, 0, bias=False)

    def forward(self, multi_frame_feature, events):
        """
        multi_frame_feature: B, D, C, H, W
        events: B, C, H, W
        """
        x1 = multi_frame_feature
        B, D, C, H, W = x1.size()
        x1 = rearrange(x1, "b d c h w -> b (d c) h w")
        x1 = self.frames_channels_reduce(x1)
        event_input = self.events_channels_reduce(events)

        x1 = torch.mul(x1, event_input)
        z = self.W1(x1)
        tmp = z
        for i in range(self.loop):
            ttmp = self.shlu(tmp)
            x = self.S1(ttmp)
            x = torch.mul(x, event_input)
            x = torch.mul(x, event_input)
            x = self.S2(x)
            x = ttmp - x
            tmp = torch.add(x, z)
        c = self.shlu(tmp)
        c = self.frames_channels_restore(c)
        c = rearrange(c, "b (d c) h w -> b d c h w", d=D)
        return c


class RSTTwithEventAdapter(nn.Module):
    def __init__(
        self,
        only_train_adapter,
        rstt_type,
        input_frame,
        events_moments,
        rstt_pretrain_type,
        event_adapter_type,
        event_adatper_config,
    ):
        super(RSTTwithEventAdapter, self).__init__()
        # check the args is correct
        assert rstt_type in ["large", "medium", "small"]
        assert event_adapter_type in ["SCN"]
        assert input_frame >= 2 and input_frame % 2 == 0, f"the input_frame must be even number, but got {input_frame}"
        if only_train_adapter is True:
            assert (
                rstt_pretrain_type == "using_pretrained"
            ), "only_train_adapter must be True, when rstt_pretrain_type is using_pretrained"
        # set the config to the class
        self.only_train_adapter = only_train_adapter
        self.rstt_type = rstt_type
        self.input_frame = input_frame
        self.events_moments = events_moments
        self.rstt_pretrain_type = rstt_pretrain_type
        self.event_adapter_type = event_adapter_type
        self.event_adatper_config = event_adatper_config
        # RSTT configs:
        NUM_ENCODER_LAYERS = 4  # for the RSTT the encoder is 4 laryer.
        self.num_enc_layers = NUM_ENCODER_LAYERS
        self.num_dec_layers = NUM_ENCODER_LAYERS
        self.scale = 2 ** (self.num_enc_layers - 1)
        # get the rstt model
        self.rstt = self._get_rstt_model()
        self.embed_dim = self.rstt.embed_dim
        # get the event adapter
        self.event_head = self._get_events_head()
        self.event_encoders = self._get_event_encoders()
        # the event adapter type, which will be trained
        self.encoder_event_adapters = self._get_encoder_event_adapters()
        self.decoder_event_adapters = self._get_decoder_event_adapters()
        self.final_adapter = self._get_final_adapter()

    def forward(self, events, frames):
        """
        events: [B, events_moments, H, W]
        frames: [B, input_frame, C, H, W]
        return: [B, inr_channel, H, W]
        """
        # D input video frames
        B, D, C, H, W = frames.size()
        # B, D, C, H, W - > B, D, embed_dim, H, W
        frames_feature = self.rstt.input_proj(frames)
        events_feature = self.event_head(events)

        Hp = int(np.ceil(H / self.scale)) * self.scale
        Wp = int(np.ceil(W / self.scale)) * self.scale
        frames_feature = F.pad(frames_feature, (0, Wp - W, 0, Hp - H))

        # events encoded feature
        events_encoded_feature = [events_feature]
        for i in range(self.num_enc_layers - 1):
            events_feature = self.event_encoders[i](events_feature)
            events_encoded_feature.append(events_feature)

        # encoder
        encoder_features = []
        x = frames_feature
        for i_layer in range(self.num_enc_layers):
            if self.only_train_adapter:
                with torch.no_grad():
                    x = self.rstt.encoder_layers[i_layer](x)
            else:
                x = self.rstt.encoder_layers[i_layer](x)
            # add the event adapter
            x_res = self.encoder_event_adapters[i_layer](x, events_encoded_feature[i_layer])
            x = x + x_res
            encoder_features.append(x)
            if i_layer != self.num_enc_layers - 1:
                if self.only_train_adapter:
                    with torch.no_grad():
                        x = self.rstt.downsample[i_layer](x)
                else:
                    x = self.rstt.downsample[i_layer](x)

        _, _, C, h, w = x.size()
        y = torch.zeros((B, self.rstt.num_out_frames, C, h, w), device=x.device)
        for i in range(self.rstt.num_out_frames):
            if i % 2 == 0:
                y[:, i, :, :, :] = x[:, i // 2]
            else:
                y[:, i, :, :, :] = (x[:, i // 2] + x[:, i // 2 + 1]) / 2

        # decoder
        for i_layer in range(self.num_dec_layers):
            if self.only_train_adapter:
                with torch.no_grad():
                    y = self.rstt.decoder_layers[i_layer](y, encoder_features[-i_layer - 1])
            else:
                y = self.rstt.decoder_layers[i_layer](y, encoder_features[-i_layer - 1])

            # add the event adapter
            y_res = self.decoder_event_adapters[i_layer](y, events_encoded_feature[-i_layer - 1])
            y = y + y_res
            if i_layer != self.num_dec_layers - 1:
                if self.only_train_adapter:
                    with torch.no_grad():
                        y = self.rstt.upsample[i_layer](y)
                else:
                    y = self.rstt.upsample[i_layer](y)
        y = y[:, :, :, :H, :W].contiguous()

        # construct INR
        # B, D, embed_dim, H, W - > B, D * C, H, W
        inr = self.final_adapter(y, events_encoded_feature[0])
        inr = rearrange(inr, "b d c h w -> b (d c) h w")
        return inr

    def _get_events_head(self):
        hidden_channels = self.embed_dim * self.input_frame
        event_c1 = nn.Conv2d(
            in_channels=self.events_moments,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        sigmod_1 = nn.Sigmoid()
        event_c2 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        sigmod_2 = nn.Sigmoid()
        return nn.Sequential(event_c1, sigmod_1, event_c2, sigmod_2)

    def _get_event_encoders(self):
        hidden_channels = self.embed_dim * self.input_frame
        event_encoders = nn.ModuleList()
        for i in range(self.num_enc_layers):
            event_encoders.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False,
                    ),
                    nn.ReLU(inplace=True),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                )
            )
        return event_encoders

    def _get_encoder_event_adapters(self):
        frames_channels = self.embed_dim * self.input_frame
        encoder_event_adapters = nn.ModuleList()
        for i in range(self.num_enc_layers):
            encoder_event_adapters.append(
                _SCN(
                    frames_channels=frames_channels,
                    events_channels=frames_channels,
                    hidden_channels=self.event_adatper_config.hidden_channels,
                    loop=self.event_adatper_config.loop,
                )
            )
        return encoder_event_adapters

    def _get_decoder_event_adapters(self):
        frames_channels = self.embed_dim * (2 * self.input_frame - 1)
        event_channels = self.embed_dim * self.input_frame

        decoder_event_adapters = nn.ModuleList()
        for i in range(self.num_dec_layers):
            decoder_event_adapters.append(
                _SCN(
                    frames_channels=frames_channels,
                    events_channels=event_channels,
                    hidden_channels=self.event_adatper_config.hidden_channels,
                    loop=self.event_adatper_config.loop,
                )
            )
        return decoder_event_adapters

    def _get_final_adapter(self):
        frames_channels = self.embed_dim * (2 * self.input_frame - 1)
        event_channels = self.embed_dim * self.input_frame
        return _SCN(
            frames_channels=frames_channels,
            events_channels=event_channels,
            hidden_channels=self.event_adatper_config.hidden_channels,
            loop=self.event_adatper_config.loop,
        )

    def _get_rstt_model(self):
        if self.rstt_type == "large":
            rstt = get_rstt_large()
            checkpoint = RSTT_L
        elif self.rstt_type == "medium":
            rstt = get_rstt_medium()
            checkpoint = RSTT_M
        else:
            rstt = get_rstt_small()
            checkpoint = RSTT_S

        # load the pretrain model
        if self.rstt_pretrain_type == "using_pretrained":
            rstt.load_state_dict(torch.load(checkpoint))
        return rstt
