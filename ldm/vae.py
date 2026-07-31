import torch
import torch.nn as nn

from ldm.autoencoder import (
    Decoder,
    Downsample,
    Normalize,
    ResnetBlock,
    make_attn,
    nonlinearity,
)


class Encoder(nn.Module):
    def __init__(
        self,
        ch,
        ch_mult,
        num_res_blocks,
        in_channels,
        z_channels,
        dropout=0.0,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.conv_in = nn.Conv2d(in_channels, ch, 3, padding=1)

        input_multipliers = (1, *ch_mult)
        self.down = nn.ModuleList()
        for level, multiplier in enumerate(ch_mult):
            block_in = ch * input_multipliers[level]
            block_out = ch * multiplier
            down = nn.Module()
            down.block = nn.ModuleList()
            for _ in range(num_res_blocks):
                down.block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=0,
                        dropout=dropout,
                    )
                )
                block_in = block_out
            if level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, True)
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=0,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=0,
            dropout=dropout,
        )
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, 3, padding=1)

    def forward(self, image):
        hidden = self.conv_in(image)
        for level, down in enumerate(self.down):
            for block in down.block:
                hidden = block(hidden, None)
            if level != self.num_resolutions - 1:
                hidden = down.downsample(hidden)

        hidden = self.mid.block_1(hidden, None)
        hidden = self.mid.attn_1(hidden)
        hidden = self.mid.block_2(hidden, None)
        hidden = nonlinearity(self.norm_out(hidden))
        return self.conv_out(hidden)


class PanoramaVAE(nn.Module):
    def __init__(self):
        super().__init__()
        config = {
            "ch": 128,
            "ch_mult": (1, 2, 4, 4),
            "num_res_blocks": 2,
            "in_channels": 3,
            "z_channels": 4,
            "dropout": 0.0,
        }
        self.encoder = Encoder(**config)
        self.decoder = Decoder(
            **config,
            out_ch=3,
            resolution=256,
            attn_resolutions=(),
        )
        self.quant_conv = nn.Conv2d(8, 8, 1)
        self.post_quant_conv = nn.Conv2d(4, 4, 1)

    def forward(self, image):
        mean, log_variance = self.quant_conv(self.encoder(image)).chunk(2, dim=1)
        deviation = torch.exp(0.5 * torch.clamp(log_variance, -30.0, 20.0))
        latent = mean + deviation * torch.randn_like(mean)
        return self.decoder(self.post_quant_conv(latent))


def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state = model.state_dict()
    filtered = {
        name: value for name, value in state_dict.items() if name in model_state
    }
    model.load_state_dict(filtered, strict=False)
    return len(filtered), len(state_dict) - len(filtered)
