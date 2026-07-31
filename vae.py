import torch
import torch.nn as nn
import torch.nn.functional as F


CHANNELS = 128
CHANNEL_MULTIPLIERS = (1, 2, 4, 4)
RESIDUAL_BLOCKS = 2
LATENT_CHANNELS = 4


def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state = model.state_dict()
    filtered = {name: value for name, value in state_dict.items() if name in model_state}
    model.load_state_dict(filtered, strict=False)
    return len(filtered), len(state_dict) - len(filtered)


def swish(value):
    return value * torch.sigmoid(value)


def normalize(channels):
    return nn.GroupNorm(32, channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, value):
        value = F.interpolate(value, scale_factor=2.0, mode="nearest")
        return self.conv(value)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2)

    def forward(self, value):
        return self.conv(F.pad(value, (0, 1, 0, 1)))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, 3, padding=1)
        self.norm2 = normalize(self.out_channels)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
        if in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, self.out_channels, 1)

    def forward(self, value):
        hidden = self.conv1(swish(self.norm1(value)))
        hidden = self.conv2(swish(self.norm2(hidden)))
        if self.in_channels != self.out_channels:
            value = self.nin_shortcut(value)
        return value + hidden


class AttnBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = normalize(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, value):
        hidden = self.norm(value)
        query = self.q(hidden)
        key = self.k(hidden)
        projected = self.v(hidden)
        batch, channels, height, width = query.shape
        query = query.reshape(batch, channels, height * width).permute(0, 2, 1)
        key = key.reshape(batch, channels, height * width)
        weights = torch.bmm(query, key) * channels**-0.5
        weights = weights.softmax(dim=2).permute(0, 2, 1)
        hidden = torch.bmm(projected.reshape(batch, channels, -1), weights)
        hidden = hidden.reshape(batch, channels, height, width)
        return value + self.proj_out(hidden)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        block_in = CHANNELS * CHANNEL_MULTIPLIERS[-1]
        self.conv_in = nn.Conv2d(LATENT_CHANNELS, block_in, 3, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in)

        self.up = nn.ModuleList()
        for level in reversed(range(len(CHANNEL_MULTIPLIERS))):
            block_out = CHANNELS * CHANNEL_MULTIPLIERS[level]
            up = nn.Module()
            up.block = nn.ModuleList()
            for _ in range(RESIDUAL_BLOCKS + 1):
                up.block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
            if level:
                up.upsample = Upsample(block_in)
            self.up.insert(0, up)

        self.norm_out = normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, 3, 3, padding=1)

    def forward(self, latent):
        hidden = self.conv_in(latent)
        hidden = self.mid.block_1(hidden)
        hidden = self.mid.attn_1(hidden)
        hidden = self.mid.block_2(hidden)
        for level in reversed(range(len(self.up))):
            for block in self.up[level].block:
                hidden = block(hidden)
            if level:
                hidden = self.up[level].upsample(hidden)
        return self.conv_out(swish(self.norm_out(hidden)))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(3, CHANNELS, 3, padding=1)
        self.down = nn.ModuleList()
        block_in = CHANNELS
        for level, multiplier in enumerate(CHANNEL_MULTIPLIERS):
            block_out = CHANNELS * multiplier
            down = nn.Module()
            down.block = nn.ModuleList()
            for _ in range(RESIDUAL_BLOCKS):
                down.block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
            if level < len(CHANNEL_MULTIPLIERS) - 1:
                down.downsample = Downsample(block_in)
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in)
        self.norm_out = normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, LATENT_CHANNELS * 2, 3, padding=1)

    def forward(self, image):
        hidden = self.conv_in(image)
        for down in self.down[:-1]:
            for block in down.block:
                hidden = block(hidden)
            hidden = down.downsample(hidden)
        for block in self.down[-1].block:
            hidden = block(hidden)
        hidden = self.mid.block_1(hidden)
        hidden = self.mid.attn_1(hidden)
        hidden = self.mid.block_2(hidden)
        return self.conv_out(swish(self.norm_out(hidden)))


class AutoencoderKL(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = Decoder()
        self.post_quant_conv = nn.Conv2d(LATENT_CHANNELS, LATENT_CHANNELS, 1)


class PanoramaVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.quant_conv = nn.Conv2d(LATENT_CHANNELS * 2, LATENT_CHANNELS * 2, 1)
        self.post_quant_conv = nn.Conv2d(LATENT_CHANNELS, LATENT_CHANNELS, 1)

    def forward(self, image):
        mean, log_variance = self.quant_conv(self.encoder(image)).chunk(2, dim=1)
        deviation = torch.exp(0.5 * torch.clamp(log_variance, -30.0, 20.0))
        latent = mean + deviation * torch.randn_like(mean)
        return self.decoder(self.post_quant_conv(latent))
