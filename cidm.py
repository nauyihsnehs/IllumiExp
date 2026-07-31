import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionConfig, CLIPVisionModel

from vae import AutoencoderKL, load_checkpoint


__all__ = ["DDIMSampler", "create_model", "load_checkpoint"]


MODEL_CHANNELS = 320
CHANNEL_MULTIPLIERS = (1, 2, 4, 4)
RESIDUAL_BLOCKS = 2
ATTENTION_RESOLUTIONS = {1, 2, 4}
ATTENTION_HEADS = 8
CONTEXT_DIM = 768
TIME_EMBED_DIM = MODEL_CHANNELS * 4
TIMESTEPS = 1000


def zero_module(module):
    for parameter in module.parameters():
        parameter.detach().zero_()
    return module


class GroupNorm32(nn.GroupNorm):
    def forward(self, value):
        return super().forward(value.float()).to(value.dtype)


def normalization(channels):
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps):
    half = MODEL_CHANNELS // 2
    frequencies = torch.exp(
        -math.log(10000)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    arguments = timesteps[:, None].float() * frequencies[None]
    return torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=-1)


class GEGLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim * 2)

    def forward(self, value):
        value, gate = self.proj(value).chunk(2, dim=-1)
        return value * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inner_dim = dim * 4
        self.net = nn.Sequential(
            GEGLU(dim, inner_dim),
            nn.Dropout(0.0),
            nn.Linear(inner_dim, dim),
        )

    def forward(self, value):
        return self.net(value)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=ATTENTION_HEADS):
        super().__init__()
        dim_head = query_dim // heads
        inner_dim = dim_head * heads
        self.scale = dim_head**-0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim or query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim or query_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(0.0))

    def forward(self, value, context=None):
        heads = self.heads
        context = value if context is None else context
        query = self.to_q(value)
        key = self.to_k(context)
        projected = self.to_v(context)
        query, key, projected = (
            tensor.reshape(tensor.shape[0], tensor.shape[1], heads, -1)
            .permute(0, 2, 1, 3)
            .reshape(tensor.shape[0] * heads, tensor.shape[1], -1)
            for tensor in (query, key, projected)
        )
        with torch.autocast(enabled=False, device_type=value.device.type):
            similarity = torch.einsum(
                "bid,bjd->bij",
                query.float(),
                key.float(),
            ) * self.scale
        weights = similarity.softmax(dim=-1)
        output = torch.einsum("bij,bjd->bid", weights, projected)
        output = output.reshape(value.shape[0], heads, value.shape[1], -1)
        output = output.permute(0, 2, 1, 3).reshape(value.shape[0], value.shape[1], -1)
        return self.to_out(output)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, context_dim):
        super().__init__()
        self.attn1 = CrossAttention(dim)
        self.ff = FeedForward(dim)
        self.attn2 = CrossAttention(dim, context_dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, value, context):
        value = self.attn1(self.norm1(value)) + value
        value = self.attn2(self.norm2(value), context) + value
        return self.ff(self.norm3(value)) + value


class SpatialTransformer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels, eps=1e-6, affine=True)
        self.proj_in = nn.Conv2d(channels, channels, 1)
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(channels, CONTEXT_DIM)]
        )
        self.proj_out = zero_module(nn.Conv2d(channels, channels, 1))

    def forward(self, value, context):
        residual = value
        batch, channels, height, width = value.shape
        value = self.proj_in(self.norm(value))
        value = value.reshape(batch, channels, height * width).permute(0, 2, 1)
        value = self.transformer_blocks[0](value, context)
        value = value.permute(0, 2, 1).reshape(batch, channels, height, width)
        return residual + self.proj_out(value)


class TimestepBlock(nn.Module):
    pass


class TimestepEmbedSequential(nn.Sequential):
    def forward(self, value, embedding, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                value = layer(value, embedding)
            elif isinstance(layer, SpatialTransformer):
                value = layer(value, context)
            else:
                value = layer(value)
        return value


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, value):
        return self.conv(F.interpolate(value, scale_factor=2, mode="nearest"))


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, value):
        return self.op(value)


class ResBlock(TimestepBlock):
    def __init__(self, channels, out_channels=None):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(TIME_EMBED_DIM, self.out_channels),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(0.0),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )
        self.skip_connection = (
            nn.Identity()
            if self.out_channels == channels
            else nn.Conv2d(channels, self.out_channels, 1)
        )

    def forward(self, value, embedding):
        hidden = self.in_layers(value)
        embedded = self.emb_layers(embedding).to(hidden.dtype)
        while embedded.ndim < hidden.ndim:
            embedded = embedded[..., None]
        hidden = self.out_layers(hidden + embedded)
        return self.skip_connection(value) + hidden


class UNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_channels = MODEL_CHANNELS
        self.dtype = torch.float32
        self.time_embed = nn.Sequential(
            nn.Linear(MODEL_CHANNELS, TIME_EMBED_DIM),
            nn.SiLU(),
            nn.Linear(TIME_EMBED_DIM, TIME_EMBED_DIM),
        )
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(nn.Conv2d(4, MODEL_CHANNELS, 3, padding=1))]
        )
        input_channels = [MODEL_CHANNELS]
        channels = MODEL_CHANNELS
        downsample = 1

        for level, multiplier in enumerate(CHANNEL_MULTIPLIERS):
            for _ in range(RESIDUAL_BLOCKS):
                output_channels = multiplier * MODEL_CHANNELS
                layers = [ResBlock(channels, output_channels)]
                channels = output_channels
                if downsample in ATTENTION_RESOLUTIONS:
                    layers.append(SpatialTransformer(channels))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_channels.append(channels)
            if level < len(CHANNEL_MULTIPLIERS) - 1:
                self.input_blocks.append(TimestepEmbedSequential(Downsample(channels)))
                input_channels.append(channels)
                downsample *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(channels),
            SpatialTransformer(channels),
            ResBlock(channels),
        )
        self.output_blocks = nn.ModuleList()
        for level, multiplier in reversed(list(enumerate(CHANNEL_MULTIPLIERS))):
            for block_index in range(RESIDUAL_BLOCKS + 1):
                skip_channels = input_channels.pop()
                output_channels = MODEL_CHANNELS * multiplier
                layers = [ResBlock(channels + skip_channels, output_channels)]
                channels = output_channels
                if downsample in ATTENTION_RESOLUTIONS:
                    layers.append(SpatialTransformer(channels))
                if level and block_index == RESIDUAL_BLOCKS:
                    layers.append(Upsample(channels))
                    downsample //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            zero_module(nn.Conv2d(MODEL_CHANNELS, 4, 3, padding=1)),
        )


class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps, context, control, control_hdr):
        hidden_states = []
        with torch.no_grad():
            embedding = self.time_embed(timestep_embedding(timesteps))
            hidden = x.to(self.dtype)
            for module in self.input_blocks:
                hidden = module(hidden, embedding, context)
                hidden_states.append(hidden)
            hidden = self.middle_block(hidden, embedding, context)

        hidden = hidden + control.pop() + control_hdr.pop()
        for module in self.output_blocks:
            skip = hidden_states.pop() + control.pop() + control_hdr.pop()
            hidden = module(torch.cat([hidden, skip], dim=1), embedding, context)
        return self.out(hidden.to(x.dtype))


def hint_block(input_channels):
    return TimestepEmbedSequential(
        nn.Conv2d(input_channels, 16, 3, padding=1),
        nn.SiLU(),
        nn.Conv2d(16, 16, 3, padding=1),
        nn.SiLU(),
        nn.Conv2d(16, 32, 3, stride=2, padding=1),
        nn.SiLU(),
        nn.Conv2d(32, 32, 3, padding=1),
        nn.SiLU(),
        nn.Conv2d(32, 96, 3, stride=2, padding=1),
        nn.SiLU(),
        nn.Conv2d(96, 96, 3, padding=1),
        nn.SiLU(),
        nn.Conv2d(96, 256, 3, stride=2, padding=1),
        nn.SiLU(),
        zero_module(nn.Conv2d(256, MODEL_CHANNELS, 3, padding=1)),
    )


def zero_conv(channels):
    return TimestepEmbedSequential(zero_module(nn.Conv2d(channels, channels, 1)))


class ControlNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_channels = MODEL_CHANNELS
        self.dtype = torch.float32
        self.time_embed = nn.Sequential(
            nn.Linear(MODEL_CHANNELS, TIME_EMBED_DIM),
            nn.SiLU(),
            nn.Linear(TIME_EMBED_DIM, TIME_EMBED_DIM),
        )
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(nn.Conv2d(4, MODEL_CHANNELS, 3, padding=1))]
        )
        self.zero_convs = nn.ModuleList([zero_conv(MODEL_CHANNELS)])
        self.zero_convs_hdr = nn.ModuleList([zero_conv(MODEL_CHANNELS)])
        self.input_hint_block = hint_block(3)
        self.input_hint_block_hdr = hint_block(3)
        channels = MODEL_CHANNELS
        downsample = 1

        for level, multiplier in enumerate(CHANNEL_MULTIPLIERS):
            for _ in range(RESIDUAL_BLOCKS):
                output_channels = multiplier * MODEL_CHANNELS
                layers = [ResBlock(channels, output_channels)]
                channels = output_channels
                if downsample in ATTENTION_RESOLUTIONS:
                    layers.append(SpatialTransformer(channels))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(zero_conv(channels))
                self.zero_convs_hdr.append(zero_conv(channels))
            if level < len(CHANNEL_MULTIPLIERS) - 1:
                self.input_blocks.append(TimestepEmbedSequential(Downsample(channels)))
                self.zero_convs.append(zero_conv(channels))
                self.zero_convs_hdr.append(zero_conv(channels))
                downsample *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(channels),
            SpatialTransformer(channels),
            ResBlock(channels),
        )
        self.middle_block_out = zero_conv(channels)
        self.middle_block_out_hdr = zero_conv(channels)

    def forward(self, x, hint, timesteps, context, hint_hdr):
        embedding = self.time_embed(timestep_embedding(timesteps))
        hidden = x.to(self.dtype)
        hidden_hdr = hidden.clone()
        hidden = self.input_blocks[0](hidden, embedding, context)
        hidden = hidden + self.input_hint_block(hint, embedding, context)
        hidden_hdr = self.input_blocks[0](hidden_hdr, embedding, context)
        hidden_hdr = hidden_hdr + self.input_hint_block_hdr(hint_hdr, embedding, context)
        outputs = [self.zero_convs[0](hidden, embedding, context)]
        hdr_outputs = [self.zero_convs_hdr[0](hidden_hdr, embedding, context)]

        modules = zip(
            self.input_blocks[1:],
            self.zero_convs[1:],
            self.zero_convs_hdr[1:],
        )
        for module, control, control_hdr in modules:
            hidden = module(hidden, embedding, context)
            hidden_hdr = module(hidden_hdr, embedding, context)
            outputs.append(control(hidden, embedding, context))
            hdr_outputs.append(control_hdr(hidden_hdr, embedding, context))

        hidden = self.middle_block(hidden, embedding, context)
        hidden_hdr = self.middle_block(hidden_hdr, embedding, context)
        outputs.append(self.middle_block_out(hidden, embedding, context))
        hdr_outputs.append(self.middle_block_out_hdr(hidden_hdr, embedding, context))
        return outputs, hdr_outputs


class CLIPVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = CLIPVisionModel(CLIPVisionConfig())

    def forward(self, images):
        return self.vision_model(pixel_values=images)


class FrozenCLIPEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = CLIPVision()
        self.vision.eval()
        for parameter in self.vision.parameters():
            parameter.requires_grad = False

    def forward(self, images):
        return self.vision(images).last_hidden_state

    def encode(self, images):
        return self(images)


class ControlLDM(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 0.18215
        self.num_timesteps = TIMESTEPS
        self.model = nn.Module()
        self.model.diffusion_model = ControlledUnetModel()
        self.control_model = ControlNet()
        self.first_stage_model = AutoencoderKL()
        self.cond_stage_model = FrozenCLIPEmbedder()
        self.control_scales = [1.0] * 13

        betas = torch.linspace(
            0.00085**0.5,
            0.012**0.5,
            TIMESTEPS,
            dtype=torch.float64,
        ).square().numpy()
        alphas_cumprod = np.cumprod(1.0 - betas)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        self.register_buffer("betas", torch.tensor(betas, dtype=torch.float32))
        self.register_buffer(
            "alphas_cumprod",
            torch.tensor(alphas_cumprod, dtype=torch.float32),
        )
        self.register_buffer(
            "alphas_cumprod_prev",
            torch.tensor(alphas_cumprod_prev, dtype=torch.float32),
        )

    def get_learned_conditioning(self, images):
        return self.cond_stage_model.encode(images)

    def decode_first_stage(self, latent):
        return self.first_stage_model.decode(latent / self.scale_factor)

    def apply_model(self, noisy, timesteps, condition):
        context = torch.cat(condition["c_crossattn"], dim=1)
        control, control_hdr = self.control_model(
            noisy,
            torch.cat(condition["c_concat"], dim=1),
            timesteps,
            context,
            torch.cat(condition["c_hdr"], dim=1),
        )
        control = [value * scale for value, scale in zip(control, self.control_scales)]
        control_hdr = [
            value * scale for value, scale in zip(control_hdr, self.control_scales)
        ]
        return self.model.diffusion_model(
            noisy,
            timesteps,
            context,
            control,
            control_hdr,
        )


class DDIMSampler:
    def __init__(self, model):
        self.model = model

    def make_schedule(self, steps, eta):
        stride = self.model.num_timesteps // steps
        self.timesteps = np.arange(0, self.model.num_timesteps, stride)[:steps] + 1
        alphas_cumprod = self.model.alphas_cumprod.detach().cpu().numpy()
        alphas = alphas_cumprod[self.timesteps]
        alphas_prev = np.asarray(
            [alphas_cumprod[0], *alphas_cumprod[self.timesteps[:-1]].tolist()]
        )
        sigmas = eta * np.sqrt(
            (1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev)
        )
        device = self.model.betas.device
        self.ddim_sigmas = torch.tensor(sigmas, dtype=torch.float32, device=device)
        self.ddim_alphas = torch.tensor(alphas, dtype=torch.float32, device=device)
        self.ddim_alphas_prev = torch.tensor(
            alphas_prev,
            dtype=torch.float32,
            device=device,
        )
        self.ddim_sqrt_one_minus_alphas = torch.sqrt(1.0 - self.ddim_alphas)

    @torch.inference_mode()
    def sample(self, steps, shape, conditioning, eta=0.0):
        self.make_schedule(steps, eta)
        device = self.model.betas.device
        latent = torch.randn(shape, device=device)
        for index, step in reversed(list(enumerate(self.timesteps))):
            timestep = torch.full(
                (shape[0],),
                step,
                device=device,
                dtype=torch.long,
            )
            latent = self.sample_step(latent, conditioning, timestep, index)
        return latent

    def sample_step(self, latent, conditioning, timestep, index):
        predicted_noise = self.model.apply_model(latent, timestep, conditioning)
        tensor_shape = (latent.shape[0], 1, 1, 1)
        alpha = torch.full(
            tensor_shape,
            self.ddim_alphas[index],
            device=latent.device,
        )
        alpha_prev = torch.full(
            tensor_shape,
            self.ddim_alphas_prev[index],
            device=latent.device,
        )
        sigma = torch.full(
            tensor_shape,
            self.ddim_sigmas[index],
            device=latent.device,
        )
        sqrt_one_minus_alpha = torch.full(
            tensor_shape,
            self.ddim_sqrt_one_minus_alphas[index],
            device=latent.device,
        )
        predicted_start = (
            latent - sqrt_one_minus_alpha * predicted_noise
        ) / alpha.sqrt()
        direction = torch.sqrt(1.0 - alpha_prev - sigma.square()) * predicted_noise
        noise = sigma * torch.randn(latent.shape, device=latent.device)
        return alpha_prev.sqrt() * predicted_start + direction + noise


def create_model():
    return ControlLDM()
