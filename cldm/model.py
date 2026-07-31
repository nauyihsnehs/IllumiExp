import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from ldm.attention import SpatialTransformer
from ldm.diffusion import Downsample, ResBlock, TimestepEmbedSequential, UNetModel
from ldm.util import (
    conv_nd,
    instantiate_from_config,
    linear,
    make_beta_schedule,
    timestep_embedding,
    zero_module,
)


def create_model(config_path):
    config = OmegaConf.load(config_path)
    return instantiate_from_config(config.model)


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state = model.state_dict()
    filtered = {name: value for name, value in state_dict.items() if name in model_state}
    model.load_state_dict(filtered, strict=False)
    return len(filtered), len(state_dict) - len(filtered)


class DiffusionWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.diffusion_model = instantiate_from_config(config)


class ControlledUnetModel(UNetModel):
    def forward(
        self,
        x,
        timesteps=None,
        context=None,
        control=None,
        control_hdr=None,
        only_mid_control=False,
        **kwargs,
    ):
        hidden_states = []
        with torch.no_grad():
            embedding = self.time_embed(
                timestep_embedding(timesteps, self.model_channels)
            )
            hidden = x.to(self.dtype)
            for module in self.input_blocks:
                hidden = module(hidden, embedding, context)
                hidden_states.append(hidden)
            hidden = self.middle_block(hidden, embedding, context)

        hidden = hidden + control.pop()
        if control_hdr is not None:
            hidden = hidden + control_hdr.pop()

        for module in self.output_blocks:
            skip = hidden_states.pop() + control.pop()
            if control_hdr is not None:
                skip = skip + control_hdr.pop()
            hidden = module(torch.cat([hidden, skip], dim=1), embedding, context)

        return self.out(hidden.to(x.dtype))


class ControlNet(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        hint_channels,
        num_res_blocks,
        attention_resolutions,
        channel_mult,
        num_heads,
        context_dim,
        hint_channels_hdr,
        dropout=0,
        use_checkpoint=True,
        use_spatial_transformer=True,
        transformer_depth=1,
        legacy=False,
        hdr=True,
    ):
        super().__init__()
        self.model_channels = model_channels
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float32
        self.hdr = hdr
        self.control_scales = None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(2, in_channels, model_channels, 3, padding=1))]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])
        self.zero_convs_hdr = nn.ModuleList([self.make_zero_conv(model_channels)])
        self.input_hint_block = self.make_hint_block(hint_channels, model_channels)
        self.input_hint_block_hdr = self.make_hint_block(
            hint_channels_hdr, model_channels
        )

        block_counts = [num_res_blocks] * len(channel_mult)
        channels = model_channels
        downsample = 1

        for level, multiplier in enumerate(channel_mult):
            for _ in range(block_counts[level]):
                layers = [
                    ResBlock(
                        channels,
                        time_embed_dim,
                        dropout,
                        out_channels=multiplier * model_channels,
                        dims=2,
                        use_checkpoint=use_checkpoint,
                    )
                ]
                channels = multiplier * model_channels
                if downsample in attention_resolutions:
                    head_dim = channels // num_heads
                    if legacy:
                        head_dim = channels // num_heads
                    layers.append(
                        SpatialTransformer(
                            channels,
                            num_heads,
                            head_dim,
                            depth=transformer_depth,
                            context_dim=context_dim,
                            use_checkpoint=use_checkpoint,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(channels))
                self.zero_convs_hdr.append(self.make_zero_conv(channels))

            if level == len(channel_mult) - 1:
                continue
            self.input_blocks.append(
                TimestepEmbedSequential(
                    Downsample(channels, True, dims=2, out_channels=channels)
                )
            )
            self.zero_convs.append(self.make_zero_conv(channels))
            self.zero_convs_hdr.append(self.make_zero_conv(channels))
            downsample *= 2

        head_dim = channels // num_heads
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                channels,
                time_embed_dim,
                dropout,
                dims=2,
                use_checkpoint=use_checkpoint,
            ),
            SpatialTransformer(
                channels,
                num_heads,
                head_dim,
                depth=transformer_depth,
                context_dim=context_dim,
                use_checkpoint=use_checkpoint,
            ),
            ResBlock(
                channels,
                time_embed_dim,
                dropout,
                dims=2,
                use_checkpoint=use_checkpoint,
            ),
        )
        self.middle_block_out = self.make_zero_conv(channels)
        self.middle_block_out_hdr = self.make_zero_conv(channels)

    def make_hint_block(self, input_channels, output_channels):
        return TimestepEmbedSequential(
            conv_nd(2, input_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(2, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(2, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(2, 256, output_channels, 3, padding=1)),
        )

    def make_zero_conv(self, channels):
        layer = zero_module(conv_nd(2, channels, channels, 1))
        return TimestepEmbedSequential(layer)

    def forward(self, x, hint, timesteps, context, hint_hdr):
        embedding = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        guided_hint = self.input_hint_block(hint, embedding, context)
        guided_hint_hdr = self.input_hint_block_hdr(hint_hdr, embedding, context)
        outputs = []
        hdr_outputs = []
        hidden = x.to(self.dtype)
        hidden_hdr = hidden.clone()

        for module, zero_conv, zero_conv_hdr in zip(
            self.input_blocks,
            self.zero_convs,
            self.zero_convs_hdr,
        ):
            hidden = module(hidden, embedding, context)
            hidden_hdr = module(hidden_hdr, embedding, context)
            if guided_hint is not None:
                hidden = hidden + guided_hint
                guided_hint = None
            if guided_hint_hdr is not None:
                hidden_hdr = hidden_hdr + guided_hint_hdr
                guided_hint_hdr = None
            outputs.append(zero_conv(hidden, embedding, context))
            hdr_outputs.append(zero_conv_hdr(hidden_hdr, embedding, context))

        hidden = self.middle_block(hidden, embedding, context)
        hidden_hdr = self.middle_block(hidden_hdr, embedding, context)
        outputs.append(self.middle_block_out(hidden, embedding, context))
        hdr_outputs.append(self.middle_block_out_hdr(hidden_hdr, embedding, context))
        return outputs, hdr_outputs


class ControlLDM(nn.Module):
    def __init__(
        self,
        control_stage_config,
        unet_config,
        first_stage_config,
        cond_stage_config,
        timesteps=1000,
        linear_start=0.00085,
        linear_end=0.012,
        scale_factor=0.18215,
        parameterization="eps",
        only_mid_control=False,
    ):
        super().__init__()
        self.parameterization = parameterization
        self.scale_factor = scale_factor
        self.only_mid_control = only_mid_control
        self.num_timesteps = timesteps
        self.model = DiffusionWrapper(unet_config)
        self.control_model = instantiate_from_config(control_stage_config)
        self.first_stage_model = instantiate_from_config(first_stage_config)
        self.cond_stage_model = instantiate_from_config(cond_stage_config)
        self.control_scales = [1.0] * 13

        betas = make_beta_schedule(timesteps, linear_start, linear_end)
        alphas_cumprod = np.cumprod(1.0 - betas)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        self.register_buffer("betas", torch.tensor(betas, dtype=torch.float32))
        self.register_buffer(
            "alphas_cumprod", torch.tensor(alphas_cumprod, dtype=torch.float32)
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
            x=noisy,
            hint=torch.cat(condition["c_concat"], dim=1),
            timesteps=timesteps,
            context=context,
            hint_hdr=torch.cat(condition["c_hdr"], dim=1),
        )
        control = [
            value * scale for value, scale in zip(control, self.control_scales)
        ]
        control_hdr = [
            value * scale
            for value, scale in zip(control_hdr, self.control_scales)
        ]
        return self.model.diffusion_model(
            x=noisy,
            timesteps=timesteps,
            context=context,
            control=control,
            control_hdr=control_hdr,
            only_mid_control=self.only_mid_control,
        )
