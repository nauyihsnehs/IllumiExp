import importlib
import math

import numpy as np
import torch
import torch.nn as nn


def exists(value):
    return value is not None


def instantiate_from_config(config):
    target = config["target"]
    module_name, object_name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)(**config.get("params", {}))


def make_beta_schedule(n_timestep, linear_start, linear_end):
    start = linear_start**0.5
    end = linear_end**0.5
    return torch.linspace(start, end, n_timestep, dtype=torch.float64).square().numpy()


def make_ddim_timesteps(num_ddim_timesteps, num_ddpm_timesteps):
    stride = num_ddpm_timesteps // num_ddim_timesteps
    steps = np.arange(0, num_ddpm_timesteps, stride)
    return steps[:num_ddim_timesteps] + 1


def make_ddim_sampling_parameters(alphas_cumprod, timesteps, eta):
    alphas = alphas_cumprod[timesteps]
    alphas_prev = np.asarray(
        [alphas_cumprod[0], *alphas_cumprod[timesteps[:-1]].tolist()]
    )
    sigmas = eta * np.sqrt(
        (1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev)
    )
    return sigmas, alphas, alphas_prev


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    if repeat_only:
        return timesteps[:, None].repeat(1, dim)
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * frequencies[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(function, inputs, parameters, enabled):
    return function(*inputs)


def zero_module(module):
    for parameter in module.parameters():
        parameter.detach().zero_()
    return module


def normalization(channels):
    return GroupNorm32(32, channels)


class GroupNorm32(nn.GroupNorm):
    def forward(self, value):
        return super().forward(value.float()).to(value.dtype)


def conv_nd(dims, *args, **kwargs):
    layers = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
    return layers[dims](*args, **kwargs)


def linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    layers = {1: nn.AvgPool1d, 2: nn.AvgPool2d, 3: nn.AvgPool3d}
    return layers[dims](*args, **kwargs)


def noise_like(shape, device, repeat=False):
    if repeat:
        return torch.randn((1, *shape[1:]), device=device).repeat(
            shape[0], *((1,) * (len(shape) - 1))
        )
    return torch.randn(shape, device=device)
