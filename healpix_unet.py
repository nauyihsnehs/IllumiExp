import pathlib

import cv2 as cv
import healpy as hp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_base


DEFAULT_RESOLUTION = 224
DEFAULT_PANO_SIZE = (512, 256)
DEFAULT_CHECKPOINT = pathlib.Path(
    "ckpts/hpunet_n32_epoch=199-val_loss=0.07196.ckpt"
    # "ckpts/hpunet_n16_epoch=99-val_loss=0.04170.ckpt"
)

_CACHED_MODEL = None
_CACHED_KEY = None


class SphericalConv(nn.Module):
    def __init__(self, nside, in_channels, out_channels):
        super().__init__()
        pixel_count = hp.nside2npix(nside)
        neighbours = np.empty(9 * pixel_count, dtype=np.int64)
        for index in range(pixel_count):
            local = hp.pixelfunc.get_all_neighbours(nside, index)
            local = np.insert(local, 4, index)
            local[local == -1] = index
            neighbours[index * 9 : index * 9 + 9] = local
        self.register_buffer("neighbours", torch.from_numpy(neighbours))
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=9, stride=9)

    def forward(self, value):
        value = F.pad(value, (0, 1), mode="constant", value=0.0)
        return self.conv(value[:, :, self.neighbours].contiguous())


class SphericalDown(nn.Module):
    def forward(self, value):
        return value[:, :, ::4].contiguous()


class SphericalUp(nn.Module):
    def forward(self, value):
        return torch.repeat_interleave(value, 4, dim=-1)


class SphericalConvBlock(nn.Module):
    def __init__(self, nside, in_channels, out_channels):
        super().__init__()
        self.conv = SphericalConv(nside, in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, value):
        return self.activate(self.bn(self.conv(value)))


class SphericalUNet(nn.Module):
    def __init__(self, input_pixels, depths):
        super().__init__()
        nside_levels = []
        pixels = input_pixels
        for _ in depths:
            nside_levels.append(max(int(np.sqrt(pixels / 12)), 1))
            pixels //= 4

        self.down_convs = nn.ModuleList()
        self.down_samplers = nn.ModuleList()
        in_channels = 3
        for index, (nside, channels) in enumerate(zip(nside_levels, depths)):
            self.down_convs.append(
                nn.Sequential(
                    SphericalConvBlock(nside, in_channels, channels),
                    SphericalConvBlock(nside, channels, channels),
                )
            )
            if index < len(depths) - 1:
                self.down_samplers.append(SphericalDown())
            in_channels = channels

        self.up_samplers = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for index in range(len(depths) - 1, 0, -1):
            self.up_samplers.append(SphericalUp())
            self.up_convs.append(
                nn.Sequential(
                    SphericalConvBlock(
                        nside_levels[index - 1],
                        depths[index] + depths[index - 1],
                        depths[index - 1],
                    ),
                    SphericalConvBlock(
                        nside_levels[index - 1],
                        depths[index - 1],
                        depths[index - 1],
                    ),
                )
            )
        self.final_conv = SphericalConv(nside_levels[0], depths[0], 3)

    def forward(self, value):
        skips = []
        for index, down_conv in enumerate(self.down_convs):
            value = down_conv(value)
            if index < len(self.down_samplers):
                skips.append(value)
                value = self.down_samplers[index](value)

        for index, (upsample, up_conv) in enumerate(
            zip(self.up_samplers, self.up_convs)
        ):
            value = upsample(value)
            skip = skips[-index - 1]
            value = value[:, :, : skip.shape[2]]
            if value.shape[2] < skip.shape[2]:
                value = F.pad(value, (0, skip.shape[2] - value.shape[2]))
            value = up_conv(torch.cat((value, skip), dim=1))
        return self.final_conv(value)


class BackboneModel(nn.Module):
    def __init__(self, nside):
        super().__init__()
        self.nside = nside
        self.backbone = convnext_base(weights=None).features
        kernel, stride = {32: (2, 1), 16: (3, 2), 8: (3, 2)}[nside]
        self.maxpool = nn.MaxPool2d(kernel_size=kernel, stride=stride)

    def forward(self, value):
        value = self.backbone(value)
        if self.nside == 8:
            value = value.view(-1, 256, 4, 7, 7).mean(2)
        return torch.flatten(self.maxpool(value), 1)


class HPUNet(nn.Module):
    def __init__(self, nside=16):
        super().__init__()
        self.nside = nside
        self.pixel_count = hp.nside2npix(nside)
        self.backbone = BackboneModel(nside)
        depths = [64, 128, 256, 512, 1024] if nside >= 16 else [32, 64, 128, 256]
        self.unet = SphericalUNet(self.pixel_count, depths)
        self.activate = nn.Softplus()

    def forward(self, value):
        value = self.backbone(value)
        value = self.unet(value.view(-1, 3, self.pixel_count))
        return self.activate(value).permute(0, 2, 1)


def load_model(checkpoint_path, device):
    global _CACHED_MODEL, _CACHED_KEY
    path = pathlib.Path(checkpoint_path).expanduser().resolve()
    key = (path, str(device))
    if _CACHED_MODEL is not None and _CACHED_KEY == key:
        return _CACHED_MODEL

    checkpoint = torch.load(path, map_location=device)
    nside = int(checkpoint.get("hyper_parameters", {}).get("nside", 16))
    model = HPUNet(nside)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)

    model.to(device).eval()
    _CACHED_MODEL = model
    _CACHED_KEY = key
    return model


def unload_model():
    global _CACHED_MODEL, _CACHED_KEY
    _CACHED_MODEL = None
    _CACHED_KEY = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prepare_image(image, device):
    image = np.asarray(image)
    image = image[..., :3].astype(np.float32)
    if image.max() > 1.0:
        image /= 255.0
    image = image[..., ::-1]
    image = cv.resize(image, (DEFAULT_RESOLUTION, DEFAULT_RESOLUTION))
    tensor = torch.from_numpy(np.ascontiguousarray(image))
    return tensor.permute(2, 0, 1)[None].to(device) * 2.0 - 1.0


def healpix_to_panorama(healpix, nside, size=DEFAULT_PANO_SIZE):
    width, height = size
    theta = np.linspace(0, np.pi, height)
    phi = np.linspace(-np.pi, np.pi, width)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    indices = hp.ang2pix(nside, theta_grid, phi_grid)
    return np.ascontiguousarray(healpix[indices], dtype=np.float32)


def predict(image, checkpoint_path, device):
    model = load_model(checkpoint_path, device)
    tensor = prepare_image(image, device)
    with torch.inference_mode():
        healpix = torch.expm1(model(tensor)[0]).cpu().numpy()
    return healpix_to_panorama(healpix, model.nside)
