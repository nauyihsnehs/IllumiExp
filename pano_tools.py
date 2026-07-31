import numpy as np
from scipy.ndimage import map_coordinates


def image_to_world(u, v):
    theta = np.pi * (u * 2.0 - 1.0)
    phi = np.pi * v
    x = np.sin(phi) * np.sin(theta)
    y = np.cos(phi)
    z = -np.sin(phi) * np.cos(theta)
    return x, y, z


def world_to_image(x, y, z):
    u = 0.5 * (1.0 + np.arctan2(x, -z) / np.pi)
    v = np.arccos(y) / np.pi
    return u, v


def world_coordinates(width, height):
    columns = np.linspace(0, 1, width * 2 + 1)[1::2].astype(np.float32)
    rows = np.linspace(0, 1, height * 2 + 1)[1::2].astype(np.float32)
    u, v = np.meshgrid(columns, rows)
    return image_to_world(u, v)


def resample_zero_rotation(panorama):
    height, width, channels = panorama.shape
    x, y, z = world_coordinates(width, height)
    x, y, z = [np.clip(value.astype(np.float64), -1.0, 1.0) for value in (x, y, z)]
    u, v = world_to_image(x, y, z)
    coordinates = [(v * height).ravel(), (u * width).ravel()]
    return np.stack(
        [
            map_coordinates(
                panorama[..., channel],
                coordinates,
                order=1,
                mode="nearest",
                prefilter=True,
            ).reshape(u.shape)
            for channel in range(channels)
        ],
        axis=-1,
    )


def pers2pano(perspective, panorama_size, vfov=90):
    panorama_width, panorama_height = panorama_size
    source_height, source_width, channels = perspective.shape
    aspect_ratio = max(source_height, source_width) / min(
        source_height,
        source_width,
    )
    horizontal_fov = (
        2.0
        * np.arctan(np.tan(vfov * np.pi / 180.0 / 2.0) * aspect_ratio)
        * 180.0
        / np.pi
    )
    horizontal = np.tan(horizontal_fov / 2.0 * np.pi / 180.0)
    vertical = np.tan(vfov / 2.0 * np.pi / 180.0)

    projected_y = np.linspace(vertical, -vertical, source_height)
    projected_x = np.linspace(-horizontal, horizontal, source_width)
    x, y = np.meshgrid(projected_x, projected_y)
    radius = np.sqrt(
        (np.square(x) + np.square(y)) / (np.square(x) + np.square(y) + 1.0)
    )
    theta = np.arctan2(x, y)
    x = radius * np.sin(theta)
    y = radius * np.cos(theta)
    z = -np.sqrt(1.0 - np.square(x) - np.square(y))
    u, v = world_to_image(x, y, z)
    columns = np.clip(
        (u * panorama_width).astype(int),
        0,
        panorama_width - 1,
    )
    rows = np.clip(
        (v * panorama_height).astype(int),
        0,
        panorama_height - 1,
    )

    panorama = np.zeros(
        (panorama_height, panorama_width, channels),
        dtype=perspective.dtype,
    )
    panorama[rows.ravel(), columns.ravel()] = perspective.reshape(-1, channels)
    return resample_zero_rotation(panorama)
