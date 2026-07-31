import argparse
import os
import pathlib

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2 as cv
import numpy as np
import torch
from PIL import Image

from cdf import cdf_to_hdr, load_quantile_cdf
from vae import PanoramaVAE, load_checkpoint

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = SCRIPT_DIR / "ckpts/vae-epoch=19-step=31320.ckpt"
DEFAULT_CDF = SCRIPT_DIR / "cdf_quantile.npz"

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
PANORAMA_SIZE = (512, 256)
EXR_SAVE_PARAMS = [48, 1, 49, 4]
LUMINANCE_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="IllumiExp VAE HDR panorama inference")
    parser.add_argument("--input", type=pathlib.Path, default="test_images/inputs-vae")
    parser.add_argument(
        "--output", type=pathlib.Path, default="test_images/outputs-vae"
    )
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cdf", type=pathlib.Path, default=DEFAULT_CDF)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=114514)
    return parser.parse_args()


def collect_inputs(path):
    path = path.expanduser().resolve()
    if path.is_file():
        return [path]
    return sorted(
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )


def resolve_outputs(inputs, output):
    output = output.expanduser().resolve()
    if output.suffix.lower() == ".exr" and not output.is_dir():
        return [output]
    return [output / f"{item.stem}.exr" for item in inputs]


def read_panorama(path):
    image = np.asarray(Image.open(path).convert("RGB"))
    image = cv.resize(image, PANORAMA_SIZE, interpolation=cv.INTER_AREA)
    return np.ascontiguousarray(image, dtype=np.float32) / 255.0


def write_exr(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = np.ascontiguousarray(rgb[..., ::-1], dtype=np.float32)
    cv.imwrite(str(path), bgr, EXR_SAVE_PARAMS)


def seed_random(seed):
    if seed < 0:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_brightness(hdr):
    luminance = np.sum(hdr * LUMINANCE_WEIGHTS, axis=-1)
    percentile = float(np.percentile(luminance, 90))
    scale = 1.0 / max(percentile, np.finfo(np.float32).eps)
    print(f"Brightness P90: output={percentile:.6g}, scale={scale:.6g}")
    return hdr * scale


def run_inference(model, image, cdf, device):
    tensor = torch.from_numpy(image).permute(2, 0, 1)[None].to(device)
    decoded = model(tensor * 2.0 - 1.0)[0].permute(1, 2, 0).cpu().numpy()
    hdr = cdf_to_hdr(decoded, *cdf)
    return normalize_brightness(hdr)


def main():
    args = parse_args()
    inputs = collect_inputs(args.input)
    outputs = resolve_outputs(inputs, args.output)
    device = torch.device(args.device)
    cdf = load_quantile_cdf(args.cdf)
    seed_random(args.seed)

    model = PanoramaVAE()
    loaded, ignored = load_checkpoint(model, args.checkpoint)
    print(f"Loaded {loaded} checkpoint tensors; ignored {ignored} archived tensors")
    model.to(device).eval()

    with torch.inference_mode():
        for input_path, output_path in zip(inputs, outputs):
            result = run_inference(model, read_panorama(input_path), cdf, device)
            write_exr(output_path, result)
            print(
                f"{output_path}: max={result.max():.6g}, "
                f"mean={result.mean():.6g}, min={result.min():.6g}"
            )


if __name__ == "__main__":
    main()
