import argparse
import os
import pathlib

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2 as cv
import numpy as np
import torch
from PIL import Image

import healpix_unet
from cldm.ddim import DDIMSampler
from cldm.model import create_model, load_checkpoint
from cdf import cdf_to_hdr, hdr_to_cdf, load_quantile_cdf
from pano_tools import pers2pano

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "model.yaml"
DEFAULT_CHECKPOINT = SCRIPT_DIR / "ckpts/v137-epoch=9-step=52200.ckpt"
DEFAULT_HPUNET_CHECKPOINT = SCRIPT_DIR / healpix_unet.DEFAULT_CHECKPOINT
DEFAULT_CDF = SCRIPT_DIR / "cdf_quantile.npz"

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
PANORAMA_SIZE = (512, 256)
EXR_SAVE_PARAMS = [48, 1, 49, 4]
LUMINANCE_WEIGHTS = np.array([0.2627, 0.6780, 0.0593], dtype=np.float32)
BRIGHTNESS_PERCENTILE = 90


def parse_args():
    parser = argparse.ArgumentParser(description="IllumiExp HDR panorama inference")
    parser.add_argument("--input", type=pathlib.Path, default="test_images/inputs")
    parser.add_argument("--output", type=pathlib.Path, default="test_images/outputs")
    parser.add_argument("--healpix", type=pathlib.Path, default="test_images/healpix")
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cdf", type=pathlib.Path, default=DEFAULT_CDF)
    parser.add_argument(
        "--healpix-unet-checkpoint",
        type=pathlib.Path,
        default=DEFAULT_HPUNET_CHECKPOINT,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=114514)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--fov", type=float, default=90.0)
    return parser.parse_args()


def collect_inputs(path):
    path = path.expanduser().resolve()
    if path.is_file():
        return [path]
    inputs = sorted(
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )
    return inputs


def resolve_outputs(inputs, output):
    output = output.expanduser().resolve()
    output_is_file = output.suffix.lower() == ".exr" and not output.is_dir()
    if output_is_file:
        return [output], output.parent
    return [output / f"{item.stem}.exr" for item in inputs], output


def resolve_healpix_paths(inputs, healpix, output_root):
    if healpix is None:
        cache_root = output_root / "healpix"
        return [cache_root / f"{item.stem}_hp.exr" for item in inputs]

    healpix = healpix.expanduser().resolve()
    is_file = healpix.suffix.lower() == ".exr" and not healpix.is_dir()
    if is_file:
        return [healpix]
    return [healpix / f"{item.stem}_hp.exr" for item in inputs]


def resolve_c_concat_paths(outputs, healpix, output_root):
    cache_root = output_root / "c_concat"
    if healpix is not None and healpix.suffix.lower() != ".exr":
        cache_root = healpix.expanduser().resolve().parent / "c_concat"
    return [cache_root / f"{output_path.stem}_c_concat.png" for output_path in outputs]


def resolve_device(name):
    return torch.device(name)


def read_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def read_healpix(path):
    image = cv.imread(str(path), cv.IMREAD_UNCHANGED)
    image = cv.resize(image, PANORAMA_SIZE, interpolation=cv.INTER_AREA)
    return np.ascontiguousarray(image[..., ::-1], dtype=np.float32)


def write_exr(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = np.ascontiguousarray(rgb[..., ::-1], dtype=np.float32)
    cv.imwrite(str(path), bgr, EXR_SAVE_PARAMS)


def write_c_concat(path, tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = tensor[0].permute(1, 2, 0).detach().cpu().numpy()
    rgb = np.clip((rgb + 1.0) * 127.5, 0, 255).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)
    print(path)


def prepare_healpix(inputs, paths, checkpoint, device):
    missing = [
        (input_path, healpix_path)
        for input_path, healpix_path in zip(inputs, paths)
        if not healpix_path.is_file()
    ]
    if not missing:
        return
    print(f"Generating {len(missing)} missing Healpix cache files")
    for input_path, healpix_path in missing:
        predicted = healpix_unet.predict(read_rgb(input_path), checkpoint, device)
        write_exr(healpix_path, predicted)
        print(healpix_path)
    healpix_unet.unload_model()


def seed_random(seed):
    if seed < 0:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_tensor(array, device):
    return torch.from_numpy(np.ascontiguousarray(array)).float().to(device) * 2.0 - 1.0


def match_healpix_brightness(hdr, healpix):
    output_luminance = np.sum(hdr * LUMINANCE_WEIGHTS, axis=-1)
    healpix_luminance = np.sum(healpix * LUMINANCE_WEIGHTS, axis=-1)
    output_p90 = float(np.percentile(output_luminance, BRIGHTNESS_PERCENTILE))
    healpix_p90 = float(np.percentile(healpix_luminance, BRIGHTNESS_PERCENTILE))
    scale = healpix_p90 / max(output_p90, np.finfo(np.float32).eps)
    print(
        f"Brightness P90: output={output_p90:.6g}, "
        f"healpix={healpix_p90:.6g}, scale={scale:.6g}"
    )
    return hdr * scale


def run_inference(model, sampler, image, healpix, cdf, args, device, c_concat_path):
    quantile_x, quantile_p, max_value = cdf
    input_image = np.asarray(Image.fromarray(image).resize((224, 224))) / 255.0
    panorama = pers2pano(input_image, PANORAMA_SIZE, vfov=args.fov)
    panorama = normalize_tensor(panorama, device)

    condition = normalize_tensor(input_image, device).permute(2, 0, 1)[None]
    healpix_hdr = np.clip(healpix, 0, 65535)
    healpix_encoded = hdr_to_cdf(healpix_hdr, quantile_x, quantile_p, max_value)
    healpix_encoded = (
        torch.from_numpy(np.ascontiguousarray(healpix_encoded)).float().to(device)
    )

    panorama = panorama[None].permute(0, 3, 1, 2)
    healpix_encoded = healpix_encoded[None].permute(0, 3, 1, 2)
    conditioning = {
        "c_concat": [panorama],
        "c_hdr": [healpix_encoded],
        "c_crossattn": [model.get_learned_conditioning(condition)],
    }
    write_c_concat(c_concat_path, conditioning["c_concat"][0])
    model.control_scales = [args.strength] * 13
    seed_random(args.seed)
    latent = sampler.sample(
        args.steps,
        (1, 4, PANORAMA_SIZE[1] // 8, PANORAMA_SIZE[0] // 8),
        conditioning,
        args.eta,
    )
    decoded = model.decode_first_stage(latent)[0].permute(1, 2, 0).cpu().numpy()
    hdr = cdf_to_hdr(decoded, quantile_x, quantile_p, max_value)
    return match_healpix_brightness(hdr, healpix_hdr)


def main():
    args = parse_args()
    inputs = collect_inputs(args.input)
    outputs, output_root = resolve_outputs(inputs, args.output)
    healpix_paths = resolve_healpix_paths(inputs, args.healpix, output_root)
    c_concat_paths = resolve_c_concat_paths(outputs, args.healpix, output_root)
    device = resolve_device(args.device)

    prepare_healpix(inputs, healpix_paths, args.healpix_unet_checkpoint, device)
    cdf = load_quantile_cdf(args.cdf)
    model = create_model(args.config)
    loaded, ignored = load_checkpoint(model, args.checkpoint)
    print(f"Loaded {loaded} checkpoint tensors; ignored {ignored} archived tensors")
    model.to(device).eval()
    sampler = DDIMSampler(model)

    with torch.inference_mode():
        for input_path, healpix_path, output_path, c_concat_path in zip(
            inputs,
            healpix_paths,
            outputs,
            c_concat_paths,
        ):
            image = read_rgb(input_path)
            healpix = read_healpix(healpix_path)
            result = run_inference(
                model,
                sampler,
                image,
                healpix,
                cdf,
                args,
                device,
                c_concat_path,
            )
            write_exr(output_path, result)
            print(output_path)


if __name__ == "__main__":
    main()
