import argparse
import os
import pathlib

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2 as cv
import numpy as np
import torch
from PIL import Image

import healpix_unet
from cdf import cdf_to_hdr, hdr_to_cdf, load_quantile_cdf
from cidm import ControlLDM, DDIMSampler, load_checkpoint
from pano_tools import pers2pano

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = SCRIPT_DIR / "ckpts/v137-epoch=9-step=52200.ckpt"
DEFAULT_HPUNET_CHECKPOINT = SCRIPT_DIR / healpix_unet.DEFAULT_CHECKPOINT
DEFAULT_CDF = SCRIPT_DIR / "cdf_quantile.npz"

PANORAMA_SIZE = (512, 256)
EXR_SAVE_PARAMS = [48, 1, 49, 4]
LUMINANCE_WEIGHTS = np.array([0.2627, 0.6780, 0.0593], dtype=np.float32)
BRIGHTNESS_PERCENTILE = 90


def parse_args():
    parser = argparse.ArgumentParser(description="IllumiExp HDR panorama inference")
    parser.add_argument("--input", type=pathlib.Path, default="test_images/inputs")
    parser.add_argument("--output", type=pathlib.Path, default="test_images/outputs")
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cdf", type=pathlib.Path, default=DEFAULT_CDF)
    parser.add_argument(
        "--healpix-unet-checkpoint",
        type=pathlib.Path,
        default=DEFAULT_HPUNET_CHECKPOINT,
    )
    parser.add_argument("--seed", type=int, default=114514)
    parser.add_argument("--steps", type=int, default=50)
    return parser.parse_args()


def collect_inputs(path):
    path = path.expanduser().resolve()
    if path.is_file():
        return [path]
    return sorted(item for item in path.iterdir() if item.is_file())


def resolve_outputs(inputs, output):
    output_root = output.expanduser().resolve()
    outputs = [output_root / "hdr" / f"{item.stem}.exr" for item in inputs]
    return outputs, output_root


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


def write_concat(path, tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = tensor[0].permute(1, 2, 0).detach().cpu().numpy()
    rgb = np.clip((rgb + 1.0) * 127.5, 0, 255).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)
    print(path)


def generate_healpix(inputs, paths, checkpoint, device):
    print(f"Generating {len(inputs)} Healpix files")
    for input_path, healpix_path in zip(inputs, paths):
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


def run_inference(model, sampler, image, healpix, cdf, args, device, concat_path):
    quantile_x, quantile_p, max_value = cdf
    input_image = np.asarray(Image.fromarray(image).resize((224, 224))) / 255.0
    panorama = pers2pano(input_image, PANORAMA_SIZE, vfov=90.0)
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
        "c_crossattn": [model.cond_stage_model(condition)],
    }
    write_concat(concat_path, conditioning["c_concat"][0])
    seed_random(args.seed)
    latent = sampler.sample(
        args.steps,
        (1, 4, PANORAMA_SIZE[1] // 8, PANORAMA_SIZE[0] // 8),
        conditioning,
    )
    latent = model.first_stage_model.post_quant_conv(latent / model.scale_factor)
    decoded = model.first_stage_model.decoder(latent)[0]
    decoded = decoded.permute(1, 2, 0).cpu().numpy()
    hdr = cdf_to_hdr(decoded, quantile_x, quantile_p, max_value)
    return match_healpix_brightness(hdr, healpix_hdr)


def main():
    args = parse_args()
    inputs = collect_inputs(args.input)
    outputs, output_root = resolve_outputs(inputs, args.output)
    healpix_paths = [
        output_root / "healpix" / f"{item.stem}_hp.exr" for item in inputs
    ]
    concat_paths = [
        output_root / "concat" / f"{item.stem}_c_concat.png" for item in inputs
    ]
    device = torch.device("cuda")

    generate_healpix(inputs, healpix_paths, args.healpix_unet_checkpoint, device)
    cdf = load_quantile_cdf(args.cdf)
    model = ControlLDM()
    loaded, ignored = load_checkpoint(model, args.checkpoint)
    print(f"Loaded {loaded} checkpoint tensors; ignored {ignored} archived tensors")
    model.to(device).eval()
    sampler = DDIMSampler(model)

    with torch.inference_mode():
        for input_path, healpix_path, output_path, concat_path in zip(
            inputs,
            healpix_paths,
            outputs,
            concat_paths,
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
                concat_path,
            )
            write_exr(output_path, result)
            print(output_path)


if __name__ == "__main__":
    main()
