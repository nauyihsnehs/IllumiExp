# IllumiExp: All-Frequency Illumination Estimation via HEALPix-Guided Diffusion, TIP 2026.

[Zhongyun Bao<sup>†</sup>](https://www.researchgate.net/profile/Zhongyun-Bao), [Shiyuan Shen<sup>†</sup>](https://nauyihsnehs.github.io/), Xiangqian Shen, [Chao Liang](https://scholar.google.com/citations?user=JQpmKD0AAAAJ), [Chunxia Xiao](https://graphvision.whu.edu.cn/). <sup>†</sup>_equal contribution_

**[📘Paper](https://doi.org/10.1109/TIP.2026.3718440)** · **[🤗Demo](https://huggingface.co/spaces/shenshiyuan/IllumiExp)**

<p>
  <img src="test_images/teaser.svg" alt="IllumiExp teaser" width="40%">
</p>

## Repository structure

```text
IllumiExp/
├── ckpts/                 # Pretrained checkpoints
├── cdf.py                 # HDR/CDF transforms
├── cdf_quantile.npz       # CDF statistics
├── cidm.py                # Conditional illumination diffusion model and sampler
├── healpix_unet.py        # HEALPix illumination predictor
├── inference.py           # Full single-image-to-HDR inference pipeline
├── inference_vae.py       # Panorama inverse-tonemapping inference pipeline
├── pano_tools.py          # Perspective-to-panorama projection utilities
├── vae.py                 # Panorama VAE
└── requirements.txt       # Python dependencies
```

## Environment

Healpy does not currently support [Windows](https://github.com/healpy/healpy/issues/25).

```bash
conda create -n illumiexp python=3.11 -y
conda activate illumiexp
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Checkpoints

Download the pretrained checkpoints from [OneDrive](https://1drv.ms/f/c/98332f857c4e88d7/IgAF-PGKjqbVSogiG0_naB9KAZx5aOzSMIxp1_x8SVsh3ZU?e=3Fs9O5) or [🤗Buckets](https://huggingface.co/buckets/shenshiyuan/illumiexp-ckpts).

Place the pretrained checkpoints in `./ckpts/` with the following default names:

```text
ckpts/
├── v137-epoch=9-step=52200.ckpt                # Main diffusion checkpoint
├── hpunet_n32_epoch=199-val_loss=0.07196.ckpt  # HEALPix U-Net checkpoint (NSIDE=32) more details
├── hpunet_n16_epoch=99-val_loss=0.04170.ckpt   # HEALPix U-Net checkpoint (NSIDE=16) smoother
└── vae-epoch=19-step=31320.ckpt                # Panorama VAE checkpoint
```

## Inference

### Full pipeline

The full pipeline accepts either one perspective image or a directory of images and produces 512 × 256 HDR panoramas in OpenEXR format.

```bash
python inference.py \
  --input test_images/inputs \
  --output test_images/outputs
```

The output layout is:

```text
test_images/outputs/
├── hdr/       # Final HDR environment maps (.exr)
├── healpix/   # Intermediate HEALPix predictions (.exr)
└── concat/    # Projected conditioning previews (.png)
```

### Panorama VAE

`inference_vae.py` reconstructs an equirectangular panorama with the released VAE. Inputs are resized to 512 × 256.

```bash
python inference_vae.py \
  --input test_images/inputs-vae \
  --output test_images/outputs-vae
```

## Dataset and training

Training code is planned for a future release.

## Acknowledgements

This project builds on ideas and components from [IllumiDiff](https://github.com/nauyihsnehs/illumidiff), [ControlNet](https://github.com/lllyasviel/ControlNet), [NNhealpix](https://github.com/aasensio/sphericalCNN), and [Skylibs](https://github.com/soravux/skylibs).

## Citation

If you find this work useful, please cite:

```bibtex
@article{bao2026illumiexp,
    title = {IllumiExp: All-Frequency Illumination Estimation via HEALPix-Guided Diffusion},
    author = {Bao, Zhongyun and Shen, Shiyuan and Shen, Xiangqian and Liang, Chao and Xiao, Chunxia},
    journal = {IEEE Transactions on Image Processing},
    year = {2026},
    publisher = {IEEE}
}
```

## Contact

For questions, please contact [syshen@whu.edu.cn](mailto:syshen@whu.edu.cn).

[![Visitors](https://visitor-badge.laobi.icu/badge?page_id=nauyihsnehs.IllumiExp)](https://visitor-badge.laobi.icu/)
