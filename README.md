# UHD-Processor: Unified UHD Image Restoration with Progressive Frequency Learning and Degradation-aware Prompts [CVPR 2025]

This repository provides an implementation overview of UHD-Processor, aligned with the CVPR 2025 paper. It includes a method summary, datasets, installation and usage guidance, and citation information for reproduction and extension.

- Paper (CVPR 2025): `https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_UHD-processer_Unified_UHD_Image_Restoration_with_Progressive_Frequency_Learning_and_CVPR_2025_paper.pdf`
- Supplementary material: `https://openaccess.thecvf.com/content/CVPR2025/supplemental/Liu_UHD-processer_Unified_UHD_CVPR_2025_supplemental.pdf`

## Overview
UHD-Processor is a unified, resource-efficient framework for ultra-high-definition (UHD) image restoration that generalizes across diverse degradations, including denoising, deblurring, dehazing, low-light enhancement, deraining, and desnowing. The framework leverages a variational autoencoder (VAE) to learn in a compact latent space and adopts a progressive frequency learning strategy to stabilize and accelerate optimization from low to high frequencies.

## Method
- Progressive Frequency Learning (PFL)
  - Decomposes the restoration problem into multiple frequency bands.
  - Trains from easy low-frequency components to harder high-frequency details, improving stability and convergence.

- Efficient VAE-based Latent Modeling
  - Performs restoration primarily in the latent space rather than pixel space, substantially reducing FLOPs and memory.
  - The encoder is equipped with degradation awareness, while the decoder performs frequency selection and detail synthesis.

- Degradation-aware Prompts
  - DALR (Degradation-Aware Low-Rank Prompt, encoder stage): injects low-rank degradation priors to unify latent modeling across degradation types.
  - DFSP (Degradation-Specific Frequency Selection Prompt, decoder stage): adaptively fuses high-frequency details and low-frequency structures conditioned on the degradation type.

According to the paper, UHD-Processor achieves state-of-the-art performance with significantly fewer parameters and FLOPs (e.g., ≈95.1% parameter and ≈97.4% FLOP reductions compared to prior art).

## Key Features
- Unified multi-degradation handling in a single model.
- UHD-friendly efficiency via latent-space learning and progressive frequency training.
- Prompt-based adaptivity (DALR/DFSP) for degradation- and frequency-aware processing.
- Strong quality–efficiency tradeoff on UHD benchmarks.

## Datasets
Download links by degradation category (as referenced in the paper):

- UHD_deblur (deblurring): `https://drive.google.com/file/d/1-RCHHPMh95Pnm0Wj773QvKpNm9WoZV9l/view?usp=sharing`
- UHD_haze (dehazing): `https://drive.google.com/file/d/10dFZZMep3k2p3r8houGkKevTw3XaVrtq/view?usp=sharing`
- UHD_LL (low-light): `https://drive.google.com/file/d/1O31UC6MJ3pHOIPLzXlqHqvcIwgfy18_f/view?usp=sharing`
- UHD_rain (deraining): `https://drive.google.com/file/d/1jkBnyVKND-f5WZ4mn8c7tnKPekB2qsxJ/view?usp=sharing`
- UHD_snow (desnowing): `https://drive.google.com/file/d/1rsn-S5EDKo5yw-wlxCjrudx7ih056Ro8/view?usp=sharing`

Recommended directory structure (example):

```
data/
  UHD_deblur/
    train/  val/  test/
  UHD_haze/
    train/  val/  test/
  UHD_LL/
    train/  val/  test/
  UHD_rain/
    train/  val/  test/
  UHD_snow/
    train/  val/  test/
```

## Installation
Clone the repository and create the environment:

```
git clone https://github.com/lyd-2022/UHD-processer.git
cd UHD-processer

conda create -n uhd_processor python=3.8 -y
conda activate uhd_processor

pip install -r requirements.txt
```

## Usage
### Training
The paper adopts a two-stage training pipeline:
1) Pre-train the VAE on clean images to learn high-quality latent representations.
2) Train the unified restoration model with the progressive frequency learning schedule and enable DALR/DFSP prompts.

Full training scripts and configs will be released. Placeholder examples:

```
# Stage 1: VAE pre-training
python tools/train_vae.py \
  --config configs/vae/vae_pretrain.yaml \
  --data_root /path/to/data \
  --work_dir ./work_dirs/vae_pretrain

# Stage 2: Unified restoration training (PFL)
python tools/train_restoration.py \
  --config configs/restoration/uhd_processor_pfl.yaml \
  --data_root /path/to/data \
  --work_dir ./work_dirs/uhd_processor
```

### Inference
Pretrained weights will be provided. Placeholder example:

```
python inference.py \
  --model_path /path/to/pretrained.pth \
  --input_dir  /path/to/degraded_images \
  --output_dir /path/to/restored_results \
  --task deblur   # also supports: denoise/dehaze/ll/rain/snow
```

## Results and Visualization
- In unified settings (4- and 6-degradation), UHD-Processor outperforms prior methods on standard metrics.
- Visual comparisons show superior detail restoration and artifact suppression, particularly at high resolutions.

Please refer to the paper and supplementary for complete quantitative tables and figures.

## FAQ
- High memory footprint? Reduce batch size, enable mixed precision, or use tiled inference for very large inputs.
- Unstable training? Verify learning rates and PFL schedule; ensure robust VAE convergence in Stage 1.
- Color shifts or over-smoothing? Tune DFSP weights or frequency gating thresholds to strengthen high-frequency fusion.

## Citation
If you find this project useful in your research, please consider citing:

```bibtex
@InProceedings{Liu_2025_CVPR,
    author    = {Liu, Yidi and Li, Dong and Fu, Xueyang and Lu, Xin and Huang, Jie and Zha, Zheng-Jun},
    title     = {UHD-processer: Unified UHD Image Restoration with Progressive Frequency Learning and Degradation-aware Prompts},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {23121-23130}
}
```

## License and Acknowledgements
- The license and usage terms for code and pretrained models will follow the forthcoming LICENSE file.
- We thank the open-source community and dataset contributors for their support.