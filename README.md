# UHD-Processor: Unified UHD Image Restoration with Progressive Frequency Learning and Degradation-aware Prompts
Overview

UHD-Processor is a unified and robust all-in-one image restoration framework, specifically designed for Ultra-High-Definition (UHD) images with remarkable resource efficiency.

Conventional all-in-one methods often rely on complex restoration backbones when processing UHD images, leading to high computational costs. To address this challenge, our strategy employs a frequency-domain decoupling progressive learning technique, inspired by curriculum learning, to incrementally learn restoration mappings from low to high frequencies. This divide-and-conquer approach utilizes specialized sub-network modules to effectively tackle different frequency bands, significantly enhancing the learning capability of simpler networks.

Furthermore, to accommodate the high-resolution characteristics of UHD images, we have developed a framework based on a Variational Autoencoder (VAE), which reduces computational complexity by modeling a concise latent space. It integrates task-specific degradation awareness in the encoder and frequency selection in the decoder, thereby enhancing the model's task comprehension and generalization.

Our unified model is capable of handling a variety of degradations, such as denoising, deblurring, dehazing, low-light enhancement, deraining, and desnowing.

Figure 1: The overall framework of our proposed UHD-Processor.

Core Features
Efficient VAE-based Framework: Drastically reduces the computational resources required for UHD image processing by performing restoration tasks within a compact latent space rather than pixel space.

Frequency-Domain Decoupling Progressive Learning Strategy: Decomposes the complex restoration task into sub-tasks across multiple frequency bands. The model begins learning from easier, low-frequency components and progressively advances to more challenging, high-frequency ones, improving adaptability to diverse optimization objectives.

Efficient Adaptive Prompt Learning:

Degradation-Aware Low-Rank Prompt (DALR): During the encoding stage, this prompt assists the VAE in adapting to different degradation types, encoding them into a unified latent space.

Degradation-Specific Frequency Selection Prompt (DFSP): In the decoding stage, this prompt allows the model to adaptively fuse high-frequency details from the encoder with low-frequency information from the decoder, based on the specific degradation type.

Superior Performance and Efficiency: In multiple benchmark tests, UHD-Processor achieves state-of-the-art performance while realizing a 95.1% reduction in parameters and a 97.4% reduction in FLOPs compared to existing methods.

Performance Highlights
We compared UHD-Processor against state-of-the-art methods in all-in-one settings involving four and six degradation types.

All-in-One Restoration Results (Six Degradations)
Method


Visual Comparisons
Figure 2: Visual comparison with other state-of-the-art all-in-one methods on four degradation removal tasks. Our method excels at restoring details and suppressing artifacts.

Installation
## Clone the repository
git clone https://github.com/lyd-2022/UHD-processer.git
cd UHD-processer

## Create and activate a conda environment
conda create -n uhd_processor python=3.8
conda activate uhd_processor

## Install dependencies
pip install -r requirements.txt

Usage
Training
Detailed training instructions and scripts will be provided upon code release. The training process is divided into two stages:

Pre-training the VAE on clean images.

Training the complete restoration model using the frequency-domain decoupling progressive learning strategy.

Inference
We will provide pre-trained models for fast inference on new UHD images upon release.

## Example inference command (coming soon)
python inference.py --model_path /path/to/pretrained_model \
                    --input_dir /path/to/degraded_images \
                    --output_dir /path/to/restored_results \
                    --task deblur # or denoise, dehaze, etc.



## ✍️ Citation
If you find this work useful in your research, please consider citing our paper:

```bibtex
@InProceedings{Liu_2025_CVPR,
    author    = {Liu, Yidi and Li, Dong and Fu, Xueyang and Lu, Xin and Huang, Jie and Zha, Zheng-Jun},
    title     = {UHD-processer: Unified UHD Image Restoration with Progressive Frequency Learning and Degradation-aware Prompts},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {23121-23130}
}
```