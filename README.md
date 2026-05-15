<div align="center">

# TemporalUpscaling

### Motion-aware video super-resolution built with PyTorch, RAFT optical flow, residual feature fusion, and real video export.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-Temporal%20SR-ee4c2c.svg)](#)
[![CUDA](https://img.shields.io/badge/CUDA-Accelerated-76b900.svg)](#)
[![Video](https://img.shields.io/badge/Video-FFmpeg-black.svg)](#)
[![Scale](https://img.shields.io/badge/Upscale-2x%20%7C%203x%20%7C%204x-purple.svg)](#)

**TemporalUpscaling** is a custom video super-resolution system that reconstructs sharper high-resolution frames by using information from neighbouring video frames instead of treating each frame as an isolated image.

It combines deep residual feature extraction, RAFT-based optical flow, temporal warping, bidirectional context fusion, perceptual training losses, dataset tooling, benchmarking, and an FFmpeg-powered video export pipeline.

</div>

---

## Why this project is cool

Most basic upscalers look at one frame at a time.

This project does something smarter:

```text
previous frame  ─┐
                 ├─ optical flow alignment ─ temporal fusion ─ reconstruction ─ super-res frame
current frame   ─┤
next frame      ─┘
```

Instead of hallucinating detail from a single low-resolution image, the model uses motion-aware context from surrounding frames. The previous super-resolved frame is warped into alignment, the next frame is used as forward temporal context, and the current frame anchors the reconstruction.

The result is a pipeline designed for video, not just a still-image model slapped onto video frames.

---

## Highlights

- **Temporal super-resolution network** using current LR, previous SR, next LR, backward flow, and forward flow as model inputs.
- **RAFT optical flow integration** for high-quality motion estimation between video frames.
- **Bidirectional feature fusion** that combines previous, current, and future frame context.
- **Residual CNN backbone** with configurable channel width, residual block count, and scale factor.
- **PixelShuffle reconstruction** for efficient 2x, 3x, and 4x upscaling.
- **Charbonnier + VGG perceptual loss** for stable pixel accuracy and sharper perceptual output.
- **Vimeo-90K and REDS dataset support**, including combined dataset training.
- **Precomputed flow support** to avoid recomputing RAFT flows during training.
- **Benchmarking suite** with PSNR, SSIM, bicubic baseline comparison, inference timing, FPS, and optional image saving.
- **Real video export** using FFmpeg raw frame IO, audio passthrough, CRF quality control, and optional fp16 inference.
- **ONNX Runtime test harness** with CUDA IO binding for lower-overhead inference experiments.

---

## Architecture

The core model lives in `model/network.py` and is built around five main pieces:

### 1. Feature extraction

Low-resolution frames are passed through a convolutional head and a stack of residual blocks:

```text
LR frame → Conv → LeakyReLU → Residual Blocks → LR features
```

The previous super-resolved frame uses its own high-resolution feature extractor so temporal history is preserved at output resolution instead of being immediately crushed back down.

### 2. Optical-flow warping

Neighbouring frames are aligned with the current frame using flow fields. The project supports:

- `RaftFlow` using torchvision's pretrained RAFT Large model.
- `SimpleFlowEstimator`, a lightweight SpyNet-style pyramid flow estimator.

### 3. Bidirectional temporal fusion

The model fuses:

- current LR features,
- warped previous HR features,
- warped next-frame LR features.

The previous HR feature map is compressed with `pixel_unshuffle`, allowing high-resolution temporal information to be fused at low resolution without simply throwing it away.

### 4. Reconstruction

The fused feature representation is reconstructed into an RGB super-resolved frame using PixelShuffle upsampling.

Supported scales:

| Scale | Reconstruction path |
|---:|---|
| 2x | One PixelShuffle stage |
| 3x | One PixelShuffle stage |
| 4x | Two chained 2x PixelShuffle stages |

### 5. Temporal feedback

During video inference, each generated SR frame becomes the previous SR input for the next frame. That gives the model temporal memory across the video stream.

---

## Repository layout

```text
TemporalUpscaling/
├── data/
│   └── dataset.py          # Vimeo-90K / REDS loaders, crops, augmentation, precomputed flow loading
├── model/
│   ├── network.py          # TemporalSRNet architecture
│   ├── losses.py           # Charbonnier, perceptual, combined loss
│   ├── raft.py             # RAFT optical flow wrapper
│   └── warp.py             # warping utilities + simple flow estimator
├── train.py                # training loop, validation, checkpointing, TensorBoard logging
├── benchmark.py            # PSNR / SSIM / bicubic baseline / inference timing
├── precompute.py           # precompute RAFT flows for Vimeo-90K
├── prepare.py              # validate datasets and compute dataset statistics
├── video.py                # full video upscaling pipeline
├── export.py               # video export/upscaling entry point
└── test.py                 # ONNX Runtime CUDA IO-binding experiment
```

---

## Installation

```bash
git clone https://github.com/<your-username>/TemporalUpscaling.git
cd TemporalUpscaling

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install torch torchvision torchaudio
pip install numpy pillow tqdm scikit-image tensorboard onnxruntime-gpu
```

You also need **FFmpeg** for video export:

```bash
ffmpeg -version
ffprobe -version
```

---

## Dataset setup

The project supports:

- **Vimeo-90K triplet dataset**
- **REDS training sequences**
- **Combined Vimeo-90K + REDS training**

Expected Vimeo-90K structure:

```text
/path/to/vimeo90k/
├── sequences/
│   └── 00001/0001/im1.png
│                    im2.png
│                    im3.png
├── tri_trainlist.txt
└── tri_testlist.txt
```

Validate the dataset:

```bash
python prepare.py \
  --dataset vimeo90k \
  --data_root /path/to/vimeo90k
```

Compute simple dataset statistics:

```bash
python prepare.py \
  --dataset vimeo90k \
  --data_root /path/to/vimeo90k \
  --compute_stats
```

---

## Precompute RAFT flow

RAFT gives better motion estimates, but computing it on the fly is expensive. You can precompute backward and forward flow for Vimeo-90K:

```bash
python precompute.py \
  --dataset vimeo90k \
  --data_root /path/to/vimeo90k \
  --scale 2 \
  --batch_size 8
```

This writes:

```text
back_flow.npy
forward_flow.npy
```

inside each triplet folder.

---

## Training

Train on Vimeo-90K:

```bash
python train.py \
  --dataset vimeo90k \
  --data_root /path/to/vimeo90k \
  --scale 2 \
  --batch_size 8 \
  --patch_size 128 \
  --epochs 100 \
  --flow_estimator raft \
  --checkpoint_dir checkpoints \
  --log_dir runs \
  --device cuda
```

Train with precomputed RAFT flow:

```bash
python train.py \
  --dataset vimeo90k \
  --data_root /path/to/vimeo90k \
  --scale 2 \
  --batch_size 8 \
  --patch_size 128 \
  --epochs 100 \
  --use_precomputed_flow \
  --checkpoint_dir checkpoints \
  --log_dir runs \
  --device cuda
```

Train on combined Vimeo-90K + REDS:

```bash
python train.py \
  --dataset combined \
  --vimeo_root /path/to/vimeo90k \
  --reds_root /path/to/REDS \
  --scale 2 \
  --batch_size 8 \
  --patch_size 128 \
  --epochs 100 \
  --flow_estimator raft \
  --device cuda
```

The training loop includes:

- AdamW optimizer
- cosine annealing LR scheduler
- AMP mixed precision support
- gradient clipping
- TensorBoard logging
- validation PSNR / SSIM
- best-checkpoint saving by PSNR
- scheduled sampling noise on previous HR input to reduce exposure bias

Launch TensorBoard:

```bash
tensorboard --logdir runs
```

---

## Benchmarking

Evaluate a PyTorch checkpoint:

```bash
python benchmark.py \
  --checkpoint checkpoints/best.pth \
  --dataset vimeo90k \
  --data_root /path/to/vimeo90k \
  --scale 2 \
  --flow_estimator raft \
  --device cuda \
  --save_images \
  --output_dir benchmark_results
```

Evaluate an ONNX model:

```bash
python benchmark.py \
  --onnx_model temporal_upscaler.onnx \
  --dataset vimeo90k \
  --data_root /path/to/vimeo90k \
  --scale 2 \
  --flow_estimator raft \
  --output_dir benchmark_results
```

The benchmark reports:

```text
Model SR:
  PSNR:  xx.xx ± x.xx dB
  SSIM:  x.xxxx ± x.xxxx

Bicubic:
  PSNR:  xx.xx ± x.xx dB
  SSIM:  x.xxxx ± x.xxxx

PSNR gain over bicubic: +x.xx dB

Inference Timing:
  Average / Median / P95 / P99 / Min / Max / FPS
```

Results are saved as `.npz` files for later analysis.

---

## Upscale a video

```bash
python video.py \
  --checkpoint checkpoints/best.pth \
  --input input.mp4 \
  --output output_2x.mp4 \
  --scale 2 \
  --device cuda \
  --crf 16 \
  --preset medium
```

Use fp16 inference:

```bash
python video.py \
  --checkpoint checkpoints/best.pth \
  --input input.mp4 \
  --output output_2x_fp16.mp4 \
  --scale 2 \
  --device cuda \
  --fp16
```

Process only the first 100 frames for testing:

```bash
python video.py \
  --checkpoint checkpoints/best.pth \
  --input input.mp4 \
  --output preview.mp4 \
  --scale 2 \
  --limit 100
```

The video pipeline:

1. probes resolution and FPS with `ffprobe`,
2. streams raw RGB frames from FFmpeg,
3. estimates backward and forward flow with RAFT,
4. generates each super-resolved frame,
5. writes raw RGB frames back to FFmpeg,
6. encodes to H.264 with optional audio passthrough.

---

## ONNX Runtime experiment

`test.py` contains a CUDA IO-binding benchmark for ONNX Runtime. It avoids repeated CPU/GPU memory transfers by binding inputs and outputs directly as CUDA OrtValues.

```bash
python test.py
```

This is useful for testing deployment performance separately from the PyTorch training path.

---

## What makes this more than a toy project

This project covers the full lifecycle of a real video super-resolution system:

| Area | Implemented |
|---|---|
| Model architecture | Custom temporal SR network |
| Motion estimation | RAFT + simple pyramid fallback |
| Temporal alignment | differentiable flow warping |
| Training | AMP, AdamW, scheduler, checkpointing |
| Losses | Charbonnier + perceptual VGG loss |
| Datasets | Vimeo-90K, REDS, combined training |
| Evaluation | PSNR, SSIM, bicubic baseline, timing |
| Export | FFmpeg video pipeline with audio passthrough |
| Performance testing | ONNX Runtime CUDA IO binding |

It is a complete research-to-demo pipeline: train the model, validate it, benchmark it, and run it on real videos.

---

## Roadmap

- Add sample before/after GIFs to the README.
- Add a public checkpoint release.
- Add ONNX export script for the trained PyTorch model.
- Add tiled inference for very high-resolution videos.
- Add temporal consistency metrics.
- Add side-by-side video comparison generation.
- Add REDS precomputed flow support.
- Add config files for repeatable experiments.

---

## Example results section

Once benchmarks are finalized, fill this in:

| Model | Dataset | Scale | PSNR | SSIM | FPS | Notes |
|---|---|---:|---:|---:|---:|---|
| Bicubic | Vimeo-90K | 2x | TBD | TBD | TBD | baseline |
| TemporalUpscaling | Vimeo-90K | 2x | TBD | TBD | TBD | RAFT flow |


---

## Acknowledgements

This project builds on ideas from video super-resolution research: optical flow alignment, recurrent temporal feedback, residual reconstruction, perceptual losses, and benchmark-driven evaluation.

It uses PyTorch, torchvision RAFT, FFmpeg, scikit-image, TensorBoard, and ONNX Runtime.

---

<div align="center">

### TemporalUpscaling
</div>