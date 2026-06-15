<h1 align="center">PROSE: Training-Free Egocentric Scene Registration<br>with Vision-Language Models</h1>

<div align="center">

[Zhiang Chen](https://RCKola.github.io/)\*<sup>,1</sup>,
[Nahyuk Lee](https://nahyuklee.github.io/)\*,
[Boyang Sun](https://boysun045.github.io/boysun-website/)<sup>1</sup>,
[Taein Kwon](https://taeinkwon.com/)<sup>2</sup>,
[Marc Pollefeys](https://cvg.ethz.ch/team/Prof-Dr-Marc-Pollefeys)<sup>1</sup>,
[Zuria Bauer](https://zuriabauer.com/)<sup>†,1</sup>,
[Sunghwan Hong](https://sunghwanhong.github.io/)<sup>†,1,3</sup>

<sup>1</sup> [ETH Zurich](https://ethz.ch/en.html) &ensp; <sup>2</sup> [VGG, University of Oxford](https://www.robots.ox.ac.uk/~vgg/) &ensp; <sup>3</sup> [ETH AI Center](https://ai.ethz.ch/)<br>
\* equal contribution &ensp; † equal advising

[![Project Page](https://img.shields.io/badge/Project_Page-PROSE-blue)](https://rckola.github.io/prose/)
[![Video](https://img.shields.io/badge/Video-YouTube-red?logo=youtube)](https://www.youtube.com/watch?v=Hf1oWjFr45M)

</div>

<p align="center">
  <img src="assets/architecture.png" width="100%">
</p>

**_TL;DR:_** *Register two egocentric RGB sequences of the same indoor scene by lifting each into an object-level 3D scene graph with off-the-shelf foundation models, then prompting a VLM to match instances across scans — no depth sensor, no training, no annotated graph.*

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Setup](#setup)
  - [Foundation-Model Weights](#foundation-model-weights)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Running the Pipeline](#running-the-pipeline)
  - [Evaluation](#evaluation)
- [Advanced](#advanced)
  - [Configurations](#configurations)
  - [FAQ](#faq)
- [Acknowledgments](#acknowledgments)
- [License](#license)


## Overview

**PROSE** (**Pro**mpted **S**cene r**E**gistration) recovers the rigid transform aligning two egocentric RGB captures of the same indoor space taken at different times. Rather than relying on the clean point clouds that egocentric, RGB-only capture lacks, PROSE uses pretrained foundation models for geometry, segmentation, and language, and prompts a VLM for both scene understanding and cross-scan matching. **It adds no learned parameters and needs no depth sensor, training, or annotated scene graph.**

The pipeline runs in six stages — four per-subscan (scene parsing) and two per-pair:

| Stage | Module | Model | Role |
|:--|:--|:--|:--|
| 1. Geometry | `stages/geometry.py` | **VGGT-Ω** / GT | per-frame depth + camera → per-subscan point cloud |
| 2. Object listing | `stages/object_listing.py` | **Qwen3.6-27B** | VLM lists the objects worth matching |
| 3. Segmentation | `stages/segmentation.py` | **SAM 3** | text-prompted, temporally consistent instance masks |
| 4. Fusion | `stages/fusion.py` | — | per-instance 3D fusion → scene graph G=(V,E): instance nodes (+ OBB) and proximity edges |
| 5. Correspondence | `stages/correspondence/` | **Qwen3.6-27B** | height-binned Set-of-Marks matching + same/different double-check |
| 6. Registration | `stages/registration.py` | GeoTransformer / FCGF / FPFH | per-pair RANSAC hypotheses + geometric-consensus voting |

### Key features
- **Training-free**: every model is an off-the-shelf checkpoint used as released.
- **Sensor-free**: geometry is estimated from RGB by VGGT-Ω; GT clouds are also supported for the benchmark setting.
- **Object-level scene graph**: the fusion stage emits G=(V,E) per scan — instance nodes (fused point set + PCA oriented box + shape descriptor) and k-NN proximity edges with coarse `above`/`below`/`near` relations — which transfers directly to downstream tasks.
- **Height-binned correspondence**: K=5 quantile height bands with 20% overlap remove trivial distractors — the largest source of improvement in the ablation.
- **Double-check verification**: a paired *same?* / *different?* query suppresses VLM false positives.
- **Three registration backends**: GeoTransformer (default), FCGF, FPFH — swap with one flag.


## Getting Started

### Setup

Clone with submodules (VGGT-Ω, GeoTransformer, FCGF):

```bash
git clone --recursive https://github.com/<org>/prose.git
cd prose
```

Create a Python 3.10 environment:

```bash
conda create -n prose python=3.10 -y
conda activate prose
```

Install PyTorch (pick the CUDA build matching your driver), then the package:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

PROSE was developed on a single **H200** GPU. Heavy models are loaded one at a time, so a 32 GB card is workable, but Qwen3.6-27B is the binding constraint — use [vLLM](#configurations) (`pip install -e ".[vllm]"`) for the object-listing and correspondence stages.

Optional extras:

```bash
pip install -e ".[vllm]"     # fast Qwen3.6 inference (recommended)
pip install -e ".[viz]"      # Open3D — required for the FPFH backend + PLY viz
pip install -e ".[ransac]"   # pygcransac (falls back to Umeyama SVD if absent)
```

Build the GeoTransformer C++ extension and fetch its 3DMatch weights (only needed for the default registration backend):

```bash
bash scripts/setup_geotransformer.sh
export PYTHONPATH="$(pwd)/third_party/GeoTransformer:$PYTHONPATH"
```

Finally, copy the secrets template and set your HuggingFace token:

```bash
cp .env.example .env
# edit .env: set HF_TOKEN (required for gated repos like facebook/sam3)
```

`.env` is auto-loaded on startup — no need to `export` manually.

### Foundation-Model Weights

PROSE is **training-free**, so there are no PROSE checkpoints to release. It composes the following off-the-shelf models:

| Model | Stage | Source | Notes |
|:--|:--|:--|:--|
| **VGGT-Ω** | geometry | [facebook/VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega) | gated; download once, set `VGGT_OMEGA_CKPT`. Only used when `use_gt_pointclouds=false` |
| **SAM 3** | segmentation | [facebook/sam3](https://huggingface.co/facebook/sam3) | gated; accept license, then auto-downloaded via `HF_TOKEN` |
| **Qwen3.6-27B** | listing + correspondence | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | auto-downloaded on first run |
| GeoTransformer | registration | [qinzheng93/GeoTransformer](https://github.com/qinzheng93/GeoTransformer) (3DMatch) | fetched by `setup_geotransformer.sh` |
| FCGF | registration | [chrischoy/FCGF](https://github.com/chrischoy/FCGF) (3DMatch) | needs MinkowskiEngine; place weights at `weights/fcgf/fcgf_3dmatch.pth` |
| FPFH | registration | Open3D (handcrafted) | no weights |

The Qwen/SAM3 weights are fetched lazily by HuggingFace on first use; `python scripts/download_checkpoints.py` only fetches the GeoTransformer asset.


## Usage

### Data Preparation

PROSE supports **Aria Digital Twin (ADT)**. ADT is gated and has no public programmatic downloader — request access and obtain the signed URLs from the [official release](https://www.projectaria.com/datasets/adt/), then:

```bash
# 1. Download a sequence (signed URLs from the ADT release)
python scripts/download_adt.py --urls-json <ADT_download_urls.json> \
    --output sample_data/adt --sequence Apartment_release_clean_seq133_M1292

# 2. Preprocess to the flat RGB/depth/pose tree the pipeline reads.
#    projectaria_tools has no Python 3.13 wheel, so run this in a 3.11 env:
conda create -n prose_adt python=3.11 -y
conda run -n prose_adt pip install projectaria_tools opencv-python
conda run -n prose_adt python preprocessing/adt/preprocess_adt.py \
    --sequence-dir sample_data/adt/Apartment_release_clean_seq133_M1292
```

Preprocessing samples frames every 0.5 s, rectifies RGB/depth, undistorts and rotates them upright (`*_rot/`), and writes the sliding-window subscan pairs (`window=6`, `stride=5`) to `anchors_val.json`. The runtime pipeline then reads only this flat tree — no `projectaria_tools` dependency at run time.

### Running the Pipeline

Run the full six-stage pipeline on the ground-truth-cloud benchmark setting:

```bash
python scripts/run_pipeline.py pairs=all
```

Run the **sensor-free** setting (geometry predicted from RGB by VGGT-Ω):

```bash
python scripts/run_pipeline.py pairs=all use_gt_pointclouds=false
```

Every intermediate artifact is cached under `outputs/<date>/<time>/` (one subdir per stage), so re-runs and `skip_stages=[...]` resume instantly. A quick smoke test on one pair:

```bash
python scripts/run_pipeline.py pairs=first_n max_pairs=1
```

### Evaluation

Evaluation runs automatically after registration (`run_evaluation=true`) and writes `evaluation/metrics.json`:

- **Registration** — Registration Recall (RR; RRE < 5° and RTE < 0.2 m), RRE, RTE, Valid Ratio.
- **Correspondence** — node precision / recall / F1 against the SAM3↔SAM3 mutual-NN IoU ground truth, for the raw and double-checked match sets.

Swap the registration descriptor (all three are reported in the paper):

```bash
python scripts/run_pipeline.py registration.corr_extractor=fcgf   # or fpfh / geotransformer
```


## Advanced

### Configurations

<details>
<summary>Hydra configuration structure</summary>

```
configs/
├── config.yaml             # entry config (defaults + global flags)
├── pipeline/default.yaml   # precision / attention (bf16 + FlashAttention-2)
├── dataset/adt.yaml        # ADT paths + sliding-window params
├── geometry/vggt_omega.yaml
├── object_listing/qwen.yaml
├── segmentation/sam3.yaml
├── fusion/default.yaml
├── correspondence/height_bins.yaml
├── registration/default.yaml
└── output/default.yaml
```

Key global parameters in `config.yaml`:
- `use_gt_pointclouds`: GT clouds (true) vs. VGGT-Ω-predicted clouds (false).
- `pairs`: `all`, `first_n` (+ `max_pairs`), or an explicit list of `"<src>__<ref>"` ids.
- `skip_stages`: list of `geometry, object_listing, segmentation, fusion, correspondence, registration`.
- `registration.corr_extractor`: `geotransformer` (default), `fcgf`, `fpfh`.
- `fusion.scene_graph`: scene-graph edges (`enabled`, `k_neighbors=5`, `radius_m=1.0`, `up_axis`). Stored in `PriorArtifact.edges` as `{src, dst, dist, relation}` records.

Any field is overridable on the command line, e.g.:

```bash
python scripts/run_pipeline.py \
    correspondence.blocking.n_bins=5 \
    correspondence.do_double_check=true \
    registration.corr_extractor=fpfh
```

On older GPUs without FlashAttention-2 / bf16:

```bash
python scripts/run_pipeline.py pipeline.torch_dtype=float16 pipeline.attn_implementation=sdpa
```

</details>

### FAQ

> [!NOTE]
> Please use [GitHub Issues](https://github.com/<org>/prose/issues) for questions.

> [!TIP]
> If the VGGT-Ω package or checkpoint is missing, the geometry stage logs a warning and falls back to GT point clouds — handy for testing the rest of the pipeline before setting up the gated checkpoint.


## Acknowledgments

PROSE builds on the released checkpoints of [VGGT-Ω](https://github.com/facebookresearch/vggt-omega), [SAM 3](https://github.com/facebookresearch/sam3), [Qwen3.6-VL](https://github.com/QwenLM/Qwen3-VL), [GeoTransformer](https://github.com/qinzheng93/GeoTransformer), and [FCGF](https://github.com/chrischoy/FCGF). We thank the authors of these works.


## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.
