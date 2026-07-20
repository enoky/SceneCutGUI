# SceneCutGUI

A modern, GPU-accelerated scene detection and video slicing tool with an intuitive Tkinter GUI. Uses **AutoShot**, **TransNetV2** or **OmniShotCut** neural networks for highly accurate shot boundary detection, with optional **DINOv3/TIPSv2-based AI validation** to filter false positives.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![CUDA](https://img.shields.io/badge/CUDA-Accelerated-76B900?logo=nvidia&logoColor=white)

---

## ✨ Features

- **AutoShot / TransNetV2 / OmniShotCut Detection** — State-of-the-art neural networks for shot boundary detection, including transition-aware detection (dissolves, wipes, fades) via OmniShotCut
- **DINOv3/TIPSv2/SSCD AI Validation** — Optional post-processing to filter out flashes, fast motion, and near-black false positives
- **GPU Acceleration** — CUDA support for both video decoding (OpenCV) and model inference (PyTorch)
- **Multiple Export Formats**:
  - CSV scene list
  - HTML report
  - `.sc` scene cut file (DaVinci Resolve compatible)
  - Thumbnail images per scene
  - FFmpeg-based video splitting with NVENC encoding
- **Configurable Parameters** — Fine-tune detection threshold, minimum scene length, and validation window
- **Settings Persistence** — Automatically saves/loads your configuration

---

## 🛠 Requirements

- **Python 3.10+**
- **NVIDIA GPU** with CUDA support (required for AI validation; recommended for optimal performance)
- **FFmpeg** (for video splitting and probing)

### Python Dependencies

```
torch>=2.0
torchvision
torchaudio
einops
transnetv2-pytorch
transformers
sentencepiece
numpy
opencv-contrib-python with CUDA (see link below)
```

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/enoky/SceneCutGUI.git
cd SceneCutGUI
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note**: For GPU-accelerated video decoding, install OpenCV with CUDA support from:
> [opencv-python-cuda-wheels](https://github.com/cudawarped/opencv-python-cuda-wheels/releases)

Install OpenCV with CUDA support command example:

```bash
pip install opencv_contrib_python-4.13.0.90-cp37-abi3-win_amd64.whl
```

### 4. Download Model Weights

<a href="https://drive.google.com/file/d/1aJ96eJE4DstJ_HBzAusuYLvfU01Jnc92/view?usp=sharing">Download</a> the model checkpoints to the `weights/` directory:

| File                             | Description                                 |
| -------------------------------- | ------------------------------------------- |
| `ckpt_0_200_0.pth`               | AutoShot model weights                      |
| `transnetv2-pytorch-weights.pth` | TransNetV2 model weights                    |
| `OmniShotCut_ckpt.pth`           | OmniShotCut model weights                   |
| `DINOv3/*`                       | DINOv3 model (for AI validation)            |
| `TIPSv2/*`                       | TIPSv2 model (for AI validation)            |
| `sscd_disc_large.torchscript.pt` | SSCD model (for AI validation)              |

---

## 🚀 Usage

### Quick Start

```bash
python scene_cut_gui.py
```

Or use the provided batch file:

```bash
RUN_SceneCutGUI.bat
```

### Workflow

1. **Select Video**: Browse and select your input video file
2. **Set Output Folder**: Choose where to save outputs (auto-populated based on video name)
3. **Configure Detection**:
   - Adjust `threshold` (0.0–1.0, lower = more sensitive)
   - Set `min_scene_len` to filter very short scenes
   - Enable **AI Validation** for higher accuracy
4. **Choose Outputs**: Select which formats to export
5. **Start Processing**: Click "Start Processing" and monitor progress

---

## ⚙️ Configuration Options

### Detection Parameters

| Parameter       | Default                                 | Description                                   |
| --------------- | --------------------------------------- | --------------------------------------------- |
| `device`        | `auto`                                  | Compute device (`auto`, `cuda`, `cpu`, `mps`) |
| `threshold`     | `0.296` (AutoShot) / `0.3` (TransNetV2) | Cut detection sensitivity (0–1)               |
| `min_scene_len` | `8`                                     | Minimum scene length in frames                |

The parameter panel changes with the selected detector. OmniShotCut picks boundaries
by argmax over its shot queries rather than by thresholding a score, so it shows no
`threshold` field and offers these instead:

| Parameter | Default      | Description                                                                                                    |
| --------- | ------------ | -------------------------------------------------------------------------------------------------------------- |
| `mode`    | `clean_shot` | `clean_shot` keeps only hard cuts (dissolves/fades are absorbed into the preceding scene); `default` keeps every detected shot, with transitions as their own segments |
| `overlap` | `20`         | Frames shared between consecutive inference windows (model window is 100 frames)                                |

### AI Validation (Optional)

| Parameter           | Default  | Description                                    |
| ------------------- | -------- | ---------------------------------------------- |
| `ai_validate`       | `false`  | Enable DINOv3/TIPSv2/SSCD validation           |
| `ai_val_model`      | `DINOv3` | Model used for validation (DINOv3 or TIPSv2)   |
| `ai_window`         | `5`      | Frames before/after cut to analyze             |
| `flash_sensitivity` | `15`     | Luma delta threshold for flash detection       |

### Image & Thumbnail Settings

| Parameter      | Default | Description                                  |
| -------------- | ------- | -------------------------------------------- |
| `num_images`   | `3`     | Number of thumbnail images to save per scene |
| `frame_margin` | `1`     | Frame offset margin for thumbnail capture    |

### FFmpeg Output Settings

| Parameter       | Default      | Description                                         |
| --------------- | ------------ | --------------------------------------------------- |
| `ffmpeg_codec`  | `h264_nvenc` | Video codec (`h264_nvenc`, `hevc_nvenc`, `libx264`) |
| `ffmpeg_preset` | `p7`         | NVENC quality preset (p1=fastest, p7=best)          |
| `ffmpeg_cq`     | `16`         | Constant quality level (lower=better)               |

---

## 📁 Output Formats

| Format          | Extension | Description                                 |
| --------------- | --------- | ------------------------------------------- |
| **CSV**         | `.csv`    | Scene list with timecodes and frame numbers |
| **HTML**        | `.html`   | Visual report with scene table              |
| **SC File**     | `.sc`     | DaVinci Resolve scene cut format            |
| **Images**      | `.jpg`    | Thumbnail frames per scene                  |
| **Video Clips** | `.mp4`    | Split video per scene (via FFmpeg)          |

---

## 🧠 How It Works

### AutoShot, TransNetV2 and OmniShotCut Detection

All three are deep learning models trained for shot boundary detection. Use the detector selector in the GUI to switch between them based on your content and performance needs.

- **AutoShot / TransNetV2** score every frame for how likely it is to be a cut, and the `threshold` controls sensitivity.
- **OmniShotCut** is a Shot-Query Transformer that predicts shot *ranges* directly and classifies each one (General, Dissolve, Wipes, Push, Slide, Zoom, Fade, Doorway). Because boundaries come from an argmax over queries, there is no threshold to tune — use `mode` to decide whether transitions become their own segments. It is trained on diverse footage (anime, vlogs, gaming, sports, screen recordings), so it is a good first choice for stylised content where the other two over- or under-segment.

OmniShotCut is vendored in `OmniShotCut/` rather than pip-installed, because the upstream
`requirements.txt` pins `transformers==4.57.3`, which would downgrade the version the
DINOv3/TIPSv2 validation depends on. The vendored copy carries three marked local edits:
the ResNet backbone is built with `pretrained=False` (the checkpoint overwrites every
backbone weight anyway, so the download is wasted and breaks offline loading), and the
two hardcoded `.to("cuda")` calls are replaced so the `device` selector works.

The GUI decodes to a memory-mapped temp file and drives inference window-by-window
instead of calling `model.inference()`, which would hold the whole video in RAM
(~36 KB/frame, so roughly 4 GB per hour of 30fps footage) with no progress or abort.
Output is bit-identical to the upstream path.

### DINOv3 / TIPSv2 AI Validation

The optional validation step uses a vision transformer (such as DINOv3 or TIPSv2) to:

- Sample frames before and after each detected cut
- Compute visual embeddings and similarity scores
- Filter out false positives (flashes, fast motion, near-black frames)
- Use adaptive thresholding for per-video optimization

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [TransNetV2](https://github.com/soCzech/TransNetV2) ? Shot boundary detection model
- [transnetv2-pytorch](https://pypi.org/project/transnetv2-pytorch/) ? PyTorch implementation
- [AutoShot](https://github.com/wentaozhu/AutoShot) — Shot boundary detection model
- [OmniShotCut](https://github.com/UVA-Computer-Vision-Lab/OmniShotCut) — Shot-Query Transformer for shot boundary and transition detection (MIT, vendored in `OmniShotCut/`)
- [DINOv3](https://github.com/facebookresearch/dinov3) — Vision transformer for validation
- [TIPSv2](https://github.com/google-deepmind/tips) — Vision language multimodal foundation model
- [FFmpeg](https://ffmpeg.org/) — Video processing backend
