# SceneCutGUI

A modern, GPU-accelerated scene detection and video slicing tool with an intuitive Tkinter GUI. Uses **TransNetV2** neural network for highly accurate shot boundary detection, with optional **DINOv3-based AI validation** to filter false positives.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![CUDA](https://img.shields.io/badge/CUDA-Accelerated-76B900?logo=nvidia&logoColor=white)

---

## ✨ Features

- **TransNetV2 Detection** — State-of-the-art neural network for shot boundary detection
- **DINOv3 AI Validation** — Optional post-processing to filter out flashes, fast motion, and near-black false positives
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
transnetv2-pytorch
transformers
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

<a href="https://drive.google.com/file/d/1aJ96eJE4DstJ_HBzAusuYLvfU01Jnc92/view?usp=sharing">Download Model Weights</a> and place them in the `weights/` directory:

| File | Description |
|------|-------------|
| `transnetv2-pytorch-weights.pth` | TransNetV2 model weights |
| `model.safetensors` | DINOv3 model (for AI validation) |
| `config.json` | DINOv3 model config |
| `preprocessor_config.json` | DINOv3 preprocessor config |

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

| Parameter | Default | Description |
|-----------|---------|-------------|
| `device` | `auto` | Compute device (`auto`, `cuda`, `cpu`, `mps`) |
| `threshold` | `0.5` | Cut detection sensitivity (0–1) |
| `min_scene_len` | `8` | Minimum scene length in frames |

### AI Validation (Optional)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ai_validate` | `false` | Enable DINOv3 validation |
| `ai_window` | `3` | Frames before/after cut to analyze |

### FFmpeg Output Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ffmpeg_codec` | `h264_nvenc` | Video codec (`h264_nvenc`, `hevc_nvenc`, `libx264`) |
| `ffmpeg_preset` | `p7` | NVENC quality preset (p1=fastest, p7=best) |
| `ffmpeg_cq` | `16` | Constant quality level (lower=better) |

---

## 📁 Output Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| **CSV** | `.csv` | Scene list with timecodes and frame numbers |
| **HTML** | `.html` | Visual report with scene table |
| **SC File** | `.sc` | DaVinci Resolve scene cut format |
| **Images** | `.jpg` | Thumbnail frames per scene |
| **Video Clips** | `.mp4` | Split video per scene (via FFmpeg) |

---

## 🧠 How It Works

### TransNetV2 Detection
TransNetV2 is a deep learning model specifically trained for shot boundary detection. It processes video frames and outputs probability scores for each frame being a scene transition.

### DINOv3 AI Validation
The optional validation step uses a DINOv3 vision transformer to:
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

- [TransNetV2](https://github.com/soCzech/TransNetV2) — Shot boundary detection model
- [transnetv2-pytorch](https://pypi.org/project/transnetv2-pytorch/) — PyTorch implementation
- [DINOv3](https://github.com/facebookresearch/dinov3) — Vision transformer for validation
- [FFmpeg](https://ffmpeg.org/) — Video processing backend
