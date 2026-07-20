# SceneCutGUI

A modern, GPU-accelerated scene detection and video slicing tool with an intuitive Tkinter GUI. Uses **AutoShot**, **TransNetV2** or **OmniShotCut** neural networks for highly accurate shot boundary detection, with optional **DINOv3-based AI validation** to filter false positives.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![CUDA](https://img.shields.io/badge/CUDA-Accelerated-76B900?logo=nvidia&logoColor=white)

---

## ✨ Features

- **AutoShot / TransNetV2 / OmniShotCut Detection** — State-of-the-art neural networks for shot boundary detection, including transition-aware detection (dissolves, wipes, fades) via OmniShotCut
- **DINOv3/SSCD AI Validation** — Optional post-processing to filter out flashes, fast motion, and near-black false positives
- **GPU Acceleration** — CUDA support for both video decoding (OpenCV) and model inference (PyTorch)
- **Multiple Export Formats**:
  - CSV scene list
  - HTML report
  - `.sc` scene cut file (DaVinci Resolve compatible)
  - Thumbnail images per scene
  - FFmpeg-based video splitting: frame-perfect cuts, 10-bit HEVC via NVENC
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
| `ai_validate`       | `false`  | Enable DINOv3/SSCD validation                  |
| `ai_window`         | `5`      | Frames before/after cut to analyze             |
| `flash_sensitivity` | `15`     | Luma delta threshold for flash detection       |

### Image & Thumbnail Settings

| Parameter      | Default | Description                                  |
| -------------- | ------- | -------------------------------------------- |
| `num_images`   | `3`     | Number of thumbnail images to save per scene |
| `frame_margin` | `1`     | Frame offset margin for thumbnail capture    |

### FFmpeg Output Settings

Scene clips are always written as **10-bit HEVC (Main10, `p010le`)** encoded on the
NVIDIA GPU via `hevc_nvenc`. The codec is not user-selectable; if the installed
FFmpeg build has no NVENC support, splitting falls back automatically to CPU
`libx265` 10-bit. Clips are tagged `hvc1` for QuickTime/Apple/Adobe compatibility,
and source colour range/space/transfer/primaries are carried through unchanged.

**The whole pipeline runs on the GPU**, not just the encode: frames are decoded by
NVDEC, stay in GPU memory through the trim, and the 8→10-bit conversion is done by
`scale_cuda` before going straight into NVENC — no frame is ever copied to system
memory. `trim` and `setpts` only pass/drop frames and rewrite timestamps, so they
work unchanged on CUDA surfaces. Sources NVDEC cannot handle (4:2:2 chroma, ProRes,
DNxHD, AV1 on older GPUs) are detected up front with a one-frame test decode and
fall back to CPU decoding, with encoding still on the GPU.

| Parameter       | Default      | Description                                         |
| --------------- | ------------ | --------------------------------------------------- |
| `ffmpeg_codec`  | `hevc_nvenc` | Fixed — 10-bit HEVC on NVENC (libx265 10-bit fallback) |
| `ffmpeg_preset` | `p7`         | NVENC quality preset (p1=fastest, p7=best)          |
| `ffmpeg_cq`     | `16`         | Constant QP level (lower=better)                    |

**Frame-perfect cutting.** Clips never start on a keyframe-rounded boundary — cuts
land on exactly the frame the detector chose. Timestamps are regenerated as strict
CFR (`setpts=N/FRAME_RATE/TB` plus `-fps_mode cfr`) at the source's exact *rational*
frame rate, so fractional rates such as 24000/1001 do not drift and no frame is
dropped or duplicated. Every clip's frame count is verified after encoding and any
mismatch is logged.

Two cutting strategies are used, both frame-exact:

| Mode | When | Cost per clip |
| ---- | ---- | ------------- |
| **Keyframe pre-seek** | Constant frame rate sources (the normal case) | Flat — decodes one GOP |
| **Full decode** | Variable frame rate, or no reliable frame rate | Grows with the clip's position in the source |

CFR is established by reading the actual packet timestamps and confirming every
frame sits within half a frame of the ideal grid (a container-index scan, no
decoding — about 3 s on a 150k-frame source). Comparing `r_frame_rate` against
`avg_frame_rate` is *not* sufficient: containers routinely report both as a clean
value like `30/1` for material whose real timestamps are irregular, and acting on
that false positive makes timestamp-based trimming select the wrong frames.

Pre-seek mode fast-seeks to a keyframe before the scene, then selects frames by
their *original* timestamps under `-copyts`. Because input seeking always lands on
a keyframe at or before the target, trimming on absolute PTS yields exactly the same
frames as a full decode — verified bit-identical via `framemd5` — while decoding only
one GOP instead of the whole file from frame 0. This matters enormously on sources
with many cuts: full-decode cost per clip scales with how far into the file the clip
sits (measured on a 10-minute source: 316 ms at 10 s in, 2532 ms at 590 s in),
whereas pre-seek stays flat at ~480 ms anywhere in the file. On a feature-length
source with a few thousand cuts that is the difference between hours and minutes.

If a pre-seek clip ever comes out the wrong length, that scene is automatically
rebuilt with full-decode frame indexing, so exactness never depends on the seek.

Measured on a 1080p30 source cut into 59 scenes (32-core machine), all three
configurations producing bit-identical clips:

| Pipeline | Wall | CPU busy |
| -------- | ---- | -------- |
| Pre-seek + NVDEC (current) | 10.9 s | 6.6 s — 0.6 cores |
| Pre-seek + CPU decode | 11.6 s | 8.4 s — 0.7 cores |
| Full decode per scene (previous behaviour) | 28.7 s | **229 s — 8.0 cores** |

The old full-decode-per-scene approach is what saturated CPUs during splitting; it
re-decoded the source from frame 0 for every single clip.

---

## 📁 Output Formats

| Format          | Extension | Description                                 |
| --------------- | --------- | ------------------------------------------- |
| **CSV**         | `.csv`    | Scene list with timecodes and frame numbers |
| **HTML**        | `.html`   | Visual report with scene table              |
| **SC File**     | `.sc`     | DaVinci Resolve scene cut format            |
| **Images**      | `.jpg`    | Thumbnail frames per scene                  |
| **Video Clips** | `.mp4`    | Frame-exact 10-bit HEVC clip per scene (NVENC) |

---

## 🧠 How It Works

### AutoShot, TransNetV2 and OmniShotCut Detection

All three are deep learning models trained for shot boundary detection. Use the detector selector in the GUI to switch between them based on your content and performance needs.

- **AutoShot / TransNetV2** score every frame for how likely it is to be a cut, and the `threshold` controls sensitivity.
- **OmniShotCut** is a Shot-Query Transformer that predicts shot *ranges* directly and classifies each one (General, Dissolve, Wipes, Push, Slide, Zoom, Fade, Doorway). Because boundaries come from an argmax over queries, there is no threshold to tune — use `mode` to decide whether transitions become their own segments. It is trained on diverse footage (anime, vlogs, gaming, sports, screen recordings), so it is a good first choice for stylised content where the other two over- or under-segment.

OmniShotCut is vendored in `OmniShotCut/` rather than pip-installed, because the upstream
`requirements.txt` pins `transformers==4.57.3`, which would downgrade the version the
DINOv3 validation depends on. The vendored copy carries three marked local edits:
the ResNet backbone is built with `pretrained=False` (the checkpoint overwrites every
backbone weight anyway, so the download is wasted and breaks offline loading), and the
two hardcoded `.to("cuda")` calls are replaced so the `device` selector works.

The GUI decodes to a memory-mapped temp file and drives inference window-by-window
instead of calling `model.inference()`, which would hold the whole video in RAM
(~36 KB/frame, so roughly 4 GB per hour of 30fps footage) with no progress or abort.
Output is bit-identical to the upstream path.

### DINOv3 AI Validation

The optional validation step uses DINOv3 to:

- Sample frames before and after each detected cut
- Compute CLS and dense patch embeddings, plus temporal novelty at the boundary
- Filter out false positives (flashes, fast motion, near-black frames) with optional SSCD
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
- [FFmpeg](https://ffmpeg.org/) — Video processing backend
