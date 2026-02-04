import threading
import queue
import os
import sys
import logging
import subprocess
import json
import csv
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from tooltips import TOOLTIPS

# ------------------------------------------------------------------ #
# Stylized, verbose console logging
# ------------------------------------------------------------------ #
class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[36m", logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m", logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[41m",
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, "")
        base = super().format(record)
        return f"{color}{base}{self.RESET}"

def _setup_logging():
    root = logging.getLogger()
    if root.handlers:
        for h in list(root.handlers): root.removeHandler(h)
    root.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(stream=sys.stdout)
    fmt = "[%(asctime)s] %(levelname)-8s | %(name)s | %(message)s"
    handler.setFormatter(ColorFormatter(fmt, datefmt="%H:%M:%S"))
    root.addHandler(handler)

_setup_logging()
logger = logging.getLogger("SceneDetectGUI")

# ANSI color helpers for console output
KEEP_FG, MERG_FG, BOLD, RESET = "\033[94m", "\033[92m", "\033[1m", "\033[0m"

# ------------------------------------------------------------------ #
# Torch dependency
# ------------------------------------------------------------------ #
TORCH_IMPORT_ERROR = None
try:
    import torch  # noqa: F401
except Exception as e:
    torch = None
    TORCH_IMPORT_ERROR = e
    logger.error("Failed to import torch. Please ensure PyTorch is installed: %s", e)

# ------------------------------------------------------------------ #
# AutoShot dependency
# ------------------------------------------------------------------ #
AUTO_SHOT_IMPORT_ERROR = None
AutoShotNet = None
try:
    if torch is None:
        raise RuntimeError("torch is not available")
    AUTO_SHOT_DIR = Path(__file__).resolve().parent / "AutoShot"
    if AUTO_SHOT_DIR.exists():
        sys.path.insert(0, str(AUTO_SHOT_DIR))
    from supernet_flattransf_3_8_8_8_13_12_0_16_60 import TransNetV2Supernet as AutoShotNet
except Exception as e:
    AUTO_SHOT_IMPORT_ERROR = e
    logger.error(
        "Failed to import AutoShot model. Ensure AutoShot repo is present and 'einops' is installed: %s",
        e,
    )

# ------------------------------------------------------------------ #
# TransNetV2 dependency
# ------------------------------------------------------------------ #
TRANSNET_IMPORT_ERROR = None
TransNetV2 = None
try:
    if torch is None:
        raise RuntimeError("torch is not available")
    from transnetv2_pytorch import TransNetV2
except Exception as e:
    TRANSNET_IMPORT_ERROR = e
    logger.error(
        "Failed to import TransNetV2 (transnetv2-pytorch). Please ensure it is installed: %s",
        e,
    )

# ------------------------------------------------------------------ #
# Minimal timecode abstraction (minimal timecode abstraction for this GUI)
# ------------------------------------------------------------------ #
@dataclass(frozen=True)
class Timecode:
    """Frame-based timecode with helpers compatible with this GUI."""
    frame: int
    fps: float

    def get_frames(self) -> int:
        return int(self.frame)

    def get_seconds(self) -> float:
        if not self.fps:
            return 0.0
        return float(self.frame) / float(self.fps)


def _format_hhmmss_ms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    ms = int(round((seconds - int(seconds)) * 1000.0))
    total = int(seconds)
    s = total % 60
    m = (total // 60) % 60
    h = total // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _parse_fraction(frac: str) -> float:
    """Parse ffprobe r_frame_rate / avg_frame_rate (e.g., '30000/1001')."""
    try:
        if not frac:
            return 0.0
        if "/" in frac:
            num, den = frac.split("/", 1)
            num_f = float(num)
            den_f = float(den)
            return num_f / den_f if den_f else 0.0
        return float(frac)
    except Exception:
        return 0.0


def get_video_info(video_path: str) -> tuple[float, int]:
    """Return (fps, total_frames) with OpenCV and ffprobe fallback."""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if fps > 0.0 and total_frames > 0:
            return fps, total_frames
    except Exception:
        pass

    # ffprobe fallback (more reliable for some formats)
    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate,r_frame_rate,nb_frames,duration",
                "-of",
                "json",
                video_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(probe.stdout)
        stream = (data.get("streams") or [{}])[0]
        fps = _parse_fraction(stream.get("avg_frame_rate") or "") or _parse_fraction(stream.get("r_frame_rate") or "")
        nb = stream.get("nb_frames")
        total_frames = int(nb) if nb and str(nb).isdigit() else 0
        if total_frames <= 0:
            dur = float(stream.get("duration") or 0.0)
            if fps > 0.0 and dur > 0.0:
                total_frames = int(round(dur * fps))
        if fps > 0.0 and total_frames > 0:
            return fps, total_frames
    except Exception:
        pass

    raise RuntimeError("Could not determine FPS / frame count for the selected video.")


def short_scene_guard_frames(fps: float, guard_seconds: float = 0.30) -> int:
    """Return frame length below which cuts are preserved (default 0.30s)."""
    if fps <= 0:
        return 8
    return max(2, int(round(float(fps) * float(guard_seconds))))


def transnetv2_scenes_to_timecodes(
    tn_scenes: list,
    fps: float,
    total_frames: int,
) -> list[tuple[Timecode, Timecode]]:
    """Convert TransNetV2 output dicts into [(start_tc, end_tc_excl), ...]."""
    scenes_out: list[tuple[Timecode, Timecode]] = []

    last_end = 0
    for s in tn_scenes:
        if isinstance(s, dict) and "start_frame" in s and "end_frame" in s:
            start_f = int(s["start_frame"])
            end_f = int(s["end_frame"])
        else:
            start_t = float(s.get("start_time", 0.0)) if isinstance(s, dict) else 0.0
            end_t = float(s.get("end_time", 0.0)) if isinstance(s, dict) else 0.0
            start_f = int(round(start_t * fps))
            end_f = int(round(end_t * fps))

        start_f = max(0, min(start_f, total_frames))
        end_f = max(start_f + 1, min(end_f, total_frames))

        # Make monotonic / non-overlapping
        start_f = max(start_f, last_end)
        end_f = max(end_f, start_f + 1)
        last_end = end_f

        scenes_out.append((Timecode(start_f, fps), Timecode(end_f, fps)))

    # If TransNet didn't cover the tail, extend last scene
    if scenes_out and scenes_out[-1][1].get_frames() < total_frames:
        scenes_out[-1] = (scenes_out[-1][0], Timecode(total_frames, fps))

    # If it returned nothing, create a single full-length scene
    if not scenes_out and total_frames > 0:
        scenes_out = [(Timecode(0, fps), Timecode(total_frames, fps))]

    return scenes_out


def _autoshot_select_device(requested: str) -> str:
    if requested in {"auto", "", None}:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    if requested == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise RuntimeError("MPS requested but not available.")
    return str(requested)


def _autoshot_load_model(weights_path: str, device: str):
    if torch is None:
        raise RuntimeError(f"PyTorch is unavailable: {TORCH_IMPORT_ERROR}")
    if AutoShotNet is None:
        raise RuntimeError(f"AutoShot is unavailable: {AUTO_SHOT_IMPORT_ERROR}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"AutoShot weights file not found: {weights_path}")

    device = _autoshot_select_device(device)
    model = AutoShotNet().eval()

    ckpt = torch.load(weights_path, map_location=device)
    if isinstance(ckpt, dict) and "net" in ckpt:
        ckpt = ckpt["net"]

    model_dict = model.state_dict()
    pretrained = {k: v for k, v in ckpt.items() if k in model_dict}
    if not pretrained:
        raise RuntimeError("AutoShot checkpoint did not match model parameters.")
    model_dict.update(pretrained)
    model.load_state_dict(model_dict)
    model.to(device)
    model.eval()
    return model, device


def _autoshot_get_frames(video_path: str, width: int = 48, height: int = 27):
    import numpy as np

    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        video_path,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, check=False, creationflags=creationflags)
    if result.returncode != 0:
        err = (result.stderr or b"").decode("utf-8", errors="ignore")
        raise RuntimeError(f"FFmpeg failed to decode frames: {err.strip()}")

    raw = result.stdout or b""
    frame_size = width * height * 3
    if len(raw) < frame_size:
        raise RuntimeError("FFmpeg returned no frames for AutoShot inference.")

    frame_count = len(raw) // frame_size
    raw = raw[: frame_count * frame_size]
    frames = np.frombuffer(raw, np.uint8).reshape((frame_count, height, width, 3))
    return frames


def _autoshot_get_batches(frames):
    import numpy as np

    remainder = 50 - len(frames) % 50
    if remainder == 50:
        remainder = 0
    frames = np.concatenate([frames[:1]] * 25 + [frames] + [frames[-1:]] * (remainder + 25), 0)

    for i in range(0, len(frames) - 50, 50):
        yield frames[i : i + 100]


def _autoshot_predictions_to_scenes(predictions):
    import numpy as np

    pred = np.asarray(predictions).reshape(-1)
    pred = pred.astype(np.uint8)
    scenes = []
    t_prev, start = 0, 0
    t = 0
    for i, t in enumerate(pred):
        t = int(t)
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        t_prev = t

    if len(pred) > 0 and t == 0:
        scenes.append([start, len(pred) - 1])

    if len(scenes) == 0 and len(pred) > 0:
        scenes = [[0, len(pred) - 1]]

    return np.array(scenes, dtype=np.int32)


def autoshot_predictions_to_timecodes(
    predictions,
    fps: float,
    total_frames: int,
    threshold: float,
) -> list[tuple[Timecode, Timecode]]:
    import numpy as np

    pred = np.asarray(predictions).reshape(-1)
    if total_frames <= 0:
        total_frames = len(pred)
    if len(pred) < total_frames:
        pred = np.pad(pred, (0, total_frames - len(pred)), mode="constant", constant_values=0)
    pred = pred[:total_frames]

    binary = (pred > float(threshold)).astype(np.uint8)
    scenes = _autoshot_predictions_to_scenes(binary)

    scenes_out: list[tuple[Timecode, Timecode]] = []
    last_end = 0
    for start_f, end_f_incl in scenes:
        start_f = int(start_f)
        end_f_excl = int(end_f_incl) + 1

        start_f = max(0, min(start_f, total_frames))
        end_f_excl = max(start_f + 1, min(end_f_excl, total_frames))

        start_f = max(start_f, last_end)
        end_f_excl = max(end_f_excl, start_f + 1)
        last_end = end_f_excl

        scenes_out.append((Timecode(start_f, fps), Timecode(end_f_excl, fps)))

    if scenes_out and scenes_out[-1][1].get_frames() < total_frames:
        scenes_out[-1] = (scenes_out[-1][0], Timecode(total_frames, fps))

    if not scenes_out and total_frames > 0:
        scenes_out = [(Timecode(0, fps), Timecode(total_frames, fps))]

    return scenes_out


def _autoshot_predict_from_frames(frames, model, device, abort_flag=None, progress_cb=None):
    import numpy as np

    if frames is None or len(frames) == 0:
        raise RuntimeError("AutoShot received no frames for inference.")

    num_batches = max(1, (len(frames) + 49) // 50)
    predictions = []

    with torch.no_grad():
        for idx, batch in enumerate(_autoshot_get_batches(frames)):
            if abort_flag is not None and abort_flag.is_set():
                raise InterruptedError

            batch_t = torch.from_numpy(
                batch.transpose((3, 0, 1, 2))[np.newaxis, ...]
            ).float()
            batch_t = batch_t.to(device)

            one_hot = model(batch_t)
            if isinstance(one_hot, tuple):
                one_hot = one_hot[0]
            one_hot = torch.sigmoid(one_hot[0]).squeeze(-1)

            predictions.append(one_hot[25:75].detach().cpu().numpy())

            if progress_cb:
                progress_cb((idx + 1) / num_batches, f"AutoShot inference ({idx + 1}/{num_batches})")

    return np.concatenate(predictions, 0)[: len(frames)]

class SceneDetectApp:
    """Main application class encapsulating the GUI and logic."""

    CONFIG_FILE = "config.json"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Scene Cut Detection")
        self.root.geometry("820x720")

        self.progress_queue: "queue.Queue[Tuple[float, str]]" = queue.Queue()
        self.abort_flag = threading.Event()
        self.detected_scenes: Optional[List[Tuple[Timecode, Timecode]]] = None
        self.total_frames: int = 0

        # --- Input & Output Variables ---
        self.video_path_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()

        # --- Detector & AI Variables ---
        self.detector_type_var = tk.StringVar(value="AutoShot")
        self.params_vars: dict = {}
        self.detector_params_cache: dict[str, dict[str, str]] = {
            "AutoShot": self._default_detector_params("AutoShot"),
            "TransNetV2": self._default_detector_params("TransNetV2"),
        }
        self._last_detector_type = self.detector_type_var.get()
        self.ai_validate_var = tk.BooleanVar(value=False)
        self.ai_window_var = tk.IntVar(value=5)
        self.flash_sensitivity_var = tk.IntVar(value=15)  # Luma delta threshold for flash detection
        self.refine_pyscenedetect_var = tk.BooleanVar(value=False)
        self.refine_snap_var = tk.IntVar(value=6)
        self.refine_threshold_var = tk.DoubleVar(value=27.0)

        # --- Output Action Variables ---
        self.export_csv_var = tk.BooleanVar(value=False)
        self.export_html_var = tk.BooleanVar(value=False)
        self.export_sc_var = tk.BooleanVar(value=False)
        self.save_images_var = tk.BooleanVar(value=False)
        self.split_video_var = tk.BooleanVar(value=False)
        self.num_images_var = tk.IntVar(value=3)
        self.frame_margin_var = tk.IntVar(value=1)

        # --- FFmpeg Variables ---
        self.ffmpeg_codec_var = tk.StringVar(value="h264_nvenc")
        self.ffmpeg_preset_var = tk.StringVar(value="p7")
        self.ffmpeg_cq_var = tk.IntVar(value=16)

        self._create_widgets()
        self._load_settings()
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.root.after(200, self._process_queues)
        logger.info("GUI Initialized.")

    class _ToolTip:
        def __init__(self, widget, text):
            self.widget, self.text, self.tipwindow = widget, text, None
            widget.bind("<Enter>", self._show_tip)
            widget.bind("<Leave>", self._hide_tip)
        def _show_tip(self, event=None):
            if self.tipwindow or not self.text: return
            x, y = self.widget.winfo_rootx() + 20, self.widget.winfo_rooty() + self.widget.winfo_height() + 10
            self.tipwindow = tw = tk.Toplevel(self.widget)
            tw.wm_overrideredirect(True); tw.wm_geometry(f"+{x}+{y}")
            label = tk.Label(tw, text=self.text, justify=tk.LEFT, background="#ffffe0", relief=tk.SOLID, borderwidth=1, font=("tahoma", "8", "normal"))
            label.pack(ipadx=1)
        def _hide_tip(self, event=None):
            if self.tipwindow: self.tipwindow.destroy()
            self.tipwindow = None
    def _attach_tooltip(self, widget, key):
        if text := TOOLTIPS.get(key): self._ToolTip(widget, text)

    def _create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- IO Frame ---
        io_frame = ttk.LabelFrame(main_frame, text="1. Input & Output")
        io_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(io_frame, text="Video File:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        video_entry = ttk.Entry(io_frame, textvariable=self.video_path_var)
        video_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self._attach_tooltip(video_entry, "video_path")
        ttk.Button(io_frame, text="Browse...", command=self.browse_video).grid(row=0, column=2, padx=5, pady=5)
        ttk.Label(io_frame, text="Output Folder:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        output_entry = ttk.Entry(io_frame, textvariable=self.output_dir_var)
        output_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self._attach_tooltip(output_entry, "output_dir")
        ttk.Button(io_frame, text="Browse...", command=self.browse_output_dir).grid(row=1, column=2, padx=5, pady=5)
        io_frame.columnconfigure(1, weight=1)

        # --- Config Frame ---
        config_frame = ttk.LabelFrame(main_frame, text="2. Detection Configuration")
        config_frame.pack(fill=tk.X, padx=5, pady=5, ipady=5)
        config_frame.columnconfigure(0, weight=2) # Detector params take more space
        config_frame.columnconfigure(1, weight=1)

        detector_row = ttk.Frame(config_frame)
        detector_row.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Label(detector_row, text="Detector:").pack(side=tk.LEFT)
        detector_combo = ttk.Combobox(
            detector_row,
            textvariable=self.detector_type_var,
            values=["AutoShot", "TransNetV2"],
            state="readonly",
            width=14,
        )
        detector_combo.pack(side=tk.LEFT, padx=(10, 0))
        detector_combo.bind("<<ComboboxSelected>>", self._on_detector_change)
        self._attach_tooltip(detector_combo, "detector_type")
        
        self.params_frame = ttk.Frame(config_frame)
        self.params_frame.grid(row=1, column=0, sticky="nsew", padx=5)
        
        # --- FFmpeg Settings Frame ---
        ffmpeg_frame = ttk.LabelFrame(config_frame, text="FFmpeg Output")
        ffmpeg_frame.grid(row=1, column=1, sticky="ns", padx=(10, 5), pady=2)

        ttk.Label(ffmpeg_frame, text="Codec:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        codec_combo = ttk.Combobox(ffmpeg_frame, textvariable=self.ffmpeg_codec_var, values=["h264_nvenc", "hevc_nvenc", "libx264"], state="readonly", width=12)
        codec_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self._attach_tooltip(codec_combo, "ffmpeg_codec")
        
        ttk.Label(ffmpeg_frame, text="Preset:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        presets = [f"p{i}" for i in range(1, 8)]
        preset_combo = ttk.Combobox(ffmpeg_frame, textvariable=self.ffmpeg_preset_var, values=presets, state="readonly", width=12)
        preset_combo.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self._attach_tooltip(preset_combo, "ffmpeg_preset")

        ttk.Label(ffmpeg_frame, text="CQ Level:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        cq_spin = ttk.Spinbox(ffmpeg_frame, from_=16, to=30, textvariable=self.ffmpeg_cq_var, width=13)
        cq_spin.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self._attach_tooltip(cq_spin, "ffmpeg_cq")

        self._build_detector_params() # Initial build

        # --- AI Validation Frame ---
        ai_frame = ttk.LabelFrame(main_frame, text="Optional: AI Validation")
        ai_frame.pack(fill=tk.X, padx=5, pady=5)
        ai_check = ttk.Checkbutton(ai_frame, text="Validate cuts with DINOv3/SSCD (filters flashes/fast motion)", variable=self.ai_validate_var)
        ai_check.grid(row=0, column=0, columnspan=3, padx=5, pady=5, sticky="w")
        self._attach_tooltip(ai_check, "ai_validate")
        ttk.Label(ai_frame, text="Validation Window (frames):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ai_spin = ttk.Spinbox(ai_frame, from_=2, to=10, textvariable=self.ai_window_var, width=5)
        ai_spin.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self._attach_tooltip(ai_spin, "ai_window")
        
        ttk.Label(ai_frame, text="Flash Sensitivity:").grid(row=1, column=2, padx=(20, 5), pady=5, sticky="w")
        flash_spin = ttk.Spinbox(ai_frame, from_=10, to=80, textvariable=self.flash_sensitivity_var, width=5)
        flash_spin.grid(row=1, column=3, padx=5, pady=5, sticky="w")
        self._attach_tooltip(flash_spin, "flash_sensitivity")

        # --- Refinement Frame ---
        refine_frame = ttk.LabelFrame(main_frame, text="Optional: PySceneDetect Refinement")
        refine_frame.pack(fill=tk.X, padx=5, pady=5)
        refine_check = ttk.Checkbutton(
            refine_frame,
            text="Refine cuts with PySceneDetect (ContentDetector)",
            variable=self.refine_pyscenedetect_var,
        )
        refine_check.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        self._attach_tooltip(refine_check, "refine_pyscenedetect")
        ttk.Label(refine_frame, text="Snap Window (frames):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        refine_snap = ttk.Spinbox(refine_frame, from_=0, to=30, textvariable=self.refine_snap_var, width=5)
        refine_snap.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self._attach_tooltip(refine_snap, "refine_snap")
        ttk.Label(refine_frame, text="Content Threshold:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        refine_thr = ttk.Spinbox(refine_frame, from_=5, to=80, increment=1, textvariable=self.refine_threshold_var, width=5)
        refine_thr.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self._attach_tooltip(refine_thr, "refine_threshold")

        # --- Output Actions Frame ---
        output_frame = ttk.LabelFrame(main_frame, text="3. Output Actions")
        output_frame.pack(fill=tk.X, padx=5, pady=5, ipady=5)
        # Checkboxes
        c1 = ttk.Checkbutton(output_frame, text="Export scene list to CSV", variable=self.export_csv_var); c1.grid(row=0, column=0, sticky="w", padx=5, pady=2)
        c2 = ttk.Checkbutton(output_frame, text="Export scene list to HTML", variable=self.export_html_var); c2.grid(row=1, column=0, sticky="w", padx=5, pady=2)
        c3 = ttk.Checkbutton(output_frame, text="Export to .sc file", variable=self.export_sc_var); c3.grid(row=2, column=0, sticky="w", padx=5, pady=2)
        c5 = ttk.Checkbutton(output_frame, text="Save scene thumbnails", variable=self.save_images_var); c5.grid(row=0, column=1, sticky="w", padx=20, pady=2)
        c6 = ttk.Checkbutton(output_frame, text="Split video into scenes (FFmpeg)", variable=self.split_video_var); c6.grid(row=1, column=1, sticky="w", padx=20, pady=2)
        self._attach_tooltip(c1, "export_csv"); self._attach_tooltip(c2, "export_html"); self._attach_tooltip(c3, "export_sc")
        self._attach_tooltip(c5, "save_images"); self._attach_tooltip(c6, "split_ffmpeg")
        # Image saving options
        img_opts_frame = ttk.Frame(output_frame)
        img_opts_frame.grid(row=0, column=2, sticky="w", padx=5)
        ttk.Label(img_opts_frame, text="Images:").pack(side=tk.LEFT)
        num_images_spin = ttk.Spinbox(img_opts_frame, from_=1, to=10, textvariable=self.num_images_var, width=5)
        num_images_spin.pack(side=tk.LEFT); self._attach_tooltip(num_images_spin, "num_images")
        ttk.Label(img_opts_frame, text="Margin:").pack(side=tk.LEFT, padx=(10,0))
        frame_margin_spin = ttk.Spinbox(img_opts_frame, from_=0, to=30, textvariable=self.frame_margin_var, width=5)
        frame_margin_spin.pack(side=tk.LEFT); self._attach_tooltip(frame_margin_spin, "frame_margin")
        output_frame.columnconfigure(1, weight=1)

        # --- Run Frame ---
        run_frame = ttk.LabelFrame(main_frame, text="4. Run Process")
        run_frame.pack(fill=tk.X, padx=5, pady=5)
        self.start_button = ttk.Button(run_frame, text="Start Processing", command=self._start_detection)
        self.start_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self._attach_tooltip(self.start_button, "start_detection")
        self.abort_button = ttk.Button(run_frame, text="Abort", command=self._abort_detection, state=tk.DISABLED)
        self.abort_button.grid(row=0, column=1, padx=5, pady=5)
        self.progress_label = ttk.Label(run_frame, text="Ready")
        self.progress_label.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(run_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        run_frame.columnconfigure(0, weight=1)

    def _on_closing(self):
        """Handle the window close event."""
        logger.info("Closing application and saving settings.")
        self._save_settings()
        self.root.destroy()

    def _default_detector_params(self, detector_type: str) -> dict[str, str]:
        threshold = "0.296" if detector_type == "AutoShot" else "0.3"
        return {
            "device": "auto",
            "threshold": threshold,
            "min_scene_len": "8",
        }

    def _cache_detector_params(self, detector_type: str | None) -> None:
        if not detector_type or not self.params_vars:
            return
        self.detector_params_cache[detector_type] = {k: v.get() for k, v in self.params_vars.items()}

    def _save_settings(self):
        """Save all GUI settings to a JSON file."""
        self._cache_detector_params(self.detector_type_var.get())
        settings = {
            "video_path": self.video_path_var.get(),
            "output_dir": self.output_dir_var.get(),
            "detector_type": self.detector_type_var.get(),
            "detector_params": self.detector_params_cache,
            "ai_validate": self.ai_validate_var.get(),
            "ai_window": self.ai_window_var.get(),
            "flash_sensitivity": self.flash_sensitivity_var.get(),
            "refine_pyscenedetect": self.refine_pyscenedetect_var.get(),
            "refine_snap": self.refine_snap_var.get(),
            "refine_threshold": self.refine_threshold_var.get(),
            "export_csv": self.export_csv_var.get(),
            "export_html": self.export_html_var.get(),
            "export_sc": self.export_sc_var.get(),
            "save_images": self.save_images_var.get(),
            "split_video": self.split_video_var.get(),
            "num_images": self.num_images_var.get(),
            "frame_margin": self.frame_margin_var.get(),
            "ffmpeg_codec": self.ffmpeg_codec_var.get(),
            "ffmpeg_preset": self.ffmpeg_preset_var.get(),
            "ffmpeg_cq": self.ffmpeg_cq_var.get(),
        }
        try:
            with open(self.CONFIG_FILE, 'w') as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save settings to {self.CONFIG_FILE}: {e}")

    def _load_settings(self):
        """Load GUI settings from a JSON file if it exists."""
        if not os.path.exists(self.CONFIG_FILE):
            logger.info(f"Configuration file not found at {self.CONFIG_FILE}. Using defaults.")
            return
        try:
            with open(self.CONFIG_FILE, 'r') as f:
                settings = json.load(f)
            
            self.video_path_var.set(settings.get("video_path", ""))
            self.output_dir_var.set(settings.get("output_dir", ""))
            
            self.ai_validate_var.set(settings.get("ai_validate", False))
            self.ai_window_var.set(settings.get("ai_window", 3))
            self.flash_sensitivity_var.set(settings.get("flash_sensitivity", 15))
            self.refine_pyscenedetect_var.set(settings.get("refine_pyscenedetect", False))
            self.refine_snap_var.set(settings.get("refine_snap", 6))
            self.refine_threshold_var.set(settings.get("refine_threshold", 27.0))
            self.export_csv_var.set(settings.get("export_csv", False))
            self.export_html_var.set(settings.get("export_html", False))
            self.export_sc_var.set(settings.get("export_sc", False))
            self.save_images_var.set(settings.get("save_images", False))
            self.split_video_var.set(settings.get("split_video", False))
            self.num_images_var.set(settings.get("num_images", 3))
            self.frame_margin_var.set(settings.get("frame_margin", 1))
            self.ffmpeg_codec_var.set(settings.get("ffmpeg_codec", "h264_nvenc"))
            self.ffmpeg_preset_var.set(settings.get("ffmpeg_preset", "p7"))
            self.ffmpeg_cq_var.set(settings.get("ffmpeg_cq", 16))
            
            # Detector type & per-detector params
            detector_type = settings.get("detector_type", "AutoShot")
            if detector_type not in {"AutoShot", "TransNetV2"}:
                detector_type = "AutoShot"

            loaded_params = settings.get("detector_params")
            if isinstance(loaded_params, dict):
                if any(k in loaded_params for k in ("AutoShot", "TransNetV2")):
                    for det in ("AutoShot", "TransNetV2"):
                        det_params = loaded_params.get(det)
                        if isinstance(det_params, dict):
                            self.detector_params_cache[det].update(det_params)
                else:
                    # Legacy flat params -> apply to selected detector
                    self.detector_params_cache[detector_type].update(loaded_params)

            self.detector_type_var.set(detector_type)
            self._build_detector_params() # Ensure params_vars is populated

            logger.info(f"Successfully loaded settings from {self.CONFIG_FILE}.")
        except Exception as e:
            logger.error(f"Failed to load or apply settings from {self.CONFIG_FILE}: {e}")

    def browse_video(self):
        if filename := filedialog.askopenfilename(title="Select video file", filetypes=[("Video files", "*.mp4;*.mkv;*.avi;*.mov"), ("All files", "*.*")]):
            self.video_path_var.set(filename)
            base_name = Path(filename).stem
            output_path = Path(filename).parent / base_name
            self.output_dir_var.set(str(output_path))
            logger.info("Selected video: %s", filename)
            logger.info("Set default output folder: %s", output_path)

    def browse_output_dir(self):
        if directory := filedialog.askdirectory(title="Select output directory"):
            self.output_dir_var.set(directory)

    def _on_detector_change(self, event=None):
        prev_detector = getattr(self, "_last_detector_type", None)
        self._cache_detector_params(prev_detector)
        self._build_detector_params()
        return

    def _build_detector_params(self):
        """Build detector parameter widgets."""
        for widget in self.params_frame.winfo_children():
            widget.destroy()

        # Clear and rebuild vars
        self.params_vars.clear()

        detector_type = self.detector_type_var.get()
        params = self.detector_params_cache.get(detector_type, self._default_detector_params(detector_type))
        default_threshold = "0.296" if detector_type == "AutoShot" else "0.3"

        # Row 0: device
        ttk.Label(self.params_frame, text="device:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        d_var = tk.StringVar(value=str(params.get("device", "auto")))
        d_combo = ttk.Combobox(self.params_frame, textvariable=d_var, values=["auto", "cpu", "cuda", "mps"], state="readonly", width=10)
        d_combo.grid(row=0, column=1, padx=5, pady=2, sticky="w")
        self._attach_tooltip(d_combo, "device")

        # Row 1: threshold
        ttk.Label(self.params_frame, text="threshold:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        t_var = tk.StringVar(value=str(params.get("threshold", default_threshold)))
        t_entry = ttk.Entry(self.params_frame, textvariable=t_var, width=10)
        t_entry.grid(row=1, column=1, padx=5, pady=2, sticky="w")
        self._attach_tooltip(t_entry, "threshold")

        # Row 2: min_scene_len
        ttk.Label(self.params_frame, text="min_scene_len (frames):").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        m_var = tk.StringVar(value=str(params.get("min_scene_len", "8")))
        m_entry = ttk.Entry(self.params_frame, textvariable=m_var, width=10)
        m_entry.grid(row=2, column=1, padx=5, pady=2, sticky="w")
        self._attach_tooltip(m_entry, "min_scene_len")

        # Save vars
        self.params_vars["device"] = d_var
        self.params_vars["threshold"] = t_var
        self.params_vars["min_scene_len"] = m_var
        self._last_detector_type = detector_type


    def _start_detection(self) -> None:
        video_path = self.video_path_var.get()
        output_dir = self.output_dir_var.get()
        if not video_path or not output_dir:
            messagebox.showerror("Error", "Please select a video file and an output folder.")
            return
        detector_type = self.detector_type_var.get()
        if detector_type == "AutoShot":
            if AutoShotNet is None:
                detail = f"\n\nImport error: {AUTO_SHOT_IMPORT_ERROR}" if AUTO_SHOT_IMPORT_ERROR else ""
                messagebox.showerror(
                    "Error",
                    "AutoShot is not available. Ensure the AutoShot repo exists and install 'einops'."
                    + detail,
                )
                return
            default_weights = "./weights/ckpt_0_200_0.pth"
        elif detector_type == "TransNetV2":
            if TransNetV2 is None:
                detail = f"\n\nImport error: {TRANSNET_IMPORT_ERROR}" if TRANSNET_IMPORT_ERROR else ""
                messagebox.showerror(
                    "Error",
                    "TransNetV2 is not available. Ensure 'transnetv2-pytorch' is installed."
                    + detail,
                )
                return
            default_weights = "./weights/transnetv2-pytorch-weights.pth"
        else:
            messagebox.showerror("Error", f"Unknown detector type: {detector_type}")
            return

        try:
            weights_path = default_weights
            device = str(self.params_vars.get("device").get()).strip() or "auto"
            threshold = float(self.params_vars.get("threshold").get())
            min_scene_len = int(float(self.params_vars.get("min_scene_len").get()))
            if not (0.0 < threshold < 1.0):
                raise ValueError("threshold must be between 0 and 1.")
            if min_scene_len < 1:
                min_scene_len = 1
        except Exception as exc:
            logger.error("Invalid %s parameters: %s", detector_type, exc)
            messagebox.showerror("Parameter Error", f"Invalid {detector_type} parameters: {exc}")
            return

        detector_cfg = {
            "type": detector_type,
            "weights_path": weights_path,
            "device": device,
            "threshold": threshold,
            "min_scene_len": min_scene_len,
        }

        self.progress_var.set(0)
        self.progress_label.config(text="Starting...")
        self.start_button.config(state=tk.DISABLED)
        self.abort_button.config(state=tk.NORMAL)
        self.abort_flag.clear()
        self.detected_scenes = None
        logger.info("Starting detection process (%s)...", detector_type)

        threading.Thread(target=self._detection_and_output_task, args=(video_path, output_dir, detector_cfg), daemon=True).start()
    def _detection_and_output_task(self, video_path: str, output_dir: str, detector: dict) -> None:
        try:
            # --- 1. Setup ---
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            self.progress_queue.put((5, "Reading video metadata..."))
            fps, total_frames = get_video_info(video_path)
            self.total_frames = total_frames
            logger.debug("Video FPS: %.6f | total frames: %d", fps, total_frames)

            if self.abort_flag.is_set():
                raise InterruptedError

            # --- 2. Detection ---
            detector_type = detector.get("type", "AutoShot")
            if detector_type == "AutoShot":
                self.progress_queue.put((10, "Loading AutoShot..."))

                weights_path = str(detector.get("weights_path", "")).strip()
                if not weights_path:
                    raise FileNotFoundError("AutoShot weights_path is empty. Please select a .pth weights file.")

                model, device = _autoshot_load_model(weights_path, detector.get("device", "auto"))

                if self.abort_flag.is_set():
                    raise InterruptedError

                self.progress_queue.put((20, "Decoding video frames (AutoShot)..."))
                frames = _autoshot_get_frames(video_path)
                frame_count = len(frames)
                if frame_count <= 0:
                    raise RuntimeError("AutoShot did not decode any frames.")

                if total_frames <= 0:
                    total_frames = frame_count
                    self.total_frames = total_frames
                elif abs(total_frames - frame_count) > 1:
                    logger.warning(
                        "Frame count mismatch: ffprobe=%d, ffmpeg=%d. Using ffprobe count for outputs.",
                        total_frames,
                        frame_count,
                    )

                if self.abort_flag.is_set():
                    raise InterruptedError

                def progress_cb(pct: float, msg: str):
                    self.progress_queue.put((20 + float(pct) * 40.0, msg))

                predictions = _autoshot_predict_from_frames(
                    frames,
                    model,
                    device,
                    abort_flag=self.abort_flag,
                    progress_cb=progress_cb,
                )
                del frames

                if self.abort_flag.is_set():
                    raise InterruptedError

                self.progress_queue.put((60, f"AutoShot produced {len(predictions)} frame scores. Converting..."))
                scenes = autoshot_predictions_to_timecodes(
                    predictions,
                    fps=fps,
                    total_frames=total_frames,
                    threshold=float(detector.get("threshold", 0.296) or 0.296),
                )

            elif detector_type == "TransNetV2":
                self.progress_queue.put((10, "Loading TransNetV2..."))
                if TransNetV2 is None:
                    raise RuntimeError(f"TransNetV2 is unavailable: {TRANSNET_IMPORT_ERROR}")

                model = TransNetV2(device=detector.get("device", "auto"))
                model.eval()

                weights_path = str(detector.get("weights_path", "")).strip()
                if not weights_path:
                    raise FileNotFoundError("TransNetV2 weights_path is empty. Please select a .pth weights file.")
                if not os.path.exists(weights_path):
                    raise FileNotFoundError(f"TransNetV2 weights file not found: {weights_path}")

                state_dict = torch.load(weights_path, map_location=model.device)
                model.load_state_dict(state_dict)

                if self.abort_flag.is_set():
                    raise InterruptedError

                self.progress_queue.put((20, "Running TransNetV2 inference..."))
                with torch.no_grad():
                    tn_scenes = model.detect_scenes(
                        video_path,
                        threshold=float(detector.get("threshold", 0.3)),
                    )

                if self.abort_flag.is_set():
                    raise InterruptedError

                self.progress_queue.put((60, f"TransNetV2 returned {len(tn_scenes)} scenes. Converting..."))
                scenes = transnetv2_scenes_to_timecodes(tn_scenes, fps=fps, total_frames=total_frames)

            else:
                raise RuntimeError(f"Unknown detector type: {detector_type}")

            # Enforce minimum scene length (frames), but preserve very short scenes by default.
            min_len = int(detector.get("min_scene_len", 1) or 1)
            guard_len = short_scene_guard_frames(fps)
            if min_len > 1 and scenes:
                filtered = []
                for st, et in scenes:
                    scene_len = int(et.get_frames() - st.get_frames())
                    if scene_len < min_len and scene_len > guard_len:
                        continue
                    filtered.append((st, et))
                # Keep at least one
                scenes = filtered or [scenes[0]]

            logger.info("Detected %d raw scenes (%s)", len(scenes), detector_type)

            # --- 3. AI Validation (Optional) ---
            if self.ai_validate_var.get() and scenes:
                self.progress_queue.put((70, "AI validating cuts..."))
                scenes = self._run_ai_validation(video_path, scenes, guard_len)

            # --- 4. PySceneDetect Refinement (Optional) ---
            if self.refine_pyscenedetect_var.get() and scenes:
                self.progress_queue.put((74, "Refining cuts with PySceneDetect..."))
                scenes = self._refine_with_pyscenedetect(video_path, scenes, fps=fps, total_frames=total_frames)

            # --- 5. Ultra-short Scene Merge (Always On) ---
            if scenes:
                scenes = self._merge_ultra_short_scenes(scenes, fps=fps, total_frames=total_frames, max_seconds=0.05)

            self.detected_scenes = scenes
            if not self.detected_scenes:
                logger.warning("No scenes were detected.")
                self.progress_queue.put((100, "No scenes found."))
                return

            # --- 6. Output Generation ---
            self.progress_queue.put((80, "Generating outputs..."))
            base_name = Path(video_path).stem

            output_tasks: List[Tuple[tk.BooleanVar, str, Callable]] = [
                (self.export_csv_var, "CSV", lambda: self._export_csv(Path(output_dir) / f"{base_name}_scenes.csv")),
                (self.export_html_var, "HTML", lambda: self._export_html(Path(output_dir) / f"{base_name}_scenes.html")),
                (self.export_sc_var, "SC", lambda: self._export_sc(fps, total_frames, Path(output_dir) / f"{base_name}.sc")),
                (self.save_images_var, "Images", lambda: self._save_images(video_path, output_dir, fps)),
                (self.split_video_var, "Splitting", lambda: self._split_video(video_path, output_dir, fps)),
            ]

            for var, name, func in output_tasks:
                if var.get():
                    self.progress_queue.put((85, f"Exporting {name}..."))
                    func()

            self.progress_queue.put((100, f"Done! Found {len(self.detected_scenes or [])} scenes."))

        except InterruptedError:
            logger.info("Process aborted by user.")
            self.progress_queue.put((0, "Aborted."))
        except Exception as err:
            logger.exception("Fatal error during task execution.")
            messagebox.showerror("Processing Error", str(err))
            self.progress_queue.put((0, "Error."))

    def _run_ai_validation(self, video_path, scenes, short_guard_frames: int):
        try:
            import cv2
            import torch
            import numpy as np
            from transformers import AutoImageProcessor, AutoModel

            class DinoCutValidator:
                # --- General ---
                DEFAULT_BASE_THRESHOLD = 0.88
                MAX_LARGE_WINDOW = 24

                # SSCD stable-frame settings (used only when SSCD is available)
                SSCD_K = 5             # median of up to 5 frames per side
                SSCD_MIN_K = 3         # need at least 3 frames per side (else fallback)

                # Adjacent-frame discontinuity = strong cut evidence (unless flash is strong)
                ADJ_STRONG_CUT = 0.68

                # --- Flash detection window (frames relative to cut_frame) ---
                FLASH_SCAN = 12         # analyze +/- this many frames around cut
                FLASH_CENTER = 4        # spike must occur within +/- this region
                FLASH_BASE_GAP = 5      # baseline windows start beyond this offset from the cut

                FLASH_MIN_DUR = 1
                FLASH_MAX_DUR = 8

                # --- Pixel cue (HSV H,S histogram intersection in [0,1]) ---
                PIXEL_HIST_MIN = 0.08
                PIXEL_HIST_STRONG = 0.14

                # --- Similarity guardrails ---
                MIN_STABLE_SIM = 0.72   # absolute floor for merging on flash evidence

                # ANSI color helpers for console output (optional)
                KEEP_FG = "\033[94m"
                MERG_FG = "\033[92m"
                BOLD = "\033[1m"
                RESET = "\033[0m"

                def __init__(self, model_dir: str = './weights', device=None, batch_size: int = 48,
                             flash_luma_delta: float = 30.0,
                             enable_sscd: bool = True, sscd_model_path: str = None, sscd_input: int = 288,
                             logger=None):
                    import os
                    import logging
                    import torch
                    from transformers import AutoImageProcessor, AutoModel

                    self._log = logger or logging.getLogger('DinoCutValidator')

                    if not os.path.isdir(model_dir):
                        raise FileNotFoundError(f'DINOv3 model directory not found: {model_dir}')

                    self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
                    if self.device == 'cpu':
                        raise RuntimeError('AI Validation requires a CUDA-enabled GPU and PyTorch.')

                    self._log.debug('Loading DINOv3 model on device: %s', self.device)
                    self.processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
                    self.model = AutoModel.from_pretrained(model_dir, local_files_only=True).to(self.device).eval()

                    # Caches keyed by frame_idx
                    self.cache = {}        # frame_idx -> torch.Tensor embedding (normalized)
                    self.luma_cache = {}   # frame_idx -> float mean luma (0-255)
                    self.vhi_cache = {}    # frame_idx -> float HSV-V p99.5 (0-255), catches sparse lightning
                    self.hist_cache = {}   # frame_idx -> np.ndarray HSV (H,S) hist (L1 normalized)

                    self.batch_size = int(batch_size)
                    self.total_video_frames = 0

                    # User sensitivity knob: smaller -> more sensitive (merge more flash cuts)
                    self.flash_luma_delta = float(flash_luma_delta)

                    # Rely on the HF image processor for rescale + normalize.
                    # ---- Optional SSCD (photometric-invariant descriptor) ----
                    self.sscd_model = None
                    self.sscd_input = int(sscd_input)
                    self._sscd_video_path = None
                    self._sscd_cap = None
                    from collections import OrderedDict
                    self._sscd_frame_cache = OrderedDict()  # frame_idx -> resized RGB uint8
                    self._sscd_frame_cache_max = 128
                    # SSCD hub models typically use ImageNet normalization
                    self._sscd_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
                    self._sscd_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
                    if bool(enable_sscd):
                        try:
                            if sscd_model_path is None:
                                sscd_model_path = os.path.join(model_dir, 'sscd_disc_large.torchscript.pt')
                            self._init_sscd(sscd_model_path)
                            self._log.info('SSCD enabled (torchscript: %s).', sscd_model_path)
                        except Exception as e:
                            self.sscd_model = None
                            self._log.warning('SSCD unavailable (continuing with DINO-only): %s', e)


                def _target_size(self):
                    size = getattr(self.processor, 'size', None) or {}
                    if isinstance(size, dict):
                        h = int(size.get('height') or size.get('shortest_edge') or 224)
                        w = int(size.get('width') or size.get('shortest_edge') or 224)
                    else:
                        h = w = 224
                    return (w, h)  # OpenCV uses (width, height)

                @staticmethod
                def _luma_from_rgb(rgb_uint8) -> float:
                    import numpy as np
                    r = rgb_uint8[:, :, 0].astype(np.float32)
                    g = rgb_uint8[:, :, 1].astype(np.float32)
                    b = rgb_uint8[:, :, 2].astype(np.float32)
                    return float((0.2126 * r + 0.7152 * g + 0.0722 * b).mean())

                @staticmethod
                def _hsv_hist_and_vhi_from_rgb(rgb_uint8):
                    """Return (hist_flat, v_hi_p995)."""
                    import numpy as np
                    import cv2

                    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)

                    # Pixel cue: coarse H,S histogram
                    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
                    hist = hist.astype(np.float32)
                    s = float(hist.sum())
                    if s > 1e-6:
                        hist /= s
                    hist_flat = hist.reshape(-1)

                    v = hsv[:, :, 2].astype(np.float32)
                    v_hi = float(np.percentile(v, 99.5))
                    return hist_flat, v_hi

                @staticmethod
                def _hist_intersection(h1, h2) -> float:
                    import numpy as np
                    if h1 is None or h2 is None:
                        return float('nan')
                    return float(np.minimum(h1, h2).sum())

                def _flash_signal(self, frame_idx: int):
                    """Single scalar flash signal per frame, designed to catch sparse lightning.

                    We use max(mean_luma, V_p99.5). V_p99.5 catches lightning bolts that occupy few pixels.
                    """
                    l = self.luma_cache.get(frame_idx)
                    v = self.vhi_cache.get(frame_idx)
                    if l is None and v is None:
                        return None
                    if l is None:
                        return float(v)
                    if v is None:
                        return float(l)
                    return float(max(float(l), float(v)))

                def _embed_batch(self, pixel_values):
                    import torch
                    with torch.inference_mode():
                        outputs = self.model(pixel_values=pixel_values)
                        pooled = getattr(outputs, 'pooler_output', None)
                        if pooled is None:
                            # Fallback to CLS token if pooler_output is unavailable.
                            pooled = outputs.last_hidden_state[:, 0]
                        return torch.nn.functional.normalize(pooled, dim=-1)

                # ---- SSCD helpers (optional) ----
                def _init_sscd(self, model_path: str):
                    """Load SSCD TorchScript model from disk.

                    SSCD is a photometric-robust descriptor for copy / near-duplicate detection.
                    Here it is used only as a *flash-robust same-shot* signal by comparing
                    median-composited stable frames (3-5 frames per side) across the boundary.

                    The common SSCD preprocessing is:
                      - resize to 288x288
                      - ToTensor() in [0,1]
                      - ImageNet normalize (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
                    """
                    import os
                    import torch

                    p = str(model_path or '')
                    if not p:
                        raise ValueError('SSCD model_path is empty.')
                    # Resolve relative paths against the current working directory.
                    if not os.path.isabs(p):
                        p = os.path.normpath(p)
                    if not os.path.exists(p):
                        # Best-effort fallback: try alongside the DINO weights directory.
                        alt = os.path.join('./weights', os.path.basename(p))
                        if os.path.exists(alt):
                            p = alt
                        else:
                            raise FileNotFoundError(f'SSCD TorchScript file not found: {model_path}')

                    self._log.info('Loading SSCD TorchScript model: %s', p)
                    m = torch.jit.load(p, map_location=self.device)
                    try:
                        m = m.to(self.device)
                    except Exception:
                        pass
                    m.eval()
                    self.sscd_model = m

                def _sscd_required_sim(self, motion: float) -> float:
                    """Dynamic SSCD threshold driven by the GUI Flash Sensitivity knob.

                    Lower Flash Sensitivity => more aggressive flash merging => lower required SSCD sim.
                    """
                    sens = float(max(15.0, min(80.0, getattr(self, 'flash_luma_delta', 30.0))))
                    # sens=15 -> ~0.68 ; sens=80 -> ~0.78
                    base = 0.78 - 0.0015 * (80.0 - sens)
                    base = base - 0.05 * float(max(0.0, min(1.0, motion)))
                    return float(max(0.62, min(0.84, base)))

                def _sscd_get_cap(self, video_path: str):
                    import cv2
                    if (self._sscd_cap is None) or (self._sscd_video_path != video_path):
                        try:
                            if self._sscd_cap is not None:
                                self._sscd_cap.release()
                        except Exception:
                            pass
                        self._sscd_video_path = video_path
                        self._sscd_cap = cv2.VideoCapture(video_path)
                        if not self._sscd_cap.isOpened():
                            raise RuntimeError(f'Could not open video for SSCD decode: {video_path}')
                    return self._sscd_cap

                def _sscd_read_frame_rgb(self, video_path: str, frame_idx: int):
                    """Read + resize an RGB frame for SSCD; cached with a small LRU."""
                    import cv2
                    import numpy as np

                    # LRU cache hit
                    if frame_idx in self._sscd_frame_cache:
                        rgb = self._sscd_frame_cache.pop(frame_idx)
                        self._sscd_frame_cache[frame_idx] = rgb
                        return rgb

                    cap = self._sscd_get_cap(video_path)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        return None
                    frame = cv2.resize(frame, (self.sscd_input, self.sscd_input), interpolation=cv2.INTER_AREA)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Insert into LRU
                    self._sscd_frame_cache[frame_idx] = rgb
                    if len(self._sscd_frame_cache) > int(self._sscd_frame_cache_max):
                        self._sscd_frame_cache.popitem(last=False)
                    return rgb

                def _sscd_embed_rgb(self, rgb_uint8):
                    import torch
                    import torch.nn.functional as F
                    if rgb_uint8 is None:
                        return None
                    x = torch.from_numpy(rgb_uint8).to(self.device).float() / 255.0
                    x = x.permute(2, 0, 1).unsqueeze(0).contiguous()
                    x = (x - self._sscd_mean) / self._sscd_std
                    with torch.inference_mode():
                        out = self.sscd_model(x)
                    # TorchScript usually returns a Tensor; keep some generic handling anyway.
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    if isinstance(out, dict):
                        out = out.get('embeddings') or out.get('embedding') or next(iter(out.values()))
                    if not torch.is_tensor(out):
                        return None
                    if out.dim() == 2:
                        out = out[0]
                    v = out.flatten()
                    return F.normalize(v, dim=0)

                @staticmethod
                def _median_composite(frames_rgb_list):
                    import numpy as np
                    if not frames_rgb_list:
                        return None
                    stack = np.stack(frames_rgb_list, axis=0).astype(np.float32)
                    comp = np.median(stack, axis=0)
                    comp = np.clip(comp, 0, 255).astype(np.uint8)
                    return comp

                def _select_stable_indices_near_cut(self, cut_frame: int, total_frames: int, side: str, flash_event: dict):
                    """Pick 3-5 stable frames close to the boundary for SSCD median compositing."""
                    spike = set(int(x) for x in (flash_event.get('spike_frames') or []))
                    baseline = flash_event.get('baseline')
                    amp = float(flash_event.get('amp', 0.0) or 0.0)

                    # Candidate order: closest to boundary first
                    if side == 'pre':
                        cands = [cut_frame - k for k in range(1, 1 + 12)]
                    else:
                        cands = [cut_frame + k for k in range(0, 12)]
                    cands = [int(max(0, min(total_frames - 1, i))) for i in cands]

                    # If we don't know baseline yet, just take the closest frames excluding spikes
                    if baseline is None:
                        chosen = []
                        for i in cands:
                            if i in spike:
                                continue
                            chosen.append(i)
                            if len(chosen) >= self.SSCD_K:
                                break
                        return chosen

                    # Tolerance: accept frames close to baseline, excluding spike frames
                    tol = float(max(12.0, 0.35 * abs(amp)))
                    chosen = []
                    for i in cands:
                        if i in spike:
                            continue
                        v = self._flash_signal(i)
                        if v is None:
                            continue
                        if abs(float(v) - float(baseline)) <= tol:
                            chosen.append(i)
                            if len(chosen) >= self.SSCD_K:
                                break

                    # If too few, relax tolerance once
                    if len(chosen) < self.SSCD_MIN_K:
                        tol2 = tol * 1.6
                        chosen = []
                        for i in cands:
                            if i in spike:
                                continue
                            v = self._flash_signal(i)
                            if v is None:
                                continue
                            if abs(float(v) - float(baseline)) <= tol2:
                                chosen.append(i)
                                if len(chosen) >= self.SSCD_K:
                                    break

                    return chosen

                def _sscd_stable_similarity(self, video_path: str, cut_frame: int, total_frames: int, flash_event: dict):
                    """Compute SSCD cosine similarity between median-composited stable frames."""
                    import math
                    if (self.sscd_model is None) or (not video_path):
                        return None

                    pre_ids = self._select_stable_indices_near_cut(cut_frame, total_frames, 'pre', flash_event)
                    post_ids = self._select_stable_indices_near_cut(cut_frame, total_frames, 'post', flash_event)
                    if len(pre_ids) < self.SSCD_MIN_K or len(post_ids) < self.SSCD_MIN_K:
                        return None

                    pre_frames = [self._sscd_read_frame_rgb(video_path, i) for i in pre_ids]
                    post_frames = [self._sscd_read_frame_rgb(video_path, i) for i in post_ids]
                    pre_frames = [f for f in pre_frames if f is not None]
                    post_frames = [f for f in post_frames if f is not None]
                    if len(pre_frames) < self.SSCD_MIN_K or len(post_frames) < self.SSCD_MIN_K:
                        return None

                    pre_comp = self._median_composite(pre_frames)
                    post_comp = self._median_composite(post_frames)
                    e1 = self._sscd_embed_rgb(pre_comp)
                    e2 = self._sscd_embed_rgb(post_comp)
                    if e1 is None or e2 is None:
                        return None
                    sim = float((e1 * e2).sum().item())
                    if math.isnan(sim):
                        return None
                    return sim


                def _embed_all_required_frames(self, video_path: str, indices_to_embed, abort_flag=None, progress_cb=None):
                    import cv2
                    import torch

                    indices = sorted(set(int(i) for i in indices_to_embed))
                    if not indices:
                        return

                    self._log.info('Embedding %d unique frames for validation...', len(indices))
                    indices_set = set(indices)

                    target_size = self._target_size()
                    batch_imgs, batch_ids = [], []

                    use_cuda_decode = True
                    try:
                        gpu_frame = cv2.cuda_GpuMat()
                        cap = cv2.cudacodec.createVideoReader(video_path)
                    except Exception as e:
                        use_cuda_decode = False
                        self._log.warning('CUDA video reader unavailable (falling back to CPU decode): %s', e)
                        cap = cv2.VideoCapture(video_path)

                    def flush_batch():
                        nonlocal batch_imgs, batch_ids
                        if not batch_imgs:
                            return
                        inputs = self.processor(
                            images=batch_imgs,
                            return_tensors='pt',
                            do_resize=False,
                            do_center_crop=False,
                        )
                        pixel_values = inputs['pixel_values'].to(self.device)
                        embeddings = self._embed_batch(pixel_values)
                        for j, fid in enumerate(batch_ids):
                            self.cache[fid] = embeddings[j]
                        batch_imgs, batch_ids = [], []

                    frame_idx = 0
                    wanted_i = 0
                    total_wanted = len(indices)

                    while True:
                        if abort_flag is not None and abort_flag.is_set():
                            raise InterruptedError

                        if use_cuda_decode:
                            ret, gpu_frame = cap.nextFrame(gpu_frame)
                            if not ret:
                                break

                            if frame_idx in indices_set:
                                resized_gpu = cv2.cuda.resize(gpu_frame, target_size)
                                try:
                                    rgb_gpu = cv2.cuda.cvtColor(resized_gpu, cv2.COLOR_BGRA2RGB)
                                except Exception:
                                    rgb_gpu = cv2.cuda.cvtColor(resized_gpu, cv2.COLOR_BGR2RGB)

                                rgb = rgb_gpu.download()

                                if rgb.ndim == 3 and rgb.shape[2] == 4:
                                    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGRA2RGB)
                                elif not (rgb.ndim == 3 and rgb.shape[2] == 3):
                                    frame_idx += 1
                                    continue

                                luma = self._luma_from_rgb(rgb)
                                hist, v_hi = self._hsv_hist_and_vhi_from_rgb(rgb)

                                self.luma_cache[frame_idx] = luma
                                self.vhi_cache[frame_idx] = v_hi
                                self.hist_cache[frame_idx] = hist

                                batch_imgs.append(rgb)
                                batch_ids.append(frame_idx)
                                wanted_i += 1
                        else:
                            ret, frame = cap.read()
                            if not ret:
                                break

                            if frame_idx in indices_set:
                                frame = cv2.resize(frame, target_size)
                                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                                luma = self._luma_from_rgb(rgb)
                                hist, v_hi = self._hsv_hist_and_vhi_from_rgb(rgb)

                                self.luma_cache[frame_idx] = luma
                                self.vhi_cache[frame_idx] = v_hi
                                self.hist_cache[frame_idx] = hist

                                batch_imgs.append(rgb)
                                batch_ids.append(frame_idx)
                                wanted_i += 1

                        if len(batch_imgs) >= self.batch_size:
                            flush_batch()

                        if progress_cb and total_wanted > 0 and frame_idx % 50 == 0:
                            progress_cb(min(1.0, wanted_i / total_wanted), f'Embedding frames ({wanted_i}/{total_wanted})')

                        frame_idx += 1

                    flush_batch()

                    try:
                        if hasattr(cap, 'release'):
                            cap.release()
                    except Exception:
                        pass

                    self._log.info('Frame embedding complete (%d cached).', len(self.cache))

                # --- Similarity helpers ---

                @staticmethod
                def _pairwise_median_similarity(pre_vecs, post_vecs) -> float:
                    import torch
                    pre = torch.stack(pre_vecs, dim=0)
                    post = torch.stack(post_vecs, dim=0)
                    sim = pre @ post.T
                    return float(sim.flatten().median().item())

                @staticmethod
                def _avg_adjacent_similarity(vecs) -> float:
                    import torch
                    if len(vecs) < 2:
                        return 1.0
                    mat = torch.stack(vecs, dim=0)
                    sims = (mat[:-1] * mat[1:]).sum(dim=1)
                    return float(sims.mean().item())

                @staticmethod
                def _otsu_threshold(scores, bins: int = 128) -> float:
                    import numpy as np
                    scores = np.clip(scores.astype(np.float32), 0.0, 1.0)
                    hist, bin_edges = np.histogram(scores, bins=bins, range=(0.0, 1.0))
                    hist = hist.astype(np.float32)

                    weight1 = np.cumsum(hist)
                    weight2 = np.cumsum(hist[::-1])[::-1]

                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
                    mean1 = np.cumsum(hist * bin_centers) / np.maximum(weight1, 1e-6)
                    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / np.maximum(weight2[::-1], 1e-6))[::-1]

                    between = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
                    if between.size == 0:
                        return float(DinoCutValidator.DEFAULT_BASE_THRESHOLD)

                    k = int(np.nanargmax(between))
                    return float(bin_centers[k])

                def _auto_threshold(self, cut_scores) -> float:
                    import numpy as np
                    if len(cut_scores) < 10:
                        return float(self.DEFAULT_BASE_THRESHOLD)

                    s = np.array(cut_scores, dtype=np.float32)
                    if np.std(s) < 0.01:
                        return float(np.clip(float(np.mean(s)) + 0.01, 0.86, 0.93))

                    t_otsu = self._otsu_threshold(s)
                    p75 = float(np.percentile(s, 75))
                    p90 = float(np.percentile(s, 90))

                    thr = max(p75, min(t_otsu, p90))
                    return float(np.clip(thr, 0.84, 0.94))

                def _window_indices(self, cut_frame: int, window: int, gap: int, total_frames: int):
                    pre_end = max(0, cut_frame - gap)
                    pre_start = max(0, pre_end - window)
                    post_start = min(total_frames, cut_frame + gap)
                    post_end = min(total_frames, post_start + window)
                    return list(range(pre_start, pre_end)), list(range(post_start, post_end))

                # --- Flash detection ---

                @staticmethod
                def _mad(values, center):
                    import numpy as np
                    arr = np.asarray(values, dtype=np.float32)
                    return float(np.median(np.abs(arr - float(center))))

                @staticmethod
                def _longest_consecutive_run(idxs):
                    """Return the indices belonging to the longest consecutive run."""
                    if not idxs:
                        return []
                    idxs = sorted(set(int(x) for x in idxs))
                    best = [idxs[0]]
                    cur = [idxs[0]]
                    for x in idxs[1:]:
                        if x == cur[-1] + 1:
                            cur.append(x)
                        else:
                            if len(cur) > len(best):
                                best = cur
                            cur = [x]
                    if len(cur) > len(best):
                        best = cur
                    return best

                def _detect_flash_event(self, cut_frame: int, total_frames: int):
                    """Detect flash/black-flash near a cut boundary using median + MAD.

                    Returns dict:
                      is_flash: bool
                      conf: 0..1
                      kind: 'bright'|'dark'|None
                      amp: float
                      dur: int
                      drift: float (pre/post baseline difference)
                      baseline_pre/baseline_post/baseline: floats
                      spike_frames: list[int] (frames to exclude)
                    """
                    import numpy as np

                    def clamp(i: int) -> int:
                        return int(max(0, min(total_frames - 1, i)))

                    # Build windows
                    center_ids = [clamp(cut_frame + k) for k in range(-self.FLASH_CENTER, self.FLASH_CENTER + 1)]
                    pre_base_ids = [clamp(cut_frame + k) for k in range(-self.FLASH_SCAN, -self.FLASH_BASE_GAP)]
                    post_base_ids = [clamp(cut_frame + k) for k in range(self.FLASH_BASE_GAP, self.FLASH_SCAN + 1)]

                    base_vals_pre = [self._flash_signal(i) for i in pre_base_ids]
                    base_vals_pre = [v for v in base_vals_pre if v is not None]
                    base_vals_post = [self._flash_signal(i) for i in post_base_ids]
                    base_vals_post = [v for v in base_vals_post if v is not None]

                    if len(base_vals_pre) + len(base_vals_post) < 6:
                        return {
                            'is_flash': False, 'conf': 0.0, 'kind': None, 'amp': 0.0, 'dur': 0, 'drift': None,
                            'baseline_pre': None, 'baseline_post': None, 'baseline': None, 'spike_frames': []
                        }

                    baseline_pre = float(np.median(base_vals_pre)) if base_vals_pre else float(np.median(base_vals_post))
                    baseline_post = float(np.median(base_vals_post)) if base_vals_post else float(np.median(base_vals_pre))
                    baseline = float(np.median([baseline_pre, baseline_post]))
                    drift = float(abs(baseline_pre - baseline_post))

                    base_vals = base_vals_pre + base_vals_post
                    mad = self._mad(base_vals, baseline)
                    mad = float(max(1.0, 1.4826 * mad))  # robust scale (avoid div0)

                    center_vals = [(i, self._flash_signal(i)) for i in center_ids]
                    center_vals = [(i, v) for (i, v) in center_vals if v is not None]
                    if not center_vals:
                        return {
                            'is_flash': False, 'conf': 0.0, 'kind': None, 'amp': 0.0, 'dur': 0, 'drift': drift,
                            'baseline_pre': baseline_pre, 'baseline_post': baseline_post, 'baseline': baseline, 'spike_frames': []
                        }

                    # Determine whether we have a bright or dark outlier.
                    pos = [(i, float(v) - baseline) for (i, v) in center_vals]
                    neg = [(i, baseline - float(v)) for (i, v) in center_vals]
                    pos_max = max(d for _, d in pos)
                    neg_max = max(d for _, d in neg)

                    kind = 'bright' if pos_max >= neg_max else 'dark'
                    deltas = pos if kind == 'bright' else neg
                    amp = float(max(d for _, d in deltas))

                    # Sensitivity: require some absolute amplitude, plus robust z-score.
                    amp_thr = float(max(10.0, 0.6 * self.flash_luma_delta))
                    z = float(amp / mad)

                    # Candidate spike frames (within center)
                    spike = [i for (i, d) in deltas if d >= max(amp_thr, 0.45 * amp_thr) and (d / mad) >= 2.8]
                    spike_run = self._longest_consecutive_run(spike)
                    dur = int(len(spike_run))

                    boundary_hit = any(abs(i - cut_frame) <= 1 for i in spike_run)

                    # Return-to-baseline: allow some drift proportional to amplitude.
                    # Lightning storms can shift exposure a bit; use ratio rather than a fixed cutoff.
                    if amp <= 1e-3:
                        return_ok = False
                    else:
                        return_ok = bool(drift <= max(12.0, 0.60 * amp))

                    # Confidence
                    amp_score = (amp - amp_thr) / max(1e-6, (1.8 * amp_thr))
                    amp_score = float(np.clip(amp_score, 0.0, 1.0))
                    z_score = float(np.clip((z - 3.0) / 3.5, 0.0, 1.0))

                    if dur <= 0:
                        dur_score = 0.0
                    elif dur <= 3:
                        dur_score = 1.0
                    elif dur <= 6:
                        dur_score = 0.85
                    elif dur <= self.FLASH_MAX_DUR:
                        # Longer flashes are still valid; give a slightly lower but positive weight.
                        dur_score = 0.70
                    else:
                        dur_score = 0.0

                    if amp <= 1e-3:
                        return_score = 0.0
                    else:
                        ratio = drift / amp
                        if ratio <= 0.30:
                            return_score = 1.0
                        elif ratio <= 0.60:
                            return_score = 0.7
                        elif ratio <= 1.00:
                            return_score = 0.25
                        else:
                            return_score = 0.0

                    conf = float(np.clip(0.52 * amp_score + 0.15 * z_score + 0.18 * dur_score + 0.15 * return_score, 0.0, 1.0))
                    if not boundary_hit:
                        conf *= 0.5

                    is_flash = bool(
                        conf >= 0.55 and boundary_hit and (self.FLASH_MIN_DUR <= dur <= self.FLASH_MAX_DUR)
                    )

                    # Exclusion frames: spike run + 1-frame padding
                    spike_frames = sorted(set([clamp(i + k) for i in spike_run for k in (-1, 0, 1)]))

                    return {
                        'is_flash': is_flash,
                        'conf': conf,
                        'kind': kind if is_flash else None,
                        'amp': amp,
                        'dur': dur,
                        'drift': drift,
                        'baseline_pre': baseline_pre,
                        'baseline_post': baseline_post,
                        'baseline': baseline,
                        'spike_frames': spike_frames,
                        'z': z,
                        'amp_thr': amp_thr,
                        'boundary_hit': boundary_hit,
                        'return_ok': return_ok,
                    }

                def _stable_indices(self, indices, flash_event):
                    """Remove flash frames + lingering outliers from similarity windows."""
                    if not indices:
                        return []
                    if not flash_event or not flash_event.get('is_flash', False):
                        return list(indices)

                    baseline = flash_event.get('baseline')
                    amp = float(flash_event.get('amp') or 0.0)
                    kind = flash_event.get('kind')
                    spike_frames = set(int(i) for i in flash_event.get('spike_frames') or [])

                    kept = []
                    for i in indices:
                        if i in spike_frames:
                            continue
                        sig = self._flash_signal(i)
                        if sig is None or baseline is None or amp <= 1e-6:
                            kept.append(i)
                            continue
                        # Remove "afterglow" frames that are still far from baseline.
                        if kind == 'bright':
                            if float(sig) > float(baseline) + 0.40 * amp:
                                continue
                        elif kind == 'dark':
                            if float(sig) < float(baseline) - 0.40 * amp:
                                continue
                        kept.append(i)
                    return kept

                def _flash_side_decision(self, cut_frame: int, total_frames: int, flash_event: dict):
                    """Decide which side should own the flash frames based on similarity."""
                    if not flash_event or not flash_event.get('is_flash', False):
                        return None
                    spike = [int(i) for i in (flash_event.get('spike_frames') or [])]
                    if not spike:
                        return None

                    # First, use spike distribution around the cut if unambiguous.
                    pre_count = sum(1 for i in spike if i < cut_frame)
                    post_count = sum(1 for i in spike if i >= cut_frame)
                    if pre_count > post_count:
                        return 'pre'
                    if post_count > pre_count:
                        return 'post'

                    flash_vecs = [self.cache[i] for i in spike if i in self.cache]
                    if not flash_vecs:
                        return None

                    pre_ids = self._select_stable_indices_near_cut(cut_frame, total_frames, 'pre', flash_event)
                    post_ids = self._select_stable_indices_near_cut(cut_frame, total_frames, 'post', flash_event)
                    pre_vecs = [self.cache[i] for i in pre_ids if i in self.cache]
                    post_vecs = [self.cache[i] for i in post_ids if i in self.cache]

                    if not pre_vecs and not post_vecs:
                        return None
                    if pre_vecs and not post_vecs:
                        return 'pre'
                    if post_vecs and not pre_vecs:
                        return 'post'

                    pre_sim = self._pairwise_median_similarity(flash_vecs, pre_vecs)
                    post_sim = self._pairwise_median_similarity(flash_vecs, post_vecs)

                    if pre_sim >= post_sim:
                        return 'pre'
                    return 'post'

                def _adjust_cut_for_flash(self, cut_frame: int, total_frames: int, flash_event: dict, cur_start_f: int, nxt_end_f: int) -> int:
                    """Move cut to keep flash frames on the most similar side."""
                    side = self._flash_side_decision(cut_frame, total_frames, flash_event)
                    if side is None:
                        return int(cut_frame)
                    spike = [int(i) for i in (flash_event.get('spike_frames') or [])]
                    if not spike:
                        return int(cut_frame)
                    flash_start = min(spike)
                    flash_end = max(spike)

                    if side == 'pre':
                        new_cut = int(flash_end + 1)
                    else:
                        new_cut = int(flash_start)

                    min_cut = int(cur_start_f) + 1
                    max_cut = int(nxt_end_f) - 1
                    if min_cut > max_cut:
                        return int(cut_frame)
                    if new_cut < min_cut:
                        new_cut = min_cut
                    if new_cut > max_cut:
                        new_cut = max_cut
                    return int(new_cut)

                # --- Cut features + decision ---

                def _cut_features(self, cut_frame: int, window: int, total_frames: int, video_path: str = None):
                    gap = max(2, int(max(3, window // 2)))
                    w_small = max(2, int(window))
                    w_large = int(min(max(w_small * 3, w_small + 2), self.MAX_LARGE_WINDOW))

                    pre_s, post_s = self._window_indices(cut_frame, w_small, gap, total_frames)
                    pre_l, post_l = self._window_indices(cut_frame, w_large, gap, total_frames)

                    pre_adj = cut_frame - 1
                    post_adj = cut_frame

                    def vecs_for(indices):
                        return [self.cache[i] for i in indices if i in self.cache]

                    pre_vecs_s = vecs_for(pre_s)
                    post_vecs_s = vecs_for(post_s)
                    pre_vecs_l = vecs_for(pre_l)
                    post_vecs_l = vecs_for(post_l)

                    if len(pre_vecs_s) < 2 or len(post_vecs_s) < 2:
                        return None

                    s_small_raw = self._pairwise_median_similarity(pre_vecs_s, post_vecs_s)
                    s_large_raw = None
                    if len(pre_vecs_l) >= 2 and len(post_vecs_l) >= 2:
                        s_large_raw = self._pairwise_median_similarity(pre_vecs_l, post_vecs_l)

                    s_comb_raw = float(s_small_raw) if s_large_raw is None else float(0.65 * s_small_raw + 0.35 * s_large_raw)

                    s_adj = None
                    if pre_adj in self.cache and post_adj in self.cache:
                        s_adj = float((self.cache[pre_adj] * self.cache[post_adj]).sum().item())

                    pre_cont = self._avg_adjacent_similarity(pre_vecs_s)
                    post_cont = self._avg_adjacent_similarity(post_vecs_s)
                    cont = float(min(pre_cont, post_cont))
                    motion = float(max(0.0, 1.0 - cont))

                    flash_event = self._detect_flash_event(cut_frame, total_frames)

                    s_sscd = None
                    if (video_path is not None) and (getattr(self, 'sscd_model', None) is not None) and flash_event.get('is_flash', False):
                        try:
                            s_sscd = self._sscd_stable_similarity(video_path, cut_frame, total_frames, flash_event)
                        except Exception as e:
                            self._log.debug('SSCD similarity failed at cut@%d: %s', cut_frame, e)
                            s_sscd = None


                    s_small_stable = None
                    s_large_stable = None
                    s_comb_stable = None
                    used_stable = False

                    if flash_event.get('is_flash', False):
                        used_stable = True
                        pre_s_st = self._stable_indices(pre_s, flash_event)
                        post_s_st = self._stable_indices(post_s, flash_event)
                        pre_l_st = self._stable_indices(pre_l, flash_event)
                        post_l_st = self._stable_indices(post_l, flash_event)

                        pre_vecs_s_st = vecs_for(pre_s_st)
                        post_vecs_s_st = vecs_for(post_s_st)
                        if len(pre_vecs_s_st) >= 2 and len(post_vecs_s_st) >= 2:
                            s_small_stable = self._pairwise_median_similarity(pre_vecs_s_st, post_vecs_s_st)

                        pre_vecs_l_st = vecs_for(pre_l_st)
                        post_vecs_l_st = vecs_for(post_l_st)
                        if len(pre_vecs_l_st) >= 2 and len(post_vecs_l_st) >= 2:
                            s_large_stable = self._pairwise_median_similarity(pre_vecs_l_st, post_vecs_l_st)

                        if s_small_stable is not None:
                            if s_large_stable is None:
                                s_comb_stable = float(s_small_stable)
                            else:
                                s_comb_stable = float(0.65 * float(s_small_stable) + 0.35 * float(s_large_stable))

                    if s_comb_stable is None:
                        s_comb_stable = float(s_comb_raw)

                    # Pixel cue on representative stable frames
                    def get_hist(cands):
                        for i in cands:
                            if 0 <= i < total_frames and i in self.hist_cache:
                                return self.hist_cache[i]
                        return None

                    # Prefer frames near the boundary but outside typical flash band
                    pre_hist = get_hist([cut_frame - 4, cut_frame - 5, cut_frame - 3, cut_frame - 6])
                    post_hist = get_hist([cut_frame + 4, cut_frame + 5, cut_frame + 3, cut_frame + 6])
                    pixel_sim = self._hist_intersection(pre_hist, post_hist)
                    if pixel_sim != pixel_sim:
                        pixel_sim = None

                    return {
                        's_small_raw': float(s_small_raw),
                        's_large_raw': None if s_large_raw is None else float(s_large_raw),
                        's_comb_raw': float(s_comb_raw),
                        's_small_stable': None if s_small_stable is None else float(s_small_stable),
                        's_large_stable': None if s_large_stable is None else float(s_large_stable),
                        's_comb_stable': float(s_comb_stable),
                        's_sscd': None if s_sscd is None else float(s_sscd),
                        's_adj': None if s_adj is None else float(s_adj),
                        'pre_cont': float(pre_cont),
                        'post_cont': float(post_cont),
                        'cont': float(cont),
                        'motion': float(motion),
                        'pixel_sim': None if pixel_sim is None else float(pixel_sim),
                        'used_stable': bool(used_stable),
                        'flash': flash_event,
                        'gap': int(gap),
                        'w_small': int(w_small),
                        'w_large': int(w_large),
                    }

                def validate_cut(self, scene_index: int, cut_frame: int, window: int, total_frames: int, base_thr: float, video_path: str = None, return_feats: bool = False):
                    """Return True if it is a true cut (KEEP), False if it should be merged (MERGE)."""
                    feats = self._cut_features(cut_frame, window, total_frames, video_path=video_path)
                    if feats is None:
                        return (True, None) if return_feats else True

                    s_raw = float(feats['s_comb_raw'])
                    s_stable = float(feats['s_comb_stable'])
                    s_sscd = feats.get('s_sscd', None)
                    s_adj = feats.get('s_adj')
                    cont = float(feats.get('cont', 1.0))
                    motion = float(feats.get('motion', 0.0))
                    pixel_sim = feats.get('pixel_sim')

                    flash = feats.get('flash') or {}
                    is_flash = bool(flash.get('is_flash', False))
                    flash_conf = float(flash.get('conf', 0.0) or 0.0)
                    flash_amp = float(flash.get('amp', 0.0) or 0.0)
                    flash_dur = int(flash.get('dur', 0) or 0)
                    flash_drift = flash.get('drift', None)

                    decision_keep = True
                    reason = 'default_keep'
                    sscd_decision_lock = False

                    ambiguous_non_flash = False
                    high_sim_non_flash = False
                    if not is_flash:
                        # Ambiguous when DINO is near threshold or cues disagree (loosened).
                        near_thr = abs(float(s_raw) - float(base_thr)) <= 0.10
                        adj_amb = (s_adj is not None) and (0.60 <= float(s_adj) <= 0.90) and (float(s_raw) >= float(base_thr) - 0.08)
                        pix_amb = (
                            (pixel_sim is None)
                            or (float(pixel_sim) < (self.PIXEL_HIST_STRONG + 0.04))
                        )
                        mot_amb = (float(motion) >= 0.20) and (float(s_raw) >= float(base_thr) - 0.08)
                        high_sim_non_flash = float(s_raw) >= float(base_thr) + 0.01
                        ambiguous_non_flash = bool(near_thr or adj_amb or pix_amb or mot_amb or high_sim_non_flash)

                    # Evaluate SSCD for flash cuts or ambiguous/high-sim non-flash cuts.
                    if (s_sscd is None) and (video_path is not None) and (getattr(self, 'sscd_model', None) is not None):
                        if is_flash or ambiguous_non_flash or high_sim_non_flash:
                            try:
                                s_sscd = self._sscd_stable_similarity(video_path, cut_frame, total_frames, flash)
                            except Exception as e:
                                self._log.debug('SSCD similarity failed at cut@%d: %s', cut_frame, e)
                                s_sscd = None
                            feats['s_sscd'] = None if s_sscd is None else float(s_sscd)

                    # 1) Flash path: default MERGE on confident flash unless strong cut evidence exists.
                    if is_flash:
                        # Lower confidence threshold to favor merging flashes.
                        min_flash_conf = 0.40
                        if s_sscd is not None:
                            min_flash_conf = 0.35

                        # Strong cut evidence overrides flash-merge default, but be very conservative.
                        low_cut_thr = max(0.48, min(0.64, float(base_thr) - 0.26 - 0.12 * motion))
                        s_cut = float(min(s_raw, s_stable))
                        strong_signals = 0

                        if (s_adj is not None) and (float(s_adj) < (self.ADJ_STRONG_CUT - 0.12)):
                            strong_signals += 1
                        if s_cut < low_cut_thr:
                            strong_signals += 1
                        if (pixel_sim is not None) and (float(pixel_sim) < 0.025):
                            strong_signals += 1
                        if s_sscd is not None:
                            try:
                                if float(s_sscd) < 0.55:
                                    strong_signals += 1
                            except Exception:
                                pass

                        # Require multiple strong indicators before keeping a flash cut.
                        strong_cut = (strong_signals >= 3) or (
                            (s_adj is not None and float(s_adj) < (self.ADJ_STRONG_CUT - 0.20))
                        ) or (s_cut < (low_cut_thr - 0.10))

                        if flash_conf >= min_flash_conf:
                            if strong_cut:
                                decision_keep = True
                                reason = 'keep_flash_strong_cut'
                            else:
                                decision_keep = False
                                reason = 'merge_flash_default'
                        else:
                            decision_keep = True
                            reason = 'keep_flash_low_conf'

                    # 2) Non-flash ambiguous/high-sim: use SSCD as tie-breaker
                    if decision_keep and (not is_flash) and (ambiguous_non_flash or high_sim_non_flash) and (s_sscd is not None):
                        try:
                            req_sscd = float(max(0.62, min(0.84, self._sscd_required_sim(motion) - 0.04)))
                            if float(s_sscd) >= req_sscd:
                                decision_keep = False
                                reason = 'merge_same_shot_sscd'
                                sscd_decision_lock = True
                            elif float(s_sscd) <= (req_sscd - 0.10):
                                # Only keep if multiple cues indicate a real cut.
                                cut_like = 0
                                if (s_adj is not None) and (float(s_adj) < 0.76):
                                    cut_like += 1
                                if cont < 0.84:
                                    cut_like += 1
                                if (pixel_sim is not None) and (float(pixel_sim) < 0.06):
                                    cut_like += 1
                                if s_raw < (float(base_thr) - 0.10):
                                    cut_like += 1
                                if cut_like >= 2:
                                    decision_keep = True
                                    reason = 'keep_cut_sscd'
                                    sscd_decision_lock = True
                        except Exception:
                            pass

                    # 2b) Non-flash moderate continuity merge: reduce false keeps
                    if decision_keep and (not is_flash) and (not sscd_decision_lock):
                        mid_cont = (
                            (cont >= 0.87)
                            and ((s_adj is None) or (float(s_adj) >= 0.83))
                            and (s_raw >= (float(base_thr) - 0.18))
                            and ((pixel_sim is None) or (float(pixel_sim) >= 0.10))
                        )
                        if mid_cont:
                            decision_keep = False
                            reason = 'merge_same_shot_continuity_mid'
                        else:
                            # High pixel similarity can override lower s_raw when continuity is strong.
                            hi_pix_merge = (
                                (pixel_sim is not None)
                                and (float(pixel_sim) >= 0.16)
                                and (cont >= 0.85)
                                and ((s_adj is None) or (float(s_adj) >= 0.80))
                                and (s_raw >= (float(base_thr) - 0.26))
                            )
                            if hi_pix_merge:
                                decision_keep = False
                                reason = 'merge_same_shot_pixel'

                    # 2c) Non-flash high-continuity merge: reduce false keeps
                    if decision_keep and (not is_flash):
                        high_cont = (
                            (cont >= 0.90)
                            and ((s_adj is None) or (float(s_adj) >= 0.88))
                            and (s_raw >= (float(base_thr) - 0.12))
                            and ((pixel_sim is None) or (float(pixel_sim) >= 0.10))
                        )
                        if high_cont:
                            decision_keep = False
                            reason = 'merge_same_shot_continuity'

                    # 3) Non-flash: strong adjacent discontinuity => KEEP (unless SSCD decided)
                    if (not sscd_decision_lock) and decision_keep and (s_adj is not None) and (float(s_adj) < self.ADJ_STRONG_CUT):
                        decision_keep = True
                        reason = 'adj_strong_cut'

                    # 4) Non-flash false-positive merge path (fast motion / shaky camera):
                    if decision_keep and (not is_flash) and (not sscd_decision_lock):
                        # Must be near within-shot continuity and above a conservative absolute threshold.
                        req_rel = max(0.82, cont - 0.06)
                        req_abs = max(0.86, min(0.92, float(base_thr)))
                        req = float(max(req_rel, req_abs))

                        pix_ok = (pixel_sim is not None and float(pixel_sim) >= self.PIXEL_HIST_STRONG)
                        if (s_raw >= req) and pix_ok and ((s_adj is None) or float(s_adj) >= 0.78):
                            decision_keep = False
                            reason = 'merge_same_shot'

                    decision = (
                        f"{self.BOLD}{self.KEEP_FG}KEEP{self.RESET}"
                        if decision_keep
                        else f"{self.BOLD}{self.MERG_FG}MERGE{self.RESET}"
                    )

                    self._log.debug(
                        'Scene #%d cut@%d | s_raw=%.3f s_stable=%.3f s_sscd=%s s_adj=%s cont=%.3f motion=%.3f pixel=%s '
                        'flash=%s(%.2f) amp=%.1f dur=%d drift=%s -> %s (%s)',
                        scene_index,
                        cut_frame,
                        s_raw,
                        s_stable,
                        '{:.3f}'.format(s_sscd) if s_sscd is not None else 'n/a',
                        '{:.3f}'.format(s_adj) if s_adj is not None else 'n/a',
                        cont,
                        motion,
                        '{:.3f}'.format(pixel_sim) if pixel_sim is not None else 'n/a',
                        'Y' if is_flash else 'N',
                        flash_conf,
                        flash_amp,
                        flash_dur,
                        '{:.1f}'.format(float(flash_drift)) if flash_drift is not None else 'n/a',
                        decision,
                        reason,
                    )

                    return (decision_keep, feats) if return_feats else decision_keep

                def filter_scenes(self, video_path: str, scenes: list, window: int, abort_flag=None, progress_cb=None) -> list:
                    import cv2

                    if len(scenes) < 2:
                        return scenes

                    cap = cv2.VideoCapture(video_path)
                    self.total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    cap.release()

                    if self.total_video_frames <= 0:
                        self._log.warning('Could not determine frame count for AI validation; skipping.')
                        return scenes

                    all_indices = set()
                    cut_frames = []
                    for _, end_tc in scenes[:-1]:
                        cut = int(end_tc.get_frames())
                        cut_frames.append(cut)

                    w = int(window)
                    gap = max(2, int(max(3, w // 2)))
                    w_small = max(2, w)
                    w_large = int(min(max(w_small * 3, w_small + 2), self.MAX_LARGE_WINDOW))

                    for cut in cut_frames:
                        # Flash analysis window
                        for k in range(-self.FLASH_SCAN, self.FLASH_SCAN + 1):
                            all_indices.add(max(0, min(self.total_video_frames - 1, cut + k)))

                        pre_s, post_s = self._window_indices(cut, w_small, gap, self.total_video_frames)
                        all_indices.update(pre_s); all_indices.update(post_s)

                        pre_l, post_l = self._window_indices(cut, w_large, gap, self.total_video_frames)
                        all_indices.update(pre_l); all_indices.update(post_l)

                    self._embed_all_required_frames(
                        video_path,
                        sorted(all_indices),
                        abort_flag=abort_flag,
                        progress_cb=progress_cb,
                    )

                    cut_scores = []
                    for i, cut in enumerate(cut_frames, start=1):
                        feats = self._cut_features(cut, w, self.total_video_frames)
                        if feats is None:
                            continue
                        cut_scores.append(float(feats['s_comb_raw']))
                        if progress_cb and i % 10 == 0:
                            progress_cb(min(1.0, i / max(1, len(cut_frames))), f'Analyzing cut scores ({i}/{len(cut_frames)})')

                    base_thr = self._auto_threshold(cut_scores)
                    self._log.info(
                        'AI validation adaptive same-shot threshold: base_thr=%.3f (window=%d, gap=%d, large=%d)',
                        base_thr,
                        w,
                        gap,
                        w_large,
                    )

                    validated_scenes = []
                    cur_start, cur_end = scenes[0]

                    for i in range(1, len(scenes)):
                        nxt_start, nxt_end = scenes[i]
                        cut_frame_num = int(nxt_start.get_frames())
                        cur_len = int(cur_end.get_frames() - cur_start.get_frames())
                        nxt_len = int(nxt_end.get_frames() - nxt_start.get_frames())

                        if abort_flag is not None and abort_flag.is_set():
                            raise InterruptedError

                        keep_cut, feats = self.validate_cut(
                            scene_index=i,
                            cut_frame=cut_frame_num,
                            window=w,
                            total_frames=self.total_video_frames,
                            base_thr=base_thr,
                            video_path=video_path,
                            return_feats=True,
                        )

                        flash = (feats or {}).get('flash') or {}
                        is_flash = bool(flash.get('is_flash', False))
                        flash_conf = float(flash.get('conf', 0.0) or 0.0)
                        s_sscd = (feats or {}).get('s_sscd')
                        s_stable = float((feats or {}).get('s_comb_stable', (feats or {}).get('s_comb_raw', 0.0)) or 0.0)
                        spike_frames = flash.get('spike_frames') or []
                        boundary_hit = bool(flash.get('boundary_hit', False))

                        min_flash_conf = 0.40
                        if s_sscd is not None:
                            min_flash_conf = 0.35

                        # Preserve very short scenes by default unless this is a flash-like cut with strong same-shot evidence.
                        if cur_len <= short_guard_frames or nxt_len <= short_guard_frames:
                            flash_like = bool(spike_frames) and boundary_hit and (flash_conf >= 0.35)
                            same_shot = float(s_stable) >= max(0.0, float(base_thr) - 0.05)
                            if flash_like and same_shot:
                                keep_cut = False
                            elif not (is_flash and flash_conf >= min_flash_conf):
                                keep_cut = True

                        if keep_cut:
                            # If it is a flash but we keep the cut, snap the boundary to the best side.
                            if is_flash and flash_conf >= min_flash_conf:
                                new_cut = self._adjust_cut_for_flash(
                                    cut_frame=cut_frame_num,
                                    total_frames=self.total_video_frames,
                                    flash_event=flash,
                                    cur_start_f=int(cur_start.get_frames()),
                                    nxt_end_f=int(nxt_end.get_frames()),
                                )
                                if new_cut != cut_frame_num:
                                    cur_end = Timecode(int(new_cut), cur_start.fps)
                                    nxt_start = Timecode(int(new_cut), cur_start.fps)

                            validated_scenes.append((cur_start, cur_end))
                            cur_start, cur_end = nxt_start, nxt_end
                        else:
                            cur_end = nxt_end

                        if progress_cb:
                            progress_cb(min(1.0, i / max(1, len(scenes) - 1)), f'Validating cuts ({i}/{len(scenes) - 1})')

                    validated_scenes.append((cur_start, cur_end))

                    # Post-pass: merge tiny flash clips (mid-scene) into the most likely neighbor.
                    try:
                        if short_guard_frames and len(validated_scenes) >= 3:
                            fps_local = float(validated_scenes[0][0].fps) if validated_scenes else 0.0
                            frames = [
                                [int(st.get_frames()), int(et.get_frames())]
                                for st, et in validated_scenes
                            ]

                            boundary_cache = {}

                            def boundary_info(cut_frame: int):
                                if cut_frame in boundary_cache:
                                    return boundary_cache[cut_frame]
                                feats_b = self._cut_features(cut_frame, w, self.total_video_frames, video_path=video_path)
                                if feats_b is None:
                                    info = {'flash_like': False, 'score': 0.0}
                                else:
                                    flash_b = (feats_b or {}).get('flash') or {}
                                    flash_conf_b = float(flash_b.get('conf', 0.0) or 0.0)
                                    spike_b = flash_b.get('spike_frames') or []
                                    boundary_hit_b = bool(flash_b.get('boundary_hit', False))
                                    flash_like_b = bool(spike_b) and boundary_hit_b and (flash_conf_b >= 0.35)
                                    s_stable_b = float((feats_b or {}).get('s_comb_stable', (feats_b or {}).get('s_comb_raw', 0.0)) or 0.0)
                                    same_shot_b = float(s_stable_b) >= max(0.0, float(base_thr) - 0.05)
                                    score_b = (flash_conf_b * 1.2) + (float(s_stable_b) - float(base_thr) if same_shot_b else 0.0)
                                    info = {'flash_like': flash_like_b, 'score': score_b}
                                boundary_cache[cut_frame] = info
                                return info

                            ultra_short_frames = max(1, int(round(float(fps_local) * 0.05))) if fps_local > 0 else 1
                            i = 1
                            merged_any = False
                            while i < len(frames) - 1:
                                seg_len = int(frames[i][1] - frames[i][0])
                                if seg_len <= int(short_guard_frames):
                                    info_pre = boundary_info(frames[i][0])
                                    info_post = boundary_info(frames[i][1])
                                    force_merge = seg_len <= ultra_short_frames
                                    if force_merge or info_pre['flash_like'] or info_post['flash_like']:
                                        if info_pre['score'] >= info_post['score']:
                                            frames[i - 1][1] = frames[i][1]
                                            del frames[i]
                                            merged_any = True
                                            continue
                                        else:
                                            frames[i + 1][0] = frames[i][0]
                                            del frames[i]
                                            merged_any = True
                                            continue
                                i += 1

                            if merged_any:
                                validated_scenes = [
                                    (Timecode(st, fps_local), Timecode(et, fps_local))
                                    for st, et in frames
                                    if et > st
                                ]
                    except Exception as e:
                        self._log.warning('Tiny flash merge post-pass failed: %s', e)

                    # Release SSCD video handle (if used)
                    try:
                        if getattr(self, '_sscd_cap', None) is not None:
                            self._sscd_cap.release()
                    except Exception:
                        pass
                    self._sscd_cap = None
                    self._sscd_video_path = None
                    try:
                        if getattr(self, '_sscd_frame_cache', None) is not None:
                            self._sscd_frame_cache.clear()
                    except Exception:
                        pass

                    return validated_scenes


            validator = DinoCutValidator(
                model_dir="./weights",
                batch_size=48,
                flash_luma_delta=float(self.flash_sensitivity_var.get())
            )

            def progress_cb(pct: float, msg: str):
                # Map into the app's overall progress bar range.
                self.progress_queue.put_nowait((75 + float(pct) * 4.0, msg))

            validated_scenes = validator.filter_scenes(
                video_path=video_path,
                scenes=scenes,
                window=int(self.ai_window_var.get()),
                abort_flag=self.abort_flag,
                progress_cb=progress_cb,
            )

            logger.info("AI validation reduced scenes from %d to %d", len(scenes), len(validated_scenes))
            return validated_scenes

        except InterruptedError:
            raise
        except (ImportError, ModuleNotFoundError, FileNotFoundError, RuntimeError) as e:
            logger.exception("AI validation failed. Skipping.")
            messagebox.showwarning("AI Validation Failed", f"Could not perform AI validation: {e}")
            return scenes

    def _refine_with_pyscenedetect(self, video_path: str, scenes, fps: float, total_frames: int):
        if not scenes or len(scenes) < 2:
            return scenes

        snap_window = int(self.refine_snap_var.get() or 0)
        if snap_window <= 0:
            return scenes

        try:
            from scenedetect.detectors import ContentDetector
        except Exception as e:
            raise RuntimeError("PySceneDetect is not installed. Install 'scenedetect'.") from e

        # Collect cuts from PySceneDetect (ContentDetector)
        logger.info(
            "PySceneDetect refinement: starting (threshold=%.2f, snap_window=%d)",
            refine_threshold,
            snap_window,
        )
        cut_frames = []
        refine_threshold = float(self.refine_threshold_var.get() or 27.0)

        try:
            from scenedetect import open_video, SceneManager

            video = open_video(video_path)
            scene_manager = SceneManager()
            try:
                detector = ContentDetector(threshold=refine_threshold)
            except TypeError:
                detector = ContentDetector()
            scene_manager.add_detector(detector)
            scene_manager.detect_scenes(video, show_progress=False)
            scene_list = scene_manager.get_scene_list()
            if scene_list and len(scene_list) > 1:
                cut_list = [s[0] for s in scene_list[1:]]
                cut_frames = [
                    int(c.get_frames() if hasattr(c, "get_frames") else int(c))
                    for c in (cut_list or [])
                ]
            else:
                cut_frames = []
            try:
                if hasattr(video, "close"):
                    video.close()
            except Exception:
                pass
        except Exception:
            # Fallback for older PySceneDetect API
            from scenedetect.video_manager import VideoManager
            from scenedetect.scene_manager import SceneManager

            video_manager = VideoManager([video_path])
            scene_manager = SceneManager()
            try:
                detector = ContentDetector(threshold=refine_threshold)
            except TypeError:
                detector = ContentDetector()
            scene_manager.add_detector(detector)
            video_manager.start()
            scene_manager.detect_scenes(frame_source=video_manager)
            scene_list = scene_manager.get_scene_list()
            if scene_list and len(scene_list) > 1:
                cut_list = [s[0] for s in scene_list[1:]]
                cut_frames = [
                    int(c.get_frames() if hasattr(c, "get_frames") else int(c))
                    for c in (cut_list or [])
                ]
            else:
                cut_frames = []
            try:
                video_manager.release()
            except Exception:
                pass

        if not cut_frames:
            logger.info("PySceneDetect refinement: no cuts found (skipping).")
            return scenes

        # Normalize + sort
        cut_frames = sorted(set(int(c) for c in cut_frames if 0 < int(c) < int(total_frames)))
        logger.info("PySceneDetect refinement: %d candidate cuts.", len(cut_frames))

        # Current cuts from detector
        orig_cuts = [int(st.get_frames()) for st, _ in scenes[1:]]
        logger.info("PySceneDetect refinement: %d original cuts.", len(orig_cuts))

        import bisect

        def snap_cut(cut: int) -> int:
            pos = bisect.bisect_left(cut_frames, cut)
            candidates = []
            if pos < len(cut_frames):
                candidates.append(cut_frames[pos])
            if pos > 0:
                candidates.append(cut_frames[pos - 1])
            if not candidates:
                return cut
            nearest = min(candidates, key=lambda x: abs(x - cut))
            if abs(nearest - cut) <= snap_window:
                return nearest
            return cut

        new_cuts = []
        last = 0
        snapped = 0
        for cut in orig_cuts:
            new_cut = snap_cut(cut)
            if new_cut != cut:
                snapped += 1
                logger.debug("PySceneDetect refinement: snap %d -> %d", cut, new_cut)
            new_cut = max(new_cut, last + 1)
            new_cut = min(new_cut, int(total_frames) - 1)
            new_cuts.append(new_cut)
            last = new_cut
        logger.info("PySceneDetect refinement: snapped %d/%d cuts.", snapped, len(orig_cuts))

        # Rebuild scenes
        out = []
        start = 0
        for cut in new_cuts:
            out.append((Timecode(start, fps), Timecode(cut, fps)))
            start = cut
        out.append((Timecode(start, fps), Timecode(total_frames, fps)))
        return out

    def _merge_ultra_short_scenes(self, scenes, fps: float, total_frames: int, max_seconds: float = 0.05):
        """Merge ultra-short scenes into neighbors regardless of validation settings."""
        if not scenes or len(scenes) < 2:
            return scenes

        if fps <= 0:
            thr = 1
        else:
            thr = max(1, int(round(float(fps) * float(max_seconds))))

        frames = [[int(st.get_frames()), int(et.get_frames())] for st, et in scenes]

        i = 0
        while i < len(frames):
            if len(frames) == 1:
                break
            seg_len = int(frames[i][1] - frames[i][0])
            if seg_len <= thr:
                if i == 0:
                    # Merge into next
                    frames[1][0] = frames[0][0]
                    del frames[0]
                    continue
                if i == len(frames) - 1:
                    # Merge into previous
                    frames[i - 1][1] = frames[i][1]
                    del frames[i]
                    i = max(0, i - 1)
                    continue

                prev_len = int(frames[i - 1][1] - frames[i - 1][0])
                next_len = int(frames[i + 1][1] - frames[i + 1][0])
                if prev_len >= next_len:
                    frames[i - 1][1] = frames[i][1]
                    del frames[i]
                    i = max(0, i - 1)
                else:
                    frames[i + 1][0] = frames[i][0]
                    del frames[i]
                continue
            i += 1

        # Rebuild scenes
        out = []
        for st_f, et_f in frames:
            if et_f > st_f:
                out.append((Timecode(st_f, fps), Timecode(et_f, fps)))
        if out and total_frames > 0 and out[-1][1].get_frames() < total_frames:
            out[-1] = (out[-1][0], Timecode(total_frames, fps))
        return out

    def _abort_detection(self): self.abort_flag.set()
    def _export_csv(self, filename: Path):
        """Export detected scenes to CSV."""
        if not self.detected_scenes:
            return
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Scene",
                "Start Time",
                "Start Frame",
                "End Time",
                "End Frame (exclusive)",
                "Duration (frames)",
                "Duration",
            ])
            for i, (st, et) in enumerate(self.detected_scenes, start=1):
                sf = st.get_frames()
                ef = et.get_frames()
                ss = st.get_seconds()
                es = et.get_seconds()
                writer.writerow([
                    i,
                    _format_hhmmss_ms(ss),
                    sf,
                    _format_hhmmss_ms(es),
                    ef,
                    max(0, ef - sf),
                    _format_hhmmss_ms(max(0.0, es - ss)),
                ])
        logger.info("Exported scene list to CSV: %s", filename)

    def _export_html(self, filename: Path):
        """Export detected scenes to a simple HTML table."""
        if not self.detected_scenes:
            return

        rows = []
        for i, (st, et) in enumerate(self.detected_scenes, start=1):
            sf = st.get_frames()
            ef = et.get_frames()
            ss = st.get_seconds()
            es = et.get_seconds()
            dur_f = max(0, ef - sf)
            dur_s = max(0.0, es - ss)
            rows.append(
                f"<tr><td>{i}</td>"
                f"<td>{_format_hhmmss_ms(ss)}</td><td>{sf}</td>"
                f"<td>{_format_hhmmss_ms(es)}</td><td>{ef}</td>"
                f"<td>{dur_f}</td><td>{_format_hhmmss_ms(dur_s)}</td></tr>"
            )

        detector_label = self.detector_type_var.get() or "Detector"
        html = (
            "<!doctype html><html><head><meta charset='utf-8'/>"
            "<title>Scene List</title>"
            "<style>"
            "body{font-family:Arial, sans-serif; padding:16px;}"
            "table{border-collapse:collapse; width:100%;}"
            "th,td{border:1px solid #ccc; padding:6px 8px; text-align:left;}"
            "th{background:#f3f3f3;}"
            "</style></head><body>"
            f"<h2>Detected Scenes ({detector_label})</h2>"
            "<table><thead><tr>"
            "<th>Scene</th><th>Start Time</th><th>Start Frame</th>"
            "<th>End Time</th><th>End Frame (exclusive)</th>"
            "<th>Duration (frames)</th><th>Duration</th>"
            "</tr></thead><tbody>"
            + "".join(rows) +
            "</tbody></table></body></html>"
        )

        filename.write_text(html, encoding="utf-8")
        logger.info("Exported scene list to HTML: %s", filename)

    def _export_sc(self, fps: float, total_frames: int, filename: Path):
        """Export a .sc file (same frame-value format as before)."""
        if not self.detected_scenes:
            return

        total_frames = int(total_frames)
        fps = float(fps) if fps else 0.0

        frame_values = [1] * max(0, total_frames)
        last_cut_frame = 0

        # Mark cut frames (skip first scene start at 0)
        for start_tc, _ in self.detected_scenes[1:]:
            cut_frame = int(start_tc.get_frames())
            if 0 <= cut_frame < total_frames:
                frame_values[cut_frame] = 255
                last_cut_frame = max(last_cut_frame, cut_frame)

        content_frames = last_cut_frame if last_cut_frame > 0 else max(0, total_frames - 1)

        with open(filename, "w", encoding="utf-8") as f:
            # Header fields preserved for compatibility with your existing tooling.
            f.write(
                f"0\n{total_frames}\n{content_frames}\n{int(fps * 1000)}\n{content_frames}\n100\n"
            )
            f.write("".join(f"{v}\n" for v in frame_values))

        logger.info("Exported scene list to .sc file: %s", filename)

    def _save_images(self, video_path: str, output_dir: str, fps: float):
        """Save a few thumbnail frames per detected scene using OpenCV."""
        if not self.detected_scenes:
            logger.warning("No scenes available for thumbnail export.")
            return

        try:
            import cv2
        except Exception as e:
            raise RuntimeError("OpenCV (cv2) is required for thumbnail export. Install opencv-python.") from e

        base = Path(video_path).stem
        images_dir = Path(output_dir) / f"{base}_images"
        images_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open video for thumbnail extraction.")

        num_images = int(self.num_images_var.get())
        margin = int(self.frame_margin_var.get())
        num_images = max(1, num_images)
        margin = max(0, margin)

        def pick_frames(start_f: int, end_f_excl: int) -> list[int]:
            # end_f_excl is exclusive
            if end_f_excl <= start_f + 1:
                return [start_f]
            lo = start_f + margin
            hi = end_f_excl - margin - 1
            if hi < lo:
                mid = (start_f + end_f_excl) // 2
                return [max(start_f, min(mid, end_f_excl - 1))]
            span = hi - lo + 1
            if num_images == 1:
                return [lo + span // 2]
            # Evenly spaced across [lo, hi]
            return [int(round(lo + (span - 1) * (i / (num_images - 1)))) for i in range(num_images)]

        try:
            for scene_idx, (st, et) in enumerate(self.detected_scenes, start=1):
                if self.abort_flag.is_set():
                    logger.warning("Image export aborted by user.")
                    break

                start_f = int(st.get_frames())
                end_f = int(et.get_frames())
                frames = pick_frames(start_f, end_f)

                for j, fidx in enumerate(frames, start=1):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        continue
                    out_path = images_dir / f"{base}_scene_{scene_idx:04d}_img_{j:02d}_f{fidx:06d}.jpg"
                    cv2.imwrite(str(out_path), frame)
        finally:
            cap.release()

        logger.info("Saved thumbnails to directory: %s", images_dir)

    def _split_video(self, video_path: str, output_dir: str, fps: float):
        """Split the input video into per-scene clips with frame-exact boundaries using FFmpeg."""
        if not self.detected_scenes:
            logger.warning("No scenes available to split.")
            return

        total_scenes = len(self.detected_scenes)
        logger.info(
            "Splitting video into %d scenes (frame-exact) using per-scene FFmpeg commands...",
            total_scenes,
        )

        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

        def probe_has_audio(path: str) -> bool:
            """Best-effort check for an audio stream using ffprobe."""
            try:
                probe = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "a",
                        "-show_entries",
                        "stream=index",
                        "-of",
                        "csv=p=0",
                        path,
                    ],
                    capture_output=True,
                    text=True,
                    creationflags=creationflags,
                    check=False,
                )
                return probe.returncode == 0 and bool(probe.stdout.strip())
            except FileNotFoundError:
                # If ffprobe isn't available, assume audio exists and let FFmpeg tell us otherwise.
                return True

        input_has_audio = probe_has_audio(video_path)

        for i, (start_tc, end_tc) in enumerate(self.detected_scenes):
            scene_num = i + 1
            if self.abort_flag.is_set():
                logger.warning("Video splitting aborted by user.")
                return

            # progress: 85 -> 100 across splitting
            self.progress_queue.put_nowait(
                (85 + (scene_num / max(1, total_scenes)) * 15, f"Splitting scene {scene_num}/{total_scenes}")
            )

            output_filename = Path(output_dir) / f"{Path(video_path).stem}-Scene-{scene_num:03d}.mp4"

            # Frame-exact boundaries from detected scene list.
            start_frame = int(start_tc.get_frames())
            end_frame = int(end_tc.get_frames())  # exclusive end

            # For audio trimming we use seconds; FFmpeg accepts floats here.
            start_sec = float(start_tc.get_seconds())
            end_sec = float(end_tc.get_seconds())

            codec = self.ffmpeg_codec_var.get()

            # --- Build the Command for a Single Scene ---
            command = ["ffmpeg", "-y", "-hide_banner", "-i", video_path]

            # Filter graph: trim by frame for video; trim by time for audio (if present).
            if input_has_audio:
                filter_graph = (
                    f"[0:v]trim=start_frame={start_frame}:end_frame={end_frame},setpts=PTS-STARTPTS[v];"
                    f"[0:a]atrim=start={start_sec}:end={end_sec},asetpts=PTS-STARTPTS[a]"
                )
                command += ["-filter_complex", filter_graph, "-map", "[v]", "-map", "[a]"]
            else:
                filter_graph = f"[0:v]trim=start_frame={start_frame}:end_frame={end_frame},setpts=PTS-STARTPTS[v]"
                command += ["-filter_complex", filter_graph, "-map", "[v]"]

            # Video encoder settings
            if "nvenc" in codec:
                command += [
                    "-c:v",
                    codec,
                    "-preset",
                    self.ffmpeg_preset_var.get(),
                    "-qp",
                    str(self.ffmpeg_cq_var.get()),
                ]
            else:
                command += [
                    "-c:v",
                    codec,
                    "-preset",
                    "fast",
                    "-crf",
                    str(self.ffmpeg_cq_var.get()),
                ]

            # Audio: must be re-encoded after filtering (cannot stream-copy filtered output).
            if input_has_audio:
                command += ["-c:a", "aac", "-b:a", "192k"]

            # Helpful for MP4 playback while downloading/streaming.
            command += ["-movflags", "+faststart", str(output_filename)]

            logger.debug("Executing FFmpeg command: %s", " ".join(command))

            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    creationflags=creationflags,
                    check=False,
                )

                # If we assumed audio exists but it doesn't, retry once without audio mapping/filtering.
                if result.returncode != 0 and input_has_audio:
                    stderr_lower = (result.stderr or "").lower()
                    if (
                        "matches no streams" in stderr_lower
                        or "stream specifier" in stderr_lower
                        or "cannot find a matching stream" in stderr_lower
                    ):
                        logger.warning(
                            "FFmpeg reported no audio stream; retrying scene %d without audio.",
                            scene_num,
                        )
                        filter_graph = f"[0:v]trim=start_frame={start_frame}:end_frame={end_frame},setpts=PTS-STARTPTS[v]"
                        command_retry = [
                            "ffmpeg",
                            "-y",
                            "-hide_banner",
                            "-i",
                            video_path,
                            "-filter_complex",
                            filter_graph,
                            "-map",
                            "[v]",
                        ]

                        if "nvenc" in codec:
                            command_retry += [
                                "-c:v",
                                codec,
                                "-preset",
                                self.ffmpeg_preset_var.get(),
                                "-qp",
                                str(self.ffmpeg_cq_var.get()),
                            ]
                        else:
                            command_retry += [
                                "-c:v",
                                codec,
                                "-preset",
                                "fast",
                                "-crf",
                                str(self.ffmpeg_cq_var.get()),
                            ]

                        command_retry += ["-movflags", "+faststart", str(output_filename)]

                        logger.debug("Retrying FFmpeg command: %s", " ".join(command_retry))
                        result = subprocess.run(
                            command_retry,
                            capture_output=True,
                            text=True,
                            creationflags=creationflags,
                            check=False,
                        )

                if result.returncode != 0:
                    logger.error(
                        "FFmpeg failed on scene %d with return code %d",
                        scene_num,
                        result.returncode,
                    )
                    logger.error("FFmpeg stderr:")
                    logger.error("%s", result.stderr)

                    msg = f"""FFmpeg failed while splitting scene {scene_num}.

    Return code: {result.returncode}

    See console for details."""
                    messagebox.showerror("FFmpeg Error", msg)
                    return

            except Exception as e:
                logger.exception("An error occurred during video splitting.")
                messagebox.showerror("Splitting Error", f"An error occurred: {e}")
                return

        logger.info("Successfully split video into %d scenes.", total_scenes)

    def _process_queues(self) -> None:
        while not self.progress_queue.empty():
            progress, message = self.progress_queue.get_nowait()
            try:
                self.progress_var.set(progress)
            except Exception:
                # progress_var might be an IntVar; fall back
                self.progress_var.set(int(progress))
            self.progress_label.config(text=message)

            # A task is considered "finished" if it completes successfully, is aborted,
            # errors out, or finds no scenes. In any of these cases, reset the UI.
            is_finished = (
                "Done!" in message
                or "Aborted" in message
                or "Error" in message
                or "No scenes found" in message
            )
            if float(progress) >= 100 or is_finished:
                self.start_button.config(state=tk.NORMAL)
                self.abort_button.config(state=tk.DISABLED)
                if "Done!" in message:
                    messagebox.showinfo("Success", message)
        self.root.after(200, self._process_queues)
def main() -> None:
    root = tk.Tk()
    app = SceneDetectApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
