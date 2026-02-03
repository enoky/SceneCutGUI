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
# Optional TransNetV2 dependency (recommended detector)
# ------------------------------------------------------------------ #
try:
    import torch  # noqa: F401
    from transnetv2_pytorch import TransNetV2
except Exception as e:
    logger.error("Failed to import TransNetV2 (transnetv2-pytorch). Please ensure it is installed: %s", e)
    TransNetV2 = None

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


def transnetv2_scenes_to_timecodes(
    tn_scenes: list,
    fps: float,
    total_frames: int,
) -> list[tuple[Timecode, Timecode]]:
    """Convert TransNetV2 output dicts into [(start_tc, end_tc_excl), ...]."""
    scenes_out: list[tuple[Timecode, Timecode]] = []

    last_end = 0
    for s in tn_scenes:
        # Prefer explicit frame indices if available
        if isinstance(s, dict) and "start_frame" in s and "end_frame" in s:
            start_f = int(s["start_frame"])
            end_f = int(s["end_frame"])
        else:
            # Fallback to seconds
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
        self.detector_type_var = tk.StringVar(value="TransNetV2")
        self.params_vars: dict = {}
        self.ai_validate_var = tk.BooleanVar(value=False)
        self.ai_window_var = tk.IntVar(value=5)
        self.flash_sensitivity_var = tk.IntVar(value=30)  # Luma delta threshold for flash detection

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

        ttk.Label(config_frame, text="Detector:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        detector_label = ttk.Label(config_frame, text="TransNetV2 (PyTorch)")
        detector_label.grid(row=0, column=0, padx=(65, 5), pady=5, sticky="w")
        self._attach_tooltip(detector_label, "detector_type")
        
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
        ai_check = ttk.Checkbutton(ai_frame, text="Validate cuts with DINOv3 (filters flashes/fast motion)", variable=self.ai_validate_var)
        ai_check.grid(row=0, column=0, columnspan=3, padx=5, pady=5, sticky="w")
        self._attach_tooltip(ai_check, "ai_validate")
        ttk.Label(ai_frame, text="Validation Window (frames):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ai_spin = ttk.Spinbox(ai_frame, from_=2, to=10, textvariable=self.ai_window_var, width=5)
        ai_spin.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self._attach_tooltip(ai_spin, "ai_window")
        ttk.Label(ai_frame, text="Flash Sensitivity:").grid(row=1, column=2, padx=(20, 5), pady=5, sticky="w")
        flash_spin = ttk.Spinbox(ai_frame, from_=15, to=80, textvariable=self.flash_sensitivity_var, width=5)
        flash_spin.grid(row=1, column=3, padx=5, pady=5, sticky="w")
        self._attach_tooltip(flash_spin, "flash_sensitivity")

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

    def _save_settings(self):
        """Save all GUI settings to a JSON file."""
        settings = {
            "video_path": self.video_path_var.get(),
            "output_dir": self.output_dir_var.get(),
            "detector_type": self.detector_type_var.get(),
            "detector_params": {key: var.get() for key, var in self.params_vars.items()},
            "ai_validate": self.ai_validate_var.get(),
            "ai_window": self.ai_window_var.get(),
            "flash_sensitivity": self.flash_sensitivity_var.get(),
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
            self.flash_sensitivity_var.set(settings.get("flash_sensitivity", 30))
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
            
            # Set detector type first, which rebuilds the params UI
            detector_type = settings.get("detector_type", "TransNetV2")
            self.detector_type_var.set("TransNetV2")
            self._build_detector_params() # Ensure params_vars is populated

            # Now set the specific parameters for that detector
            if "detector_params" in settings:
                loaded_params = settings["detector_params"]
                for key, var in self.params_vars.items():
                    if key in loaded_params:
                        var.set(loaded_params[key])

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
        # Detector is fixed to TransNetV2.
        return

    def _build_detector_params(self):
        """Build TransNetV2 parameter widgets."""
        for widget in self.params_frame.winfo_children():
            widget.destroy()

        # Clear and rebuild vars
        self.params_vars.clear()

        # Row 0: device
        ttk.Label(self.params_frame, text="device:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        d_var = tk.StringVar(value="auto")
        d_combo = ttk.Combobox(self.params_frame, textvariable=d_var, values=["auto", "cpu", "cuda", "mps"], state="readonly", width=10)
        d_combo.grid(row=0, column=1, padx=5, pady=2, sticky="w")
        self._attach_tooltip(d_combo, "device")

        # Row 1: threshold
        ttk.Label(self.params_frame, text="threshold:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        t_var = tk.StringVar(value="0.5")
        t_entry = ttk.Entry(self.params_frame, textvariable=t_var, width=10)
        t_entry.grid(row=1, column=1, padx=5, pady=2, sticky="w")
        self._attach_tooltip(t_entry, "threshold")

        # Row 2: min_scene_len
        ttk.Label(self.params_frame, text="min_scene_len (frames):").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        m_var = tk.StringVar(value="8")
        m_entry = ttk.Entry(self.params_frame, textvariable=m_var, width=10)
        m_entry.grid(row=2, column=1, padx=5, pady=2, sticky="w")
        self._attach_tooltip(m_entry, "min_scene_len")

        # Save vars
        self.params_vars["device"] = d_var
        self.params_vars["threshold"] = t_var
        self.params_vars["min_scene_len"] = m_var


    def _start_detection(self) -> None:
        video_path = self.video_path_var.get()
        output_dir = self.output_dir_var.get()
        if not video_path or not output_dir:
            messagebox.showerror("Error", "Please select a video file and an output folder.")
            return
        if TransNetV2 is None:
            messagebox.showerror("Error", "TransNetV2 (transnetv2-pytorch) is not installed.")
            return

        try:
            weights_path = "./weights/transnetv2-pytorch-weights.pth"
            device = str(self.params_vars.get("device").get()).strip() or "auto"
            threshold = float(self.params_vars.get("threshold").get())
            min_scene_len = int(float(self.params_vars.get("min_scene_len").get()))
            if not (0.0 < threshold < 1.0):
                raise ValueError("threshold must be between 0 and 1.")
            if min_scene_len < 1:
                min_scene_len = 1
        except Exception as exc:
            logger.error("Invalid TransNetV2 parameters: %s", exc)
            messagebox.showerror("Parameter Error", f"Invalid TransNetV2 parameters: {exc}")
            return

        detector_cfg = {
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
        logger.info("Starting detection process (TransNetV2)...")

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

            # --- 2. Detection (TransNetV2) ---
            self.progress_queue.put((10, "Loading TransNetV2..."))
            import torch

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
                tn_scenes = model.detect_scenes(video_path, threshold=float(detector.get("threshold", 0.5)))

            if self.abort_flag.is_set():
                raise InterruptedError

            self.progress_queue.put((60, f"TransNetV2 returned {len(tn_scenes)} scenes. Converting..."))
            scenes = transnetv2_scenes_to_timecodes(tn_scenes, fps=fps, total_frames=total_frames)

            # Enforce minimum scene length (frames)
            min_len = int(detector.get("min_scene_len", 1) or 1)
            if min_len > 1 and scenes:
                filtered = []
                for st, et in scenes:
                    if (et.get_frames() - st.get_frames()) >= min_len:
                        filtered.append((st, et))
                # Keep at least one
                scenes = filtered or [scenes[0]]

            logger.info("Detected %d raw scenes (TransNetV2)", len(scenes))

            # --- 3. AI Validation (Optional) ---
            if self.ai_validate_var.get() and scenes:
                self.progress_queue.put((70, "AI validating cuts..."))
                scenes = self._run_ai_validation(video_path, scenes)

            self.detected_scenes = scenes
            if not self.detected_scenes:
                logger.warning("No scenes were detected.")
                self.progress_queue.put((100, "No scenes found."))
                return

            # --- 4. Output Generation ---
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

    def _run_ai_validation(self, video_path, scenes):
        """Validate potential cuts using DINOv3 + practical heuristics.

        Practical accuracy upgrades vs the original cosine(mean(pre), mean(post)) < 0.88:
          - Sample *away* from the cut boundary (gap) to avoid blurred/dissolve frames.
          - Multi-scale validation (small + large windows) to handle fades/fast motion.
          - Robust similarity score: median pairwise similarity (pre x post), not mean vectors.
          - Per-video adaptive merge threshold (Otsu + percentiles) instead of fixed 0.88.
          - Flash + near-black gating and an adjacent-frame continuity check.

        Returns a new scene list with likely false-positive cuts merged.
        """
        try:
            import cv2
            import torch
            import numpy as np
            from transformers import AutoImageProcessor, AutoModel

            class DinoCutValidator:
                """
                DINOv3-based cut validator that filters false-positive cuts (flashes / fast motion / near-black).

                Drop-in notes:
                  - Expects: os, logging, cv2, torch, numpy as np
                  - Expects: from transformers import AutoImageProcessor, AutoModel
                  - You can pass your app logger in; otherwise it creates its own.
                """

                # Conservative defaults: only merge when the evidence is strong.
                DEFAULT_BASE_THRESHOLD = 0.88

                # Windows & sampling
                MAX_LARGE_WINDOW = 24

                # Decision margins
                MERGE_MARGIN = 0.01

                # Cheap cues
                FLASH_LUMA_DELTA = 30.0  # Lower default for better flash detection
                FLASH_HIGH_LUMA = 200.0  # White-flash detection threshold
                NEAR_BLACK_LUMA = 18.0

                # Adjacent-frame continuity gating
                ADJ_KEEP_CUTOFF = 0.70
                ADJ_MERGE_CUTOFF = 0.78

                # Confidence / motion adjustments
                MOTION_BOOST = 0.06      # threshold += MOTION_BOOST * motion_score
                BLACK_BOOST = 0.03       # threshold += BLACK_BOOST if near-black

                # ANSI color helpers for console output (optional)
                KEEP_FG = "\033[94m"
                MERG_FG = "\033[92m"
                BOLD = "\033[1m"
                RESET = "\033[0m"

                def __init__(self, model_dir: str = "./weights", device=None, batch_size: int = 48, flash_luma_delta: float = 30.0, logger=None):
                    import os
                    import logging
                    import torch
                    from transformers import AutoImageProcessor, AutoModel

                    self._log = logger or logging.getLogger("DinoCutValidator")

                    if not os.path.isdir(model_dir):
                        raise FileNotFoundError(f"DINOv3 model directory not found: {model_dir}")

                    self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
                    if self.device == "cpu":
                        raise RuntimeError("AI Validation requires a CUDA-enabled GPU and PyTorch.")

                    self._log.debug("Loading DINOv3 model on device: %s", self.device)
                    self.processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
                    self.model = AutoModel.from_pretrained(model_dir, local_files_only=True).to(self.device).eval()

                    # Cache
                    self.cache = {}         # frame_idx -> torch.Tensor embedding (normalized)
                    self.luma_cache = {}    # frame_idx -> float luma mean (0-255)
                    self.batch_size = int(batch_size)
                    self.total_video_frames = 0
                    
                    # Configurable flash sensitivity (luma delta threshold)
                    self.flash_luma_delta = float(flash_luma_delta)

                    # Precompute normalization tensors
                    self._mean = torch.tensor(self.processor.image_mean, device=self.device).view(1, 3, 1, 1)
                    self._std = torch.tensor(self.processor.image_std, device=self.device).view(1, 3, 1, 1)

                def _target_size(self):
                    # Transformers image processor may define size in a few forms.
                    size = getattr(self.processor, "size", None) or {}
                    if isinstance(size, dict):
                        h = int(size.get("height") or size.get("shortest_edge") or 224)
                        w = int(size.get("width") or size.get("shortest_edge") or 224)
                    else:
                        h = w = 224
                    # OpenCV uses (width, height)
                    return (w, h)

                @staticmethod
                def _luma_from_rgb(rgb_uint8) -> float:
                    # rgb_uint8: HxWx3 uint8; Rec. 709 luma coefficients.
                    import numpy as np

                    r = rgb_uint8[:, :, 0].astype(np.float32)
                    g = rgb_uint8[:, :, 1].astype(np.float32)
                    b = rgb_uint8[:, :, 2].astype(np.float32)
                    return float((0.2126 * r + 0.7152 * g + 0.0722 * b).mean())

                def _embed_batch(self, images_tensor):
                    import torch

                    # images_tensor: (B, 3, H, W), float32 0-255
                    with torch.inference_mode():
                        images_tensor = (images_tensor / 255.0 - self._mean) / self._std
                        outputs = self.model(pixel_values=images_tensor)
                        pooled = outputs.pooler_output
                        return torch.nn.functional.normalize(pooled, dim=-1)

                def _embed_all_required_frames(self, video_path: str, indices_to_embed, abort_flag=None, progress_cb=None):
                    import cv2
                    import torch

                    indices = sorted(set(int(i) for i in indices_to_embed))
                    if not indices:
                        return

                    self._log.info("Embedding %d unique frames for validation...", len(indices))
                    indices_set = set(indices)

                    target_size = self._target_size()
                    batch, batch_ids = [], []

                    # Prefer GPU decode, but fall back to CPU decode if unavailable.
                    use_cuda_decode = True
                    try:
                        gpu_frame = cv2.cuda_GpuMat()
                        cap = cv2.cudacodec.createVideoReader(video_path)
                    except Exception as e:
                        use_cuda_decode = False
                        self._log.warning("CUDA video reader unavailable (falling back to CPU decode): %s", e)
                        cap = cv2.VideoCapture(video_path)

                    def flush_batch():
                        nonlocal batch, batch_ids
                        if not batch:
                            return
                        images = torch.stack(batch, dim=0)
                        embeddings = self._embed_batch(images)
                        for j, fid in enumerate(batch_ids):
                            self.cache[fid] = embeddings[j]
                        batch, batch_ids = [], []

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
                                # cudacodec often returns BGRA; try BGRA->RGB, else BGR->RGB.
                                try:
                                    rgb_gpu = cv2.cuda.cvtColor(resized_gpu, cv2.COLOR_BGRA2RGB)
                                except Exception:
                                    rgb_gpu = cv2.cuda.cvtColor(resized_gpu, cv2.COLOR_BGR2RGB)

                                rgb = rgb_gpu.download()

                                # Ensure we have 3 channels
                                if rgb.ndim == 3 and rgb.shape[2] == 4:
                                    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGRA2RGB)
                                elif rgb.ndim == 3 and rgb.shape[2] == 3:
                                    pass
                                else:
                                    frame_idx += 1
                                    continue

                                self.luma_cache[frame_idx] = self._luma_from_rgb(rgb)

                                t = torch.as_tensor(rgb, device=self.device)
                                t = t.permute(2, 0, 1).contiguous().float()
                                batch.append(t)
                                batch_ids.append(frame_idx)
                                wanted_i += 1
                        else:
                            ret, frame = cap.read()
                            if not ret:
                                break

                            if frame_idx in indices_set:
                                frame = cv2.resize(frame, target_size)
                                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                self.luma_cache[frame_idx] = self._luma_from_rgb(rgb)

                                t = torch.from_numpy(rgb).to(self.device)
                                t = t.permute(2, 0, 1).contiguous().float()
                                batch.append(t)
                                batch_ids.append(frame_idx)
                                wanted_i += 1

                        if len(batch) >= self.batch_size:
                            flush_batch()

                        if progress_cb and total_wanted > 0 and frame_idx % 50 == 0:
                            progress_cb(min(1.0, wanted_i / total_wanted), f"Embedding frames ({wanted_i}/{total_wanted})")

                        frame_idx += 1

                    flush_batch()

                    if not use_cuda_decode:
                        cap.release()

                    self._log.info("Frame embedding complete (%d cached).", len(self.cache))

                @staticmethod
                def _pairwise_median_similarity(pre_vecs, post_vecs) -> float:
                    import torch

                    # pre/post vectors are already normalized.
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

                    # scores in [0, 1]
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
                        # Not much signal in this video; keep conservative.
                        return float(np.clip(float(np.mean(s)) + 0.01, 0.86, 0.93))

                    t_otsu = self._otsu_threshold(s)
                    p75 = float(np.percentile(s, 75))
                    p90 = float(np.percentile(s, 90))

                    # Keep threshold in a safe, high-similarity band.
                    thr = max(p75, min(t_otsu, p90))
                    return float(np.clip(thr, 0.86, 0.94))

                def _window_indices(self, cut_frame: int, window: int, gap: int, total_frames: int):
                    # Sample away from boundary by 'gap' frames.
                    pre_end = max(0, cut_frame - gap)
                    pre_start = max(0, pre_end - window)
                    post_start = min(total_frames, cut_frame + gap)
                    post_end = min(total_frames, post_start + window)
                    return list(range(pre_start, pre_end)), list(range(post_start, post_end))

                def _cut_features(self, cut_frame: int, window: int, total_frames: int):
                    gap = max(2, int(window))
                    w_small = max(2, int(window))
                    w_large = int(min(max(w_small * 3, w_small + 2), self.MAX_LARGE_WINDOW))

                    # Small + large windows
                    pre_s, post_s = self._window_indices(cut_frame, w_small, gap, total_frames)
                    pre_l, post_l = self._window_indices(cut_frame, w_large, gap, total_frames)

                    # Adjacent frames right at the cut
                    pre_adj = cut_frame - 1
                    post_adj = cut_frame

                    def vecs_for(indices):
                        return [self.cache[i] for i in indices if i in self.cache]

                    pre_vecs_s = vecs_for(pre_s)
                    post_vecs_s = vecs_for(post_s)
                    pre_vecs_l = vecs_for(pre_l)
                    post_vecs_l = vecs_for(post_l)

                    # Must have at least some evidence
                    if len(pre_vecs_s) < 2 or len(post_vecs_s) < 2:
                        return None

                    s_small = self._pairwise_median_similarity(pre_vecs_s, post_vecs_s)

                    # Large window may not be fully available near start/end
                    s_large = None
                    if len(pre_vecs_l) >= 2 and len(post_vecs_l) >= 2:
                        s_large = self._pairwise_median_similarity(pre_vecs_l, post_vecs_l)

                    # Combine scales
                    s_comb = float(s_small) if s_large is None else float(0.65 * s_small + 0.35 * s_large)

                    # Adjacent similarity for discontinuity spike
                    s_adj = None
                    if pre_adj in self.cache and post_adj in self.cache:
                        s_adj = float((self.cache[pre_adj] * self.cache[post_adj]).sum().item())

                    # Motion confidence (low adjacent similarity inside window => high motion)
                    pre_motion = self._avg_adjacent_similarity(pre_vecs_s)
                    post_motion = self._avg_adjacent_similarity(post_vecs_s)
                    motion_score = float(max(0.0, 1.0 - 0.5 * (pre_motion + post_motion)))

                    # Luma gating cues - enhanced with multi-frame and white-flash detection
                    luma_pre = self.luma_cache.get(pre_adj)
                    luma_post = self.luma_cache.get(post_adj)
                    flash = False
                    near_black = False
                    
                    if luma_pre is not None and luma_post is not None:
                        # Basic single-frame luma delta check (using configurable threshold)
                        flash = abs(luma_pre - luma_post) >= self.flash_luma_delta
                        near_black = (luma_pre <= self.NEAR_BLACK_LUMA) or (luma_post <= self.NEAR_BLACK_LUMA)
                        
                        # White-flash detection: one frame is very bright while the other isn't
                        is_white_flash = (
                            (luma_pre > self.FLASH_HIGH_LUMA) != (luma_post > self.FLASH_HIGH_LUMA)
                        )
                        flash = flash or is_white_flash
                    
                    # Multi-frame flash detection: check luma spike within a ±2 frame window
                    if not flash:
                        flash_window_frames = [cut_frame - 2, cut_frame - 1, cut_frame, cut_frame + 1]
                        lumas = [self.luma_cache.get(f) for f in flash_window_frames if f in self.luma_cache]
                        if len(lumas) >= 3:
                            luma_range = max(lumas) - min(lumas)
                            if luma_range >= self.flash_luma_delta:
                                flash = True

                    return {
                        "s_small": float(s_small),
                        "s_large": None if s_large is None else float(s_large),
                        "s_comb": float(s_comb),
                        "s_adj": None if s_adj is None else float(s_adj),
                        "motion": float(motion_score),
                        "flash": bool(flash),
                        "near_black": bool(near_black),
                        "gap": int(gap),
                        "w_small": int(w_small),
                        "w_large": int(w_large),
                    }

                def validate_cut(self, scene_index: int, cut_frame: int, window: int, total_frames: int, base_thr: float) -> bool:
                    """Return True if it's a true cut (KEEP), False if likely a false-positive (MERGE)."""
                    feats = self._cut_features(cut_frame, window, total_frames)
                    if feats is None:
                        return True

                    s_comb = feats["s_comb"]
                    s_adj = feats["s_adj"]
                    motion = feats["motion"]
                    flash = feats["flash"]
                    near_black = feats["near_black"]

                    # Dynamic threshold: motion & near-black make us *more conservative* about merging.
                    thr = float(base_thr + self.MOTION_BOOST * motion + (self.BLACK_BOOST if near_black else 0.0))
                    thr = float(min(max(thr, 0.80), 0.97))

                    # Strong adjacent discontinuity almost always means a true cut.
                    if s_adj is not None and s_adj < self.ADJ_KEEP_CUTOFF:
                        decision_keep = True
                        reason = "adj_discontinuity"
                    else:
                        # Flash-like spike + reasonably high similarity => likely false positive.
                        # More aggressive flash gating: removed s_adj requirement for easier triggering
                        if flash and s_comb >= (thr - 0.02):
                            decision_keep = False
                            reason = "flash_gate"
                        else:
                            # Default: merge only when similarity is clearly above threshold.
                            if s_comb >= (thr + self.MERGE_MARGIN) and (s_adj is None or s_adj >= self.ADJ_MERGE_CUTOFF):
                                # If near-black, require even more evidence.
                                if near_black and s_comb < (thr + 0.03):
                                    decision_keep = True
                                    reason = "near_black_conservative"
                                else:
                                    decision_keep = False
                                    reason = "high_similarity"
                            else:
                                decision_keep = True
                                reason = "below_threshold"

                    decision = (
                        f"{self.BOLD}{self.KEEP_FG}KEEP{self.RESET}"
                        if decision_keep
                        else f"{self.BOLD}{self.MERG_FG}MERGE{self.RESET}"
                    )
                    self._log.debug(
                        "Scene #%d cut@%d | s_comb=%.3f (small=%.3f large=%s) s_adj=%s motion=%.2f thr=%.3f flash=%s black=%s -> %s (%s)",
                        scene_index,
                        cut_frame,
                        s_comb,
                        feats["s_small"],
                        "{:.3f}".format(feats["s_large"]) if feats["s_large"] is not None else "n/a",
                        "{:.3f}".format(s_adj) if s_adj is not None else "n/a",
                        motion,
                        thr,
                        str(flash),
                        str(near_black),
                        decision,
                        reason,
                    )

                    return decision_keep

                def filter_scenes(self, video_path: str, scenes: list, window: int, abort_flag=None, progress_cb=None) -> list:
                    import cv2

                    if len(scenes) < 2:
                        return scenes

                    cap = cv2.VideoCapture(video_path)
                    self.total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    cap.release()

                    if self.total_video_frames <= 0:
                        self._log.warning("Could not determine frame count for AI validation; skipping.")
                        return scenes

                    # Collect all unique frame indices required for all candidate cuts.
                    all_indices = set()
                    cut_frames = []
                    for _, end_tc in scenes[:-1]:
                        cut = int(end_tc.get_frames())
                        cut_frames.append(cut)

                    w = int(window)
                    gap = max(2, w)
                    w_small = max(2, w)
                    w_large = int(min(max(w_small * 3, w_small + 2), self.MAX_LARGE_WINDOW))

                    for cut in cut_frames:
                        # Adjacent frames at boundary
                        all_indices.add(max(0, cut - 1))
                        all_indices.add(min(self.total_video_frames - 1, cut))

                        # Small window (gapped)
                        pre_s, post_s = self._window_indices(cut, w_small, gap, self.total_video_frames)
                        all_indices.update(pre_s)
                        all_indices.update(post_s)

                        # Large window (gapped)
                        pre_l, post_l = self._window_indices(cut, w_large, gap, self.total_video_frames)
                        all_indices.update(pre_l)
                        all_indices.update(post_l)

                    # Embed once.
                    self._embed_all_required_frames(
                        video_path,
                        sorted(all_indices),
                        abort_flag=abort_flag,
                        progress_cb=progress_cb,
                    )

                    # First pass: compute per-cut combined similarity for adaptive thresholding.
                    cut_scores = []
                    for i, cut in enumerate(cut_frames, start=1):
                        feats = self._cut_features(cut, w, self.total_video_frames)
                        if feats is None:
                            continue
                        cut_scores.append(feats["s_comb"])
                        if progress_cb and i % 10 == 0:
                            progress_cb(min(1.0, i / max(1, len(cut_frames))), f"Analyzing cut scores ({i}/{len(cut_frames)})")

                    base_thr = self._auto_threshold(cut_scores)
                    self._log.info(
                        "AI validation adaptive threshold: base_thr=%.3f (window=%d, gap=%d, large=%d)",
                        base_thr,
                        w,
                        gap,
                        w_large,
                    )

                    # Second pass: merge sequentially.
                    validated_scenes = []
                    cur_start, cur_end = scenes[0]

                    for i in range(1, len(scenes)):
                        nxt_start, nxt_end = scenes[i]
                        cut_frame_num = int(nxt_start.get_frames())

                        if abort_flag is not None and abort_flag.is_set():
                            raise InterruptedError

                        keep_cut = self.validate_cut(
                            scene_index=i,
                            cut_frame=cut_frame_num,
                            window=w,
                            total_frames=self.total_video_frames,
                            base_thr=base_thr,
                        )

                        if keep_cut:
                            validated_scenes.append((cur_start, cur_end))
                            cur_start, cur_end = nxt_start, nxt_end
                        else:
                            cur_end = nxt_end

                        if progress_cb:
                            progress_cb(min(1.0, i / max(1, len(scenes) - 1)), f"Validating cuts ({i}/{len(scenes) - 1})")

                    validated_scenes.append((cur_start, cur_end))
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

        html = (
            "<!doctype html><html><head><meta charset='utf-8'/>"
            "<title>Scene List</title>"
            "<style>"
            "body{font-family:Arial, sans-serif; padding:16px;}"
            "table{border-collapse:collapse; width:100%;}"
            "th,td{border:1px solid #ccc; padding:6px 8px; text-align:left;}"
            "th{background:#f3f3f3;}"
            "</style></head><body>"
            "<h2>Detected Scenes (TransNetV2)</h2>"
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
