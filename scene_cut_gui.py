import threading
import queue
import os
import sys
import logging
import subprocess
import functools
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
# OmniShotCut dependency
# ------------------------------------------------------------------ #
OMNISHOT_IMPORT_ERROR = None
omnishotcut = None
try:
    if torch is None:
        raise RuntimeError("torch is not available")
    OMNI_SHOT_DIR = Path(__file__).resolve().parent / "OmniShotCut"
    if OMNI_SHOT_DIR.exists():
        sys.path.insert(0, str(OMNI_SHOT_DIR))
    import omnishotcut  # noqa: F401
except Exception as e:
    OMNISHOT_IMPORT_ERROR = e
    logger.error(
        "Failed to import OmniShotCut. Ensure the OmniShotCut folder is present and "
        "'ffmpeg-python' / 'huggingface_hub' are installed: %s",
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


def _parse_fraction_exact(frac: str) -> tuple[int, int]:
    """Parse an ffprobe rational ('30000/1001') into an exact (num, den) pair.

    Returns (0, 0) when the value cannot be interpreted, so callers can fall
    back to a float frame rate.
    """
    try:
        if not frac:
            return (0, 0)
        if "/" in frac:
            num, den = frac.split("/", 1)
            num_i, den_i = int(num), int(den)
            return (num_i, den_i) if num_i > 0 and den_i > 0 else (0, 0)
        num_i = int(round(float(frac) * 1000))
        return (num_i, 1000) if num_i > 0 else (0, 0)
    except Exception:
        return (0, 0)


# How far before a scene start to place the fast input seek. Correctness does
# not depend on this (input seeking always lands on a keyframe at or before the
# target); it only absorbs imprecise container indexes.
SEEK_MARGIN_SEC = 0.5


@functools.lru_cache(maxsize=1)
def has_nvenc_hevc() -> bool:
    """True when this FFmpeg build exposes the NVIDIA hevc_nvenc encoder."""
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    try:
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-hide_banner", "-encoders"],
            capture_output=True, text=True, creationflags=creationflags, check=False,
        )
        return "hevc_nvenc" in (result.stdout or "")
    except FileNotFoundError:
        return False


def verify_cfr(video_path: str, fps_num: int, fps_den: int) -> bool:
    """Check that every video packet actually sits on the constant-frame-rate grid.

    Comparing r_frame_rate to avg_frame_rate is not enough: containers routinely
    report both as a clean value (e.g. 30/1) for material whose real timestamps
    are irregular, and acting on that false positive makes timestamp-based
    trimming select the wrong frames. Reading packet timestamps settles it - this
    inspects the container index only and never decodes.
    """
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "packet=pts_time", "-of", "csv=p=0", video_path],
            capture_output=True, text=True, creationflags=creationflags, check=False,
        )
        if result.returncode != 0:
            return False
        stamps = []
        for line in (result.stdout or "").split():
            try:
                stamps.append(float(line))
            except ValueError:
                # "N/A" means the container has no usable timestamp for a packet,
                # which by itself rules out timestamp-based trimming.
                return False
        stamps.sort()
    except FileNotFoundError:
        return False

    if len(stamps) < 2:
        return False
    # Packets arrive in decode order, so sort into presentation order first, then
    # require every frame within half a frame of its ideal slot.
    frame_dur = fps_den / fps_num
    tolerance = 0.5 * frame_dur
    origin = stamps[0]
    for i, pts in enumerate(stamps):
        if abs(pts - (origin + i * frame_dur)) > tolerance:
            return False
    return True


def can_hwdecode(video_path: str) -> bool:
    """Test whether this source can run the full NVDEC -> CUDA -> NVENC pipeline.

    NVDEC does not cover every codec and chroma layout (4:2:2, ProRes, DNxHD and
    AV1 on older GPUs are common gaps), so rather than guess from the codec name
    we run the real filter chain over a single frame and see if it survives.
    """
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    try:
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-hide_banner",
             "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
             "-i", video_path, "-frames:v", "1",
             "-filter_complex", "[0:v]scale_cuda=format=p010le[v]", "-map", "[v]",
             "-c:v", "hevc_nvenc", "-f", "null", "-"],
            capture_output=True, text=True, creationflags=creationflags, check=False,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def probe_stream_props(video_path: str) -> dict:
    """Probe the properties needed for a frame-exact, colour-faithful re-encode."""
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    props: dict = {"has_audio": False, "fps_num": 0, "fps_den": 0,
                   "start_time": 0.0, "is_cfr": False}
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=codec_type,r_frame_rate,avg_frame_rate,start_time,"
             "color_range,color_space,color_transfer,color_primaries",
             "-of", "json", video_path],
            capture_output=True, text=True, creationflags=creationflags, check=False,
        )
        if result.returncode != 0:
            return props
        for stream in json.loads(result.stdout or "{}").get("streams", []):
            if stream.get("codec_type") == "audio":
                props["has_audio"] = True
            elif stream.get("codec_type") == "video" and not props["fps_num"]:
                r_num, r_den = _parse_fraction_exact(stream.get("r_frame_rate", ""))
                props["fps_num"], props["fps_den"] = r_num, r_den
                try:
                    props["start_time"] = max(0.0, float(stream.get("start_time") or 0.0))
                except (TypeError, ValueError):
                    props["start_time"] = 0.0
                # Frame index <-> timestamp arithmetic is only valid on constant
                # frame rate material. The declared rates agreeing is a cheap
                # prerequisite but NOT proof - containers report a clean
                # r_frame_rate/avg_frame_rate pair for genuinely irregular
                # material - so confirm against real packet timestamps.
                a_num, a_den = _parse_fraction_exact(stream.get("avg_frame_rate", ""))
                rates_agree = bool(
                    r_num and a_num and abs(r_num / r_den - a_num / a_den) < 1e-6
                )
                props["is_cfr"] = rates_agree and verify_cfr(video_path, r_num, r_den)
                for key in ("color_range", "color_space", "color_transfer", "color_primaries"):
                    value = stream.get(key)
                    if value and value != "unknown":
                        props[key] = value
    except (FileNotFoundError, ValueError):
        pass
    return props


def probe_frame_count(video_path: str) -> int:
    """Container-reported frame count of a video (0 when unavailable)."""
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
             "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", video_path],
            capture_output=True, text=True, creationflags=creationflags, check=False,
        )
        return int((result.stdout or "0").strip() or 0)
    except (FileNotFoundError, ValueError):
        return 0


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


# ------------------------------------------------------------------ #
# Detector registry
# ------------------------------------------------------------------ #
# Each detector declares the parameters it exposes, its default weights file and
# how to report unavailability. The GUI builds its combobox, its parameter
# widgets and its settings round-trip from this table, so adding a detector
# means adding an entry here plus a branch in _detection_and_output_task.


def _validate_threshold(value: float) -> None:
    if not (0.0 < value < 1.0):
        raise ValueError("threshold must be between 0 and 1.")


def _cast_min_scene_len(text: str) -> int:
    return max(1, int(float(text)))


def _cast_overlap(text: str) -> int:
    value = int(float(text))
    if value < 0:
        raise ValueError("overlap must be 0 or greater.")
    return value


def _validate_confidence(value: float) -> None:
    # 0.0 keeps every boundary the model proposes (upstream behaviour); 1.0 would
    # reject all of them, so it is excluded.
    if not (0.0 <= value < 1.0):
        raise ValueError("confidence must be between 0 and 1 (0 disables filtering).")


@dataclass(frozen=True)
class DetectorParam:
    """One parameter widget on the detector panel."""
    key: str
    label: str
    default: str
    tooltip: str
    values: Tuple[str, ...] = ()          # non-empty -> readonly combobox
    cast: Callable[[str], object] = str
    validate: Optional[Callable[[object], None]] = None


@dataclass(frozen=True)
class DetectorSpec:
    name: str
    weights_path: str
    params: Tuple[DetectorParam, ...]
    unavailable_reason: Callable[[], Optional[str]]


_DEVICE_PARAM = DetectorParam(
    key="device", label="device:", default="auto", tooltip="device",
    values=("auto", "cpu", "cuda", "mps"),
)
_MIN_SCENE_LEN_PARAM = DetectorParam(
    key="min_scene_len", label="min_scene_len (frames):", default="12",
    tooltip="min_scene_len", cast=_cast_min_scene_len,
)


def _threshold_param(default: str) -> DetectorParam:
    return DetectorParam(
        key="threshold", label="threshold:", default=default, tooltip="threshold",
        cast=float, validate=_validate_threshold,
    )


def _autoshot_unavailable() -> Optional[str]:
    if AutoShotNet is not None:
        return None
    detail = f"\n\nImport error: {AUTO_SHOT_IMPORT_ERROR}" if AUTO_SHOT_IMPORT_ERROR else ""
    return "AutoShot is not available. Ensure the AutoShot repo exists and install 'einops'." + detail


def _transnetv2_unavailable() -> Optional[str]:
    if TransNetV2 is not None:
        return None
    detail = f"\n\nImport error: {TRANSNET_IMPORT_ERROR}" if TRANSNET_IMPORT_ERROR else ""
    return "TransNetV2 is not available. Ensure 'transnetv2-pytorch' is installed." + detail


def _omnishotcut_unavailable() -> Optional[str]:
    if omnishotcut is not None:
        return None
    detail = f"\n\nImport error: {OMNISHOT_IMPORT_ERROR}" if OMNISHOT_IMPORT_ERROR else ""
    return (
        "OmniShotCut is not available. Ensure the OmniShotCut folder is present and "
        "'ffmpeg-python' / 'huggingface_hub' are installed." + detail
    )


DETECTORS: dict[str, DetectorSpec] = {
    "AutoShot": DetectorSpec(
        name="AutoShot",
        weights_path="./weights/ckpt_0_200_0.pth",
        params=(_DEVICE_PARAM, _threshold_param("0.24"), _MIN_SCENE_LEN_PARAM),
        unavailable_reason=_autoshot_unavailable,
    ),
    "TransNetV2": DetectorSpec(
        name="TransNetV2",
        weights_path="./weights/transnetv2-pytorch-weights.pth",
        params=(_DEVICE_PARAM, _threshold_param("0.3"), _MIN_SCENE_LEN_PARAM),
        unavailable_reason=_transnetv2_unavailable,
    ),
    # OmniShotCut has no frame-difference threshold like the other detectors: it
    # localises boundaries with a query head. Its equivalent dial is how sharply
    # that head pins down the cut's frame, exposed here as "confidence" — see
    # _omnishot_predict_windows for why that and not the no-object class.
    "OmniShotCut": DetectorSpec(
        name="OmniShotCut",
        weights_path="./weights/OmniShotCut_ckpt.pth",
        params=(
            _DEVICE_PARAM,
            DetectorParam(
                key="mode", label="mode:", default="clean_shot", tooltip="omnishot_mode",
                values=("clean_shot", "default"),
            ),
            DetectorParam(
                key="confidence", label="min confidence:", default="0.0",
                tooltip="omnishot_confidence", cast=float, validate=_validate_confidence,
            ),
            DetectorParam(
                key="overlap", label="overlap (frames):", default="20",
                tooltip="omnishot_overlap", cast=_cast_overlap,
            ),
            _MIN_SCENE_LEN_PARAM,
        ),
        unavailable_reason=_omnishotcut_unavailable,
    ),
}


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

    # Sort by start frame to be safe
    tn_scenes.sort(key=lambda x: int(x.get("start_frame", 0)) if "start_frame" in x else float(x.get("start_time", 0.0)))
    
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
        
        # Enforce continuity: 
        # If there is a gap between last_end and this start_f, the gap is likely a transition.
        # We can either:
        # 1. Extend previous scene to start_f (last_end -> start_f)
        # 2. Start this scene at last_end (start_f -> last_end)
        # 3. Split the gap.
        #
        # Simple robust approach: "Scene ends where next starts".
        # So we fix PREVIOUS scene's end to be CURRENT scene's start.
        
        if scenes_out:
            prev_start_tc, _ = scenes_out[-1]
            # Update previous scene end to meet current start
            # But ensure we don't shrink it to zero/negative
            new_prev_end = max(prev_start_tc.get_frames() + 1, start_f)
            scenes_out[-1] = (prev_start_tc, Timecode(new_prev_end, fps))
            
            # Now current scene starts exactly where previous ended
            start_f = new_prev_end
        else:
            # First scene always starts at 0
            start_f = 0
            
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


def _autoshot_frame_generator(video_path: str, width: int = 48, height: int = 27):
    import numpy as np
    
    frame_size = width * height * 3
    cmd = [
        "ffmpeg", "-v", "error", 
        "-vsync", "0",          # Force frame-exact passthrough
        "-i", video_path,
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-"
    ]
    
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=creationflags
    )
    
    try:
        while True:
            # Read a chunk of frames to reduce overhead (50 frames ~ 200KB)
            chunk_size = 50 * frame_size 
            raw = process.stdout.read(chunk_size)
            if not raw:
                break
                
            n_bytes = len(raw)
            n_frames = n_bytes // frame_size
            if n_frames == 0:
                break 
                
            frames = np.frombuffer(raw[:n_frames*frame_size], dtype=np.uint8).reshape((n_frames, height, width, 3))
            yield frames
            
            if n_bytes < chunk_size:
                break
    except Exception as e:
        logger.error(f"FFmpeg streaming error: {e}")
        raise
    finally:
        if process.stdout: process.stdout.close()
        if process.stderr: process.stderr.close()
        process.wait()




def _find_peak_cuts(predictions: list, threshold: float) -> list[int]:
    """Find local maxima (peaks) in contiguous regions above threshold."""
    import numpy as np
    
    pred = np.asarray(predictions).reshape(-1)
    # Identify regions above threshold
    is_candidate = (pred > threshold).astype(np.int32)
    
    # Pad to handle edge cases
    padded = np.pad(is_candidate, (1, 1), mode='constant', constant_values=0)
    diff = np.diff(padded)
    
    # Start and end indices of candidate regions
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    cuts = []
    for s, e in zip(starts, ends):
        # Look at the original predictions in this range [s, e)
        # We need the relative index of max value within the slice
        region = pred[s:e]
        if len(region) == 0: continue
            
        peak_idx = np.argmax(region)
        # The absolute frame index of the cut
        cut_frame = s + peak_idx
        cuts.append(cut_frame)
        
    return cuts


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
        
    # Ensure prediction array matches total_frames length
    if len(pred) < total_frames:
        pred = np.pad(pred, (0, total_frames - len(pred)), mode="constant", constant_values=0)
    pred = pred[:total_frames]

    # Find peaks to determine exact cut locations
    cut_frames = _find_peak_cuts(pred, threshold)
    
    # Construct contiguous scenes from cuts
    # Format: [0, cut1), [cut1, cut2), ... [cutN, total_frames)
    scene_boundaries = [0] + sorted(cut_frames)
    
    # Ensure the last boundary is total_frames if not already
    # If the last cut was somehow at total_frames, we don't want a 0-length scene
    if scene_boundaries[-1] >= total_frames:
        scene_boundaries[-1] = total_frames 
    else:
        scene_boundaries.append(total_frames)
        
    # Filter unique (in case 0 was a cut) and sort
    scene_boundaries = sorted(list(set(scene_boundaries)))
    
    scenes_out: list[tuple[Timecode, Timecode]] = []
    
    for i in range(len(scene_boundaries) - 1):
        start_f = scene_boundaries[i]
        end_f = scene_boundaries[i+1]
        
        # Skip empty scenes (though set/sorted should prevent this unless total_frames=0)
        if end_f <= start_f:
            continue
            
        scenes_out.append((Timecode(start_f, fps), Timecode(end_f, fps)))
        
    return scenes_out


def _autoshot_predict_from_stream(frame_gen, model, device, total_frames_est=None, abort_flag=None, progress_cb=None):
    import numpy as np
    
    buffer = [] 
    predictions = []
    
    real_frames_count = 0
    first_frame_seen = False
    
    with torch.no_grad():
        for chunk in frame_gen:
            if abort_flag and abort_flag.is_set():
                raise InterruptedError
            
            # Convert chunk (N,H,W,C) to list of frames
            # Using list of arrays is flexible
            frames_list = list(chunk)
            chunk_len = len(frames_list)
            real_frames_count += chunk_len
            
            if not first_frame_seen and chunk_len > 0:
                 # Pre-pad with 25 copies of start
                 buffer.extend([frames_list[0]] * 25)
                 first_frame_seen = True
            
            buffer.extend(frames_list)
            
            # Process complete batches (sliding window)
            while len(buffer) >= 100:
                if abort_flag and abort_flag.is_set():
                    raise InterruptedError
                    
                batch_frames = buffer[:100]
                batch_np = np.stack(batch_frames)
                
                batch_t = torch.from_numpy(
                    batch_np.transpose((3, 0, 1, 2))[np.newaxis, ...]
                ).float().to(device)
                
                one_hot = model(batch_t)
                if isinstance(one_hot, tuple): one_hot = one_hot[0]
                one_hot = torch.sigmoid(one_hot[0]).squeeze(-1)
                
                predictions.append(one_hot[25:75].detach().cpu().numpy())
                
                # Slide by 50
                del buffer[:50]
                
                if progress_cb and total_frames_est:
                     processed_so_far = len(predictions) * 50
                     progress_cb(min(0.95, processed_so_far / total_frames_est), f"AutoShot inference ({processed_so_far} frames)")

    if real_frames_count == 0:
        raise RuntimeError("AutoShot received no frames.")

    # Flush remaining buffer with padding
    if buffer:
        last_frame = buffer[-1]
        # Pad enough to flush pending frames through the center window
        buffer.extend([last_frame] * 100)
        
        while len(buffer) >= 100:
            batch_frames = buffer[:100]
            batch_np = np.stack(batch_frames)
            
            batch_t = torch.from_numpy(
                batch_np.transpose((3, 0, 1, 2))[np.newaxis, ...]
            ).float().to(device)
            
            one_hot = model(batch_t)
            if isinstance(one_hot, tuple): one_hot = one_hot[0]
            one_hot = torch.sigmoid(one_hot[0]).squeeze(-1)
            
            predictions.append(one_hot[25:75].detach().cpu().numpy())
            del buffer[:50]
            
            # Stop if we have generated enough predictions to cover real_frames_count
            if len(predictions) * 50 >= real_frames_count + 50:
                 break

    final_preds = np.concatenate(predictions, 0)
    return final_preds[:real_frames_count], real_frames_count

def _omnishot_decode_to_memmap(video_path: str, width: int, height: int, raw_path: str,
                               abort_flag=None, progress_cb=None, total_frames_est=None):
    """Decode a video to a raw rgb24 temp file and return it as a memmap (T,H,W,3).

    OmniShotCut's own _decode_video() materialises the whole video in RAM at the
    model's process resolution (~36 KB/frame, so ~4 GB for an hour). Writing to
    disk and memory-mapping keeps peak RAM flat, lets the OS page in only the
    window under inference, and gives us somewhere to check the abort flag.
    """
    import numpy as np

    frame_bytes = width * height * 3
    frames_written = 0

    with open(raw_path, "wb") as raw:
        for chunk in _autoshot_frame_generator(video_path, width=width, height=height):
            if abort_flag is not None and abort_flag.is_set():
                raise InterruptedError
            raw.write(np.ascontiguousarray(chunk).tobytes())
            frames_written += len(chunk)
            if progress_cb and total_frames_est:
                progress_cb(min(1.0, frames_written / total_frames_est),
                            f"Decoding for OmniShotCut ({frames_written} frames)")

    if frames_written == 0:
        raise RuntimeError("OmniShotCut received no frames.")

    video_np = np.memmap(raw_path, dtype=np.uint8, mode="r",
                         shape=(frames_written, height, width, 3))
    return video_np, frames_written


def _omnishot_predict_windows(video_np, model, model_args, overlap: int,
                              confidence: float = 0.0,
                              abort_flag=None, progress_cb=None):
    """Run OmniShotCut window-by-window.

    Returns (ranges, intra_labels, inter_labels, confidences).

    This mirrors omnishotcut.engine._run_on_numpy but reuses that module's own
    split_videos()/merge_predictions() so the windows and valid regions are
    identical by construction, while adding per-window abort, progress and
    confidence filtering.

    `confidence` is a minimum per-boundary localisation score in [0, 1), taken as
    the peak of the range head's softmax over frame positions.

    The intra head does have a trailing no-object column (10 outputs for 9 real
    labels), but measuring it on the shipped checkpoint shows p(no-object) is
    ~0 for every query, so it carries no signal. This model rejects a query a
    different way: idle queries emit range_idx == window_length with intra label
    "padding", which the degenerate start >= end check below already discards.
    What remains variable is how sharply the range head localises the boundary it
    did propose, and that is what we threshold on. 0.0 reproduces upstream
    behaviour exactly.
    """
    from omnishotcut.engine import video_transform, split_videos, merge_predictions

    window = int(model_args.max_process_window_length)
    if not (0 <= overlap < window):
        raise ValueError(f"overlap must be between 0 and {window - 1} (model window length).")
    if not (0.0 <= confidence < 1.0):
        raise ValueError("confidence must be between 0 and 1 (0 disables filtering).")

    device = next(model.parameters()).device
    windows = split_videos(video_np, window, overlap)
    pred_boundary_full: list = []
    seen_confidences: list[float] = []
    num_rejected = 0

    for idx, (chunk, _num_pad, window_start, valid_start, valid_end, valid_len) in enumerate(windows):
        if abort_flag is not None and abort_flag.is_set():
            raise InterruptedError

        # np.asarray materialises just this window from the memmap.
        import numpy as np
        video_tensor = video_transform(np.asarray(chunk)).unsqueeze(0).to(device)

        with torch.inference_mode():
            outputs = model(video_tensor)

        query_intra_idx = outputs["intra_clip_logits"].softmax(-1)[0, :, :-1].argmax(dim=-1)
        query_inter_idx = outputs["inter_clip_logits"].softmax(-1)[0, :, :-1].argmax(dim=-1)

        # Peak of the range head doubles as the boundary's confidence: a flat
        # distribution means the model is unsure where the cut actually falls.
        range_probs = outputs["pred_shot_logits"].softmax(-1)[0, :, :-1]
        range_peak = range_probs.max(dim=-1)
        query_range_idx = range_peak.indices
        query_conf = range_peak.values.detach().cpu()

        pred_boundary = []
        start_local = 0
        for keep_idx in range(len(query_intra_idx)):
            end_local = min(int(query_range_idx[keep_idx].detach().cpu()), valid_len)
            if start_local >= end_local:
                continue

            end_global = window_start + end_local
            if valid_start < end_global <= valid_end:
                conf = float(query_conf[keep_idx])
                seen_confidences.append(conf)
                if conf >= confidence:
                    pred_boundary.append({
                        "end_frame_idx": int(end_global),
                        "intra_label": int(query_intra_idx[keep_idx].detach().cpu()),
                        "inter_label": int(query_inter_idx[keep_idx].detach().cpu()),
                        "confidence": conf,
                    })
                else:
                    # Dropping the boundary merges this segment into the next one,
                    # because ranges are rebuilt contiguously from what survives.
                    num_rejected += 1

            # Advance regardless: start_local chains the queries' local ranges, so
            # it must track the model's segmentation even where we reject a boundary.
            start_local = end_local
            if end_local >= valid_len:
                break

        pred_boundary_full = merge_predictions(pred_boundary_full, pred_boundary)

        if progress_cb:
            progress_cb((idx + 1) / len(windows),
                        f"OmniShotCut inference (window {idx + 1}/{len(windows)})")

    # Boundaries -> contiguous ranges, identical to upstream's final pass.
    ranges, intra_labels, inter_labels, confidences = [], [], [], []
    start_frame_idx = 0
    for item in pred_boundary_full:
        end_frame_idx = min(int(item["end_frame_idx"]), len(video_np))
        if end_frame_idx <= start_frame_idx:
            continue
        ranges.append([int(start_frame_idx), int(end_frame_idx)])
        intra_labels.append(int(item["intra_label"]))
        inter_labels.append(int(item["inter_label"]))
        confidences.append(float(item.get("confidence", 1.0)))
        start_frame_idx = end_frame_idx

    # Log the distribution so the threshold can be calibrated against a real run
    # instead of guessed at.
    if seen_confidences:
        ordered = sorted(seen_confidences)
        quantiles = {
            f"p{q}": round(ordered[min(len(ordered) - 1, int(len(ordered) * q / 100))], 4)
            for q in (5, 25, 50, 75, 95)
        }
        logger.info(
            "OmniShotCut confidence over %d proposed boundaries: %s (rejected %d at confidence>=%.3f)",
            len(seen_confidences), quantiles, num_rejected, confidence,
        )

    return ranges, intra_labels, inter_labels, confidences


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
        self.loaded_models: dict = {}

        # --- Input & Output Variables ---
        self.video_path_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()

        # --- Detector & AI Variables ---
        self.detector_type_var = tk.StringVar(value="TransNetV2")
        self.params_vars: dict = {}
        self.detector_params_cache: dict[str, dict[str, str]] = {
            name: self._default_detector_params(name) for name in DETECTORS
        }
        self._last_detector_type = self.detector_type_var.get()
        self.snap_cuts_var = tk.BooleanVar(value=True)
        self.ai_validate_var = tk.BooleanVar(value=False)
        self.ai_window_var = tk.IntVar(value=5)
        self.flash_sensitivity_var = tk.IntVar(value=15)  # Luma delta threshold for flash detection

        # --- Output Action Variables ---
        self.export_csv_var = tk.BooleanVar(value=False)
        self.export_html_var = tk.BooleanVar(value=False)
        self.export_sc_var = tk.BooleanVar(value=False)
        self.save_images_var = tk.BooleanVar(value=False)
        self.split_video_var = tk.BooleanVar(value=False)
        self.num_images_var = tk.IntVar(value=3)
        self.frame_margin_var = tk.IntVar(value=1)
        self.sc_offset_var = tk.IntVar(value=0)

        # --- FFmpeg Variables ---
        # Output codec is fixed: 10-bit HEVC on the NVIDIA GPU (CPU libx265 fallback).
        self.ffmpeg_codec_var = tk.StringVar(value="hevc_nvenc")
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
            values=list(DETECTORS),
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
        codec_label = ttk.Label(ffmpeg_frame, text="HEVC 10-bit (NVENC)")
        codec_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self._attach_tooltip(codec_label, "ffmpeg_codec")
        
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

        # --- AI & Refinements Frame ---
        ai_frame = ttk.LabelFrame(main_frame, text="Optional: Refinements & Validation")
        ai_frame.pack(fill=tk.X, padx=5, pady=5)
        
        snap_check = ttk.Checkbutton(ai_frame, text="Snap cuts to frame-perfect boundary (Fixes +/- 2 frame AI inaccuracies)", variable=self.snap_cuts_var)
        snap_check.grid(row=0, column=0, columnspan=4, padx=5, pady=5, sticky="w")
        self._attach_tooltip(snap_check, "snap_cuts")

        ai_check = ttk.Checkbutton(ai_frame, text="Validate cuts with DINOv3 AI:", variable=self.ai_validate_var)
        ai_check.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        self._attach_tooltip(ai_check, "ai_validate")

        ttk.Label(ai_frame, text="Validation Window (frames):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        ai_spin = ttk.Spinbox(ai_frame, from_=2, to=10, textvariable=self.ai_window_var, width=5)
        ai_spin.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self._attach_tooltip(ai_spin, "ai_window")

        ttk.Label(ai_frame, text="Flash Sensitivity:").grid(row=2, column=2, padx=(20, 5), pady=5, sticky="w")
        flash_spin = ttk.Spinbox(ai_frame, from_=10, to=80, textvariable=self.flash_sensitivity_var, width=5)
        flash_spin.grid(row=2, column=3, padx=5, pady=5, sticky="w")
        self._attach_tooltip(flash_spin, "flash_sensitivity")


        # --- Output Actions Frame ---
        output_frame = ttk.LabelFrame(main_frame, text="3. Output Actions")
        output_frame.pack(fill=tk.X, padx=5, pady=5, ipady=5)
        # Checkboxes
        c1 = ttk.Checkbutton(output_frame, text="Export scene list to CSV", variable=self.export_csv_var); c1.grid(row=0, column=0, sticky="w", padx=5, pady=2)
        c2 = ttk.Checkbutton(output_frame, text="Export scene list to HTML", variable=self.export_html_var); c2.grid(row=1, column=0, sticky="w", padx=5, pady=2)
        c3 = ttk.Checkbutton(output_frame, text="Export to .sc file", variable=self.export_sc_var); c3.grid(row=2, column=0, sticky="w", padx=5, pady=2)
        
        ttk.Label(output_frame, text="SC Offset:").grid(row=2, column=1, sticky="w", padx=(20, 5), pady=2)
        sc_off_spin = ttk.Spinbox(output_frame, from_=-5, to=5, textvariable=self.sc_offset_var, width=5)
        sc_off_spin.grid(row=2, column=2, sticky="w", padx=5, pady=2)
        self._attach_tooltip(sc_off_spin, "sc_offset")

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

    def _get_or_load_model(self, detector_type: str, weights_path: str, device: str):
        key = (detector_type, weights_path, device)
        if key in self.loaded_models:
            return self.loaded_models[key]

        logger.info("Loading %s model (Device: %s)...", detector_type, device)
        if detector_type == "AutoShot":
            # Returns (model, device)
            result = _autoshot_load_model(weights_path, device)
            self.loaded_models[key] = result
            return result
        elif detector_type == "TransNetV2":
            if TransNetV2 is None:
                raise RuntimeError(f"TransNetV2 is unavailable: {TRANSNET_IMPORT_ERROR}")
            if not weights_path or not os.path.exists(weights_path):
                raise FileNotFoundError(f"TransNetV2 weights file not found: {weights_path}")

            resolved_device = _autoshot_select_device(device)
            model = TransNetV2(device=resolved_device)
            model.eval()

            state_dict = torch.load(weights_path, map_location=model.device)
            model.load_state_dict(state_dict)
            
            self.loaded_models[key] = model
            return model
        elif detector_type == "OmniShotCut":
            if omnishotcut is None:
                raise RuntimeError(f"OmniShotCut is unavailable: {OMNISHOT_IMPORT_ERROR}")
            if not weights_path or not os.path.exists(weights_path):
                raise FileNotFoundError(
                    f"OmniShotCut weights file not found: {weights_path}\n\n"
                    "Download 'OmniShotCut_ckpt.pth' from https://huggingface.co/uva-cv-lab/OmniShotCut "
                    "into the weights/ directory."
                )

            resolved_device = _autoshot_select_device(device)
            model = omnishotcut.load(weights_path, device=resolved_device)

            self.loaded_models[key] = model
            return model
        else:
            raise ValueError(f"Unknown detector: {detector_type}")

    def _on_closing(self):
        """Handle the window close event."""
        logger.info("Closing application and saving settings.")
        self._save_settings()
        self.root.destroy()

    def _default_detector_params(self, detector_type: str) -> dict[str, str]:
        spec = DETECTORS.get(detector_type)
        if spec is None:
            return {}
        return {p.key: p.default for p in spec.params}

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
            "snap_cuts": self.snap_cuts_var.get(),
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
            "sc_offset": self.sc_offset_var.get(),
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
            
            self.snap_cuts_var.set(settings.get("snap_cuts", True))
            self.ai_validate_var.set(settings.get("ai_validate", False))
            self.ai_window_var.set(settings.get("ai_window", 3))
            self.flash_sensitivity_var.set(settings.get("flash_sensitivity", 15))
            self.export_csv_var.set(settings.get("export_csv", False))
            self.export_html_var.set(settings.get("export_html", False))
            self.export_sc_var.set(settings.get("export_sc", False))
            self.save_images_var.set(settings.get("save_images", False))
            self.split_video_var.set(settings.get("split_video", False))
            self.num_images_var.set(settings.get("num_images", 3))
            self.frame_margin_var.set(settings.get("frame_margin", 1))
            self.sc_offset_var.set(settings.get("sc_offset", 0))
            # Codec is no longer user-selectable; ignore any legacy saved value.
            self.ffmpeg_codec_var.set("hevc_nvenc")
            self.ffmpeg_preset_var.set(settings.get("ffmpeg_preset", "p7"))
            self.ffmpeg_cq_var.set(settings.get("ffmpeg_cq", 16))
            
            # Detector type & per-detector params
            detector_type = settings.get("detector_type", "AutoShot")
            if detector_type not in DETECTORS:
                detector_type = "AutoShot"

            loaded_params = settings.get("detector_params")
            if isinstance(loaded_params, dict):
                if any(k in loaded_params for k in DETECTORS):
                    for det in DETECTORS:
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
        """Build detector parameter widgets from the selected detector's schema."""
        for widget in self.params_frame.winfo_children():
            widget.destroy()

        # Clear and rebuild vars
        self.params_vars.clear()

        detector_type = self.detector_type_var.get()
        spec = DETECTORS.get(detector_type)
        if spec is None:
            self._last_detector_type = detector_type
            return

        params = self.detector_params_cache.get(detector_type, self._default_detector_params(detector_type))

        for row, param in enumerate(spec.params):
            ttk.Label(self.params_frame, text=param.label).grid(row=row, column=0, padx=5, pady=2, sticky="w")
            var = tk.StringVar(value=str(params.get(param.key, param.default)))
            if param.values:
                widget = ttk.Combobox(
                    self.params_frame, textvariable=var,
                    values=list(param.values), state="readonly", width=10,
                )
            else:
                widget = ttk.Entry(self.params_frame, textvariable=var, width=10)
            widget.grid(row=row, column=1, padx=5, pady=2, sticky="w")
            self._attach_tooltip(widget, param.tooltip)
            self.params_vars[param.key] = var

        self._last_detector_type = detector_type


    def _start_detection(self) -> None:
        video_path = self.video_path_var.get()
        output_dir = self.output_dir_var.get()
        if not video_path or not output_dir:
            messagebox.showerror("Error", "Please select a video file and an output folder.")
            return
        detector_type = self.detector_type_var.get()
        spec = DETECTORS.get(detector_type)
        if spec is None:
            messagebox.showerror("Error", f"Unknown detector type: {detector_type}")
            return

        if reason := spec.unavailable_reason():
            messagebox.showerror("Error", reason)
            return

        detector_cfg: dict = {
            "type": detector_type,
            "weights_path": spec.weights_path,
        }
        try:
            for param in spec.params:
                var = self.params_vars.get(param.key)
                raw = str(var.get()).strip() if var is not None else param.default
                if not raw:
                    raw = param.default
                value = param.cast(raw)
                if param.validate is not None:
                    param.validate(value)
                detector_cfg[param.key] = value
        except Exception as exc:
            logger.error("Invalid %s parameters: %s", detector_type, exc)
            messagebox.showerror("Parameter Error", f"Invalid {detector_type} parameters: {exc}")
            return

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
                self.progress_queue.put((10, "Checking AutoShot model..."))

                weights_path = str(detector.get("weights_path", "")).strip()
                if not weights_path:
                    raise FileNotFoundError("AutoShot weights_path is empty. Please select a .pth weights file.")

                model, device = self._get_or_load_model("AutoShot", weights_path, detector.get("device", "auto"))

                if self.abort_flag.is_set():
                    raise InterruptedError

                if self.abort_flag.is_set():
                    raise InterruptedError

                self.progress_queue.put((20, "Running AutoShot inference (streaming)..."))
                
                frames_gen = _autoshot_frame_generator(video_path)
                
                def progress_cb(pct: float, msg: str):
                    self.progress_queue.put((20 + float(pct) * 40.0, msg))

                predictions, frame_count = _autoshot_predict_from_stream(
                    frames_gen,
                    model,
                    device,
                    total_frames_est=total_frames if total_frames > 0 else None,
                    abort_flag=self.abort_flag,
                    progress_cb=progress_cb,
                )

                if total_frames <= 0:
                    total_frames = frame_count
                    self.total_frames = total_frames
                elif abs(total_frames - frame_count) > 1:
                    logger.warning(
                        "Frame count mismatch: ffprobe=%d, decoded=%d. Using ffprobe count for outputs.",
                        total_frames,
                        frame_count,
                    )

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
                self.progress_queue.put((10, "Checking TransNetV2 model..."))
                
                weights_path = str(detector.get("weights_path", "")).strip()
                if not weights_path:
                    raise FileNotFoundError("TransNetV2 weights_path is empty. Please select a .pth weights file.")
                    
                model = self._get_or_load_model("TransNetV2", weights_path, detector.get("device", "auto"))

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

            elif detector_type == "OmniShotCut":
                self.progress_queue.put((10, "Checking OmniShotCut model..."))

                weights_path = str(detector.get("weights_path", "")).strip()
                if not weights_path:
                    raise FileNotFoundError("OmniShotCut weights_path is empty.")

                model = self._get_or_load_model("OmniShotCut", weights_path, detector.get("device", "auto"))

                if self.abort_flag.is_set():
                    raise InterruptedError

                model_args = model._model_args
                proc_w = int(model_args.process_width)
                proc_h = int(model_args.process_height)
                mode = str(detector.get("mode", "clean_shot"))
                overlap = int(detector.get("overlap", 20))
                confidence = float(detector.get("confidence", 0.0) or 0.0)

                # Decode to a raw temp file and memory-map it, rather than using
                # model.inference(), which buffers the whole video in RAM and
                # offers no progress or abort hook.
                import tempfile
                from collections import Counter
                from omnishotcut.label_correspondence import (
                    unique_intra_label_mapping, intra_int2string, inter_int2string,
                )

                raw_fd, raw_path = tempfile.mkstemp(suffix=".rgb24", prefix="omnishotcut_")
                os.close(raw_fd)
                video_np = None
                try:
                    self.progress_queue.put((20, "Decoding video for OmniShotCut..."))
                    video_np, decoded_frames = _omnishot_decode_to_memmap(
                        video_path, proc_w, proc_h, raw_path,
                        abort_flag=self.abort_flag,
                        progress_cb=lambda pct, msg: self.progress_queue.put((20 + pct * 15.0, msg)),
                        total_frames_est=total_frames if total_frames > 0 else None,
                    )

                    if total_frames <= 0:
                        total_frames = decoded_frames
                        self.total_frames = total_frames
                    elif abs(total_frames - decoded_frames) > 1:
                        logger.warning(
                            "Frame count mismatch: ffprobe=%d, decoded=%d. Using ffprobe count for outputs.",
                            total_frames, decoded_frames,
                        )

                    self.progress_queue.put((35, f"Running OmniShotCut inference (mode={mode})..."))
                    ranges, intra_ids, inter_ids, confs = _omnishot_predict_windows(
                        video_np, model._model, model_args, overlap,
                        confidence=confidence,
                        abort_flag=self.abort_flag,
                        progress_cb=lambda pct, msg: self.progress_queue.put((35 + pct * 25.0, msg)),
                    )
                finally:
                    # Windows refuses to delete a file while it is still mapped.
                    if video_np is not None and hasattr(video_np, "_mmap"):
                        video_np._mmap.close()
                    video_np = None
                    try:
                        os.remove(raw_path)
                    except OSError as exc:
                        logger.warning("Could not remove OmniShotCut temp file %s: %s", raw_path, exc)

                if mode == "clean_shot":
                    # Keep only general (hard) cuts; dropped transitions leave gaps
                    # that the conversion below closes.
                    general_id = unique_intra_label_mapping["general"]
                    ranges = [r for r, lbl in zip(ranges, intra_ids) if lbl == general_id]
                else:
                    logger.info(
                        "OmniShotCut transition labels: %s",
                        dict(Counter(intra_int2string.get(x, str(x)) for x in intra_ids)),
                    )
                    logger.debug(
                        "OmniShotCut inter labels: %s",
                        dict(Counter(inter_int2string.get(x, str(x)) for x in inter_ids)),
                    )

                if confs:
                    logger.debug(
                        "OmniShotCut kept %d boundaries, confidence min=%.4f mean=%.4f",
                        len(confs), min(confs), sum(confs) / len(confs),
                    )

                if self.abort_flag.is_set():
                    raise InterruptedError

                self.progress_queue.put((60, f"OmniShotCut returned {len(ranges)} ranges. Converting..."))

                # Ranges are [start, end) with end exclusive; in "default" mode they
                # already tile the video, while "clean_shot" leaves gaps where the
                # dropped transitions were. transnetv2_scenes_to_timecodes closes
                # those gaps by extending the preceding scene.
                osc_scenes = [
                    {"start_frame": int(start), "end_frame": int(end)}
                    for start, end in ranges
                ]
                scenes = transnetv2_scenes_to_timecodes(osc_scenes, fps=fps, total_frames=total_frames)

            else:
                raise RuntimeError(f"Unknown detector type: {detector_type}")

            # Enforce minimum scene length (frames) by merging short scenes
            # into neighbours (not dropping them, which would create timeline gaps).
            min_len = int(detector.get("min_scene_len", 1) or 1)
            guard_len = short_scene_guard_frames(fps)
            if min_len > 1 and scenes and len(scenes) > 1:
                frames = [[int(st.get_frames()), int(et.get_frames())] for st, et in scenes]
                i = 0
                while i < len(frames) and len(frames) > 1:
                    seg_len = frames[i][1] - frames[i][0]
                    if seg_len >= min_len:
                        i += 1
                        continue
                    # Merge into a neighbour
                    if i == 0:
                        frames[1][0] = frames[0][0]
                        del frames[0]
                        continue  # re-check index 0
                    if i == len(frames) - 1:
                        frames[i - 1][1] = frames[i][1]
                        del frames[i]
                        i = max(0, i - 1)
                        continue
                    # Middle scene: merge into the shorter neighbour
                    prev_len = frames[i - 1][1] - frames[i - 1][0]
                    next_len = frames[i + 1][1] - frames[i + 1][0]
                    if prev_len >= next_len:
                        frames[i - 1][1] = frames[i][1]
                        del frames[i]
                        i = max(0, i - 1)
                    else:
                        frames[i + 1][0] = frames[i][0]
                        del frames[i]
                    continue
                scenes = [(Timecode(s, fps), Timecode(e, fps)) for s, e in frames]

            logger.info("Detected %d raw scenes (%s)", len(scenes), detector_type)

            # --- 3. Optional Refinements ---
            if self.snap_cuts_var.get() and scenes:
                self.progress_queue.put((65, "Snapping cuts to exact frames..."))
                scenes = self._snap_cuts_to_exact_pixel_diff(video_path, scenes, search_radius=2)
                
            if self.ai_validate_var.get() and scenes:
                self.progress_queue.put((70, "AI validating cuts..."))
                scenes = self._run_ai_validation(video_path, scenes, guard_len)

            # --- 4. Ultra-short Scene Merge (Always On) ---
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
                (self.export_sc_var, "SC", lambda: self._export_sc(fps, total_frames, Path(output_dir) / f"{base_name}.sc", self.sc_offset_var.get())),
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

    def _snap_cuts_to_exact_pixel_diff(self, video_path: str, scenes: list[tuple[Timecode, Timecode]], search_radius: int = 2) -> list[tuple[Timecode, Timecode]]:
        """Refines cut boundaries by finding the exact frame with maximum visual disruption within a small local search window."""
        import cv2
        import numpy as np

        if len(scenes) < 2:
            return scenes

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning("Could not open video to snap cuts. Returning original scenes.")
            return scenes

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        
        # We only need to adjust the boundaries between scenes
        cut_frames = []
        for _, end_tc in scenes[:-1]:
            cut_frames.append(int(end_tc.get_frames()))
            
        snapped_cuts = []
            
        for i, cut in enumerate(cut_frames):
            if self.abort_flag.is_set():
                break

            # Search window: [cut - radius - 1, cut + radius]
            start_search = max(0, cut - search_radius - 1)
            end_search = cut + search_radius
            if total_frames > 0:
                end_search = min(total_frames - 1, end_search)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_search)
            frames_cache = {}
            for f in range(start_search, end_search + 1):
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Resize very small for ultra-fast, robust mean difference
                    frames_cache[f] = cv2.resize(frame, (64, 36))
                
            max_diff = -1.0
            best_cut = cut
            
            for f in range(max(1, cut - search_radius), end_search + 1):
                prev_f = frames_cache.get(f - 1)
                curr_f = frames_cache.get(f)
                if prev_f is not None and curr_f is not None:
                    diff = np.mean(np.abs(curr_f.astype(np.float32) - prev_f.astype(np.float32)))
                    if diff > max_diff:
                        max_diff = float(diff)
                        best_cut = f
                        
            snapped_cuts.append(best_cut)
            
            if self.progress_queue and i % 20 == 0:
                self.progress_queue.put((65 + (i/len(cut_frames))*4.0, f"Snapping cuts to perfect frame ({i+1}/{len(cut_frames)})"))

        cap.release()
        
        if not snapped_cuts:
            return scenes
            
        fps = scenes[0][0].fps
        snapped_scenes = []
        cur_start = scenes[0][0].get_frames()
        
        for new_cut in snapped_cuts:
            if new_cut <= cur_start:
                new_cut = cur_start + 1 
            snapped_scenes.append((Timecode(cur_start, fps), Timecode(new_cut, fps)))
            cur_start = new_cut
            
        final_end = scenes[-1][1].get_frames()
        if final_end <= cur_start:
            final_end = cur_start + 1
            if total_frames > 0 and final_end > total_frames:
                final_end = total_frames
        snapped_scenes.append((Timecode(cur_start, fps), Timecode(final_end, fps)))
        
        logger.info("Pixel snapping adjusted %d boundaries to their exact maximum visual difference.", len(snapped_cuts))
        return snapped_scenes

    def _run_ai_validation(self, video_path, scenes, short_guard_frames: int):
        """Filter detector output through the AI cut validator.

        The validator itself lives in cut_validator.py; it is imported lazily so
        that a missing torch/transformers/OpenCV only disables validation rather
        than preventing the GUI from starting.
        """
        try:
            from cut_validator import CutValidator, ValidatorConfig

            model_dir = "./weights/DINOv3"
            cache_key = ("CutValidator", model_dir)
            validator = self.loaded_models.get(cache_key)
            if validator is None:
                logger.info("Loading DINOv3 cut validator from %s...", model_dir)
                validator = CutValidator(model_dir=model_dir, log=logger)
                self.loaded_models[cache_key] = validator
            else:
                validator.reset_caches()

            validator.cfg = ValidatorConfig(flash_luma_delta=float(self.flash_sensitivity_var.get()))

            def progress_cb(pct: float, msg: str):
                # Map into the app's overall progress bar range.
                self.progress_queue.put_nowait((75 + float(pct) * 4.0, msg))

            validated_scenes = validator.filter_scenes(
                video_path=video_path,
                scenes=scenes,
                window=int(self.ai_window_var.get()),
                short_guard_frames=short_guard_frames,
                total_frames=int(self.total_frames or 0),
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

    def _export_sc(self, fps: float, total_frames: int, filename: Path, offset: int = 0):
        """Export a .sc file (same frame-value format as before)."""
        if not self.detected_scenes:
            return

        total_frames = int(total_frames)
        fps = float(fps) if fps else 0.0

        frame_values = [1] * max(0, total_frames)
        last_cut_frame = 0

        # Mark cut frames (skip first scene start at 0)
        for start_tc, _ in self.detected_scenes[1:]:
            cut_frame = int(start_tc.get_frames()) + offset
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
        """Split the input video into per-scene clips, frame-exact, as 10-bit HEVC.

        Video is always re-encoded to 10-bit HEVC (Main10) on the NVIDIA GPU via
        hevc_nvenc, falling back to libx265 10-bit only when no NVENC build is
        available. Boundaries are cut with the ``trim`` filter on decoded frame
        indices - never on keyframes or wall-clock seeks - and timestamps are
        rebuilt as strict CFR so every clip holds exactly the frames the
        detector assigned to it.
        """
        import concurrent.futures

        if not self.detected_scenes:
            logger.warning("No scenes available to split.")
            return

        total_scenes = len(self.detected_scenes)
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

        # --- Capture Settings (Main Thread) ---
        preset = self.ffmpeg_preset_var.get()
        cq_val = str(self.ffmpeg_cq_var.get())

        props = probe_stream_props(video_path)
        input_has_audio = props.get("has_audio", False)

        # Exact rational frame rate keeps 23.976/29.97 sources from drifting.
        fps_num, fps_den = props.get("fps_num", 0), props.get("fps_den", 0)
        have_exact_rate = bool(fps_num and fps_den)
        if not have_exact_rate:
            fps_num, fps_den = int(round(float(fps) * 1000)), 1000
        rate_arg = f"{fps_num}/{fps_den}"
        start_time = props.get("start_time", 0.0)

        # Keyframe pre-seek turns each clip into a one-GOP decode instead of a
        # full decode from frame 0 - the difference between minutes and days on
        # sources with thousands of scenes. It needs a trustworthy frame
        # index <-> timestamp mapping, so it is restricted to CFR sources.
        use_seek = have_exact_rate and props.get("is_cfr", False)
        if not use_seek:
            logger.warning(
                "Source is variable frame rate or has no reliable frame rate - using "
                "full-decode frame indexing. This is exact but slow on long sources."
            )

        use_nvenc = has_nvenc_hevc()
        if not use_nvenc:
            logger.warning(
                "hevc_nvenc is unavailable in this FFmpeg build - falling back to "
                "libx265 10-bit on the CPU (much slower)."
            )

        # Decode on NVDEC and keep frames in GPU memory all the way to NVENC, so
        # the CPU never touches pixel data. Without this the GPU only does the
        # encode while the CPU still decodes every frame and does the 8->10-bit
        # conversion, which pins cores when several clips run in parallel.
        use_hwdec = use_nvenc and can_hwdecode(video_path)
        hwdec_lock = threading.Lock()
        if use_nvenc and not use_hwdec:
            logger.warning(
                "This source cannot be decoded by NVDEC (unsupported codec or chroma "
                "format) - decoding on the CPU. Encoding still runs on the GPU."
            )
        logger.info(
            "Splitting video into %d scenes (frame-exact, 10-bit HEVC via %s, %s, %s decode) "
            "at %s fps...",
            total_scenes, "NVENC" if use_nvenc else "libx265",
            "keyframe pre-seek" if use_seek else "full decode",
            "NVDEC" if use_hwdec else "CPU", rate_arg,
        )

        def video_encoder_args(hwdec: bool) -> list:
            if use_nvenc:
                args = [
                    "-c:v", "hevc_nvenc",
                    "-preset", preset,
                    "-tune", "hq",
                    "-profile:v", "main10",
                    "-rc", "constqp", "-qp", cq_val, "-b:v", "0",
                    "-spatial-aq", "1", "-temporal-aq", "1",
                ]
                # With NVDEC the frames are already CUDA p010 surfaces; naming a
                # software pix_fmt here would force a needless GPU->CPU->GPU trip.
                if not hwdec:
                    args += ["-pix_fmt", "p010le"]
                return args
            return [
                "-c:v", "libx265",
                "-preset", "medium",
                "-profile:v", "main10",
                "-pix_fmt", "yuv420p10le",
                "-crf", cq_val,
            ]

        def color_args() -> list:
            """Carry source colour tags forward so the 10-bit clips match the source."""
            args = []
            for key, flag in (
                ("color_range", "-color_range"),
                ("color_space", "-colorspace"),
                ("color_transfer", "-color_trc"),
                ("color_primaries", "-color_primaries"),
            ):
                if props.get(key):
                    args += [flag, props[key]]
            return args

        def build_command(start_frame: int, end_frame: int, with_audio: bool,
                          out_path: Path, seek: bool, hwdec: bool) -> list:
            # Two cutting strategies, both frame-exact:
            #
            # seek=True (CFR sources): fast keyframe seek to just before the
            #   scene, then select frames by their ORIGINAL timestamps under
            #   -copyts. Decoding only touches one GOP instead of the whole file
            #   from frame 0, which is what makes thousands of scenes tractable.
            #   Input seeking always lands on a keyframe at or before the target,
            #   so trimming on absolute PTS yields exactly the same frames as a
            #   full decode no matter where the seek lands.
            #
            # seek=False (VFR/unknown rate): decode from frame 0 and trim on
            #   decoded frame indices, the only correct option when frame numbers
            #   cannot be mapped to timestamps.
            command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
            if hwdec:
                command += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]

            if seek:
                start_pts = start_time + start_frame * fps_den / fps_num
                end_pts = start_time + end_frame * fps_den / fps_num
                # Nudge bounds by half a frame so floating-point rounding can
                # never straddle a frame's exact presentation time.
                half = 0.5 * fps_den / fps_num
                lo, hi = start_pts - half, end_pts - half
                seek_to = max(0.0, start_pts - SEEK_MARGIN_SEC)
                command += ["-copyts", "-ss", f"{seek_to:.9f}", "-i", video_path]
                v_trim = f"trim=start={lo:.9f}:end={hi:.9f}"
                a_trim = f"atrim=start={lo:.9f}:end={hi:.9f}"
            else:
                # trim's end_frame is exclusive, matching PySceneDetect's
                # exclusive end timecode: the clip holds [start_frame, end_frame).
                command += ["-i", video_path]
                v_trim = f"trim=start_frame={start_frame}:end_frame={end_frame}"
                a_trim = (f"atrim=start={start_frame * fps_den / fps_num:.9f}"
                          f":end={end_frame * fps_den / fps_num:.9f}")

            # setpts=N/FRAME_RATE/TB regenerates a perfectly uniform CFR ramp
            # from the trimmed frame index, so neither the seek offset nor any
            # source timestamp jitter can survive into the clip. trim and setpts
            # only pass or drop frames and rewrite timestamps - they never read
            # pixels - so both work unchanged on CUDA surfaces.
            # The 8->10-bit conversion is the one step that touches pixels, so it
            # runs as scale_cuda on the GPU when hardware decoding is active.
            to_10bit = "scale_cuda=format=p010le" if hwdec else "format=p010le"
            video_chain = f"[0:v]{v_trim},setpts=N/FRAME_RATE/TB,{to_10bit}[v]"
            if with_audio:
                command += ["-filter_complex",
                            f"{video_chain};[0:a]{a_trim},asetpts=N/SR/TB[a]",
                            "-map", "[v]", "-map", "[a]"]
            else:
                command += ["-filter_complex", video_chain, "-map", "[v]", "-an"]

            command += video_encoder_args(hwdec)
            command += color_args()
            # Force CFR at the source rate: no frame may be dropped or duplicated.
            command += ["-fps_mode", "cfr", "-r", rate_arg, "-video_track_timescale", str(fps_num)]
            if with_audio:
                command += ["-c:a", "aac", "-b:a", "192k"]
            # hvc1 tag keeps HEVC playable in QuickTime/Apple/Adobe tooling.
            command += ["-tag:v", "hvc1", "-movflags", "+faststart",
                        "-max_muxing_queue_size", "1024", str(out_path)]
            return command

        def _split_worker(packed_args):
            idx, start_tc, end_tc = packed_args
            if self.abort_flag.is_set():
                return

            scene_num = idx + 1
            output_filename = Path(output_dir) / f"{Path(video_path).stem}-Scene-{scene_num:03d}.mp4"

            start_frame = int(start_tc.get_frames())
            end_frame = int(end_tc.get_frames())
            expected_frames = end_frame - start_frame

            with_audio = input_has_audio
            nonlocal use_hwdec
            hwdec = use_hwdec
            result = subprocess.run(
                build_command(start_frame, end_frame, with_audio, output_filename, use_seek, hwdec),
                capture_output=True, text=True, creationflags=creationflags, check=False,
            )

            # NVDEC can also fail part-way through a file (surface exhaustion, a
            # stream switching to an unsupported format). Drop to CPU decoding
            # once for the whole run rather than paying the failure per scene.
            if result.returncode != 0 and hwdec:
                logger.warning(
                    "Scene %d: hardware decode failed, falling back to CPU decode "
                    "for the rest of this split. FFmpeg said: %s",
                    scene_num, (result.stderr or "").strip().splitlines()[-1:] or "",
                )
                with hwdec_lock:
                    use_hwdec = False
                hwdec = False
                result = subprocess.run(
                    build_command(start_frame, end_frame, with_audio, output_filename, use_seek, False),
                    capture_output=True, text=True, creationflags=creationflags, check=False,
                )

            if result.returncode != 0 and with_audio:
                stderr_lower = (result.stderr or "").lower()
                if "matches no streams" in stderr_lower or "stream specifier" in stderr_lower:
                    logger.warning("Scene %d: audio stream issues, retrying video-only.", scene_num)
                    with_audio = False
                    result = subprocess.run(
                        build_command(start_frame, end_frame, False, output_filename, use_seek, hwdec),
                        capture_output=True, text=True, creationflags=creationflags, check=False,
                    )

            if result.returncode != 0:
                logger.error("FFmpeg failed on scene %d: %s", scene_num, result.stderr)
                raise RuntimeError(f"FFmpeg failed: {result.returncode}")

            # Verify the cut really is frame-exact rather than assuming it. This
            # is also exactly how a bad seek would show up, so a mismatch in seek
            # mode triggers a rebuild via the full-decode path.
            actual_frames = probe_frame_count(str(output_filename))
            if actual_frames and actual_frames != expected_frames and use_seek:
                logger.warning(
                    "Scene %d: seek-mode cut produced %d frames, expected %d - "
                    "rebuilding with full-decode frame indexing.",
                    scene_num, actual_frames, expected_frames,
                )
                result = subprocess.run(
                    build_command(start_frame, end_frame, with_audio, output_filename, False, hwdec),
                    capture_output=True, text=True, creationflags=creationflags, check=False,
                )
                if result.returncode != 0:
                    logger.error("FFmpeg rebuild failed on scene %d: %s", scene_num, result.stderr)
                    raise RuntimeError(f"FFmpeg failed: {result.returncode}")
                actual_frames = probe_frame_count(str(output_filename))

            if actual_frames and actual_frames != expected_frames:
                logger.warning(
                    "Scene %d frame count mismatch: expected %d, got %d.",
                    scene_num, expected_frames, actual_frames,
                )
            return actual_frames, expected_frames

        # NVENC allows only a handful of concurrent encode sessions on consumer
        # GPUs, so keep the pool narrow when encoding on the GPU.
        max_workers = 3 if use_nvenc else min(4, os.cpu_count() or 4)
        tasks = []
        completed_count = 0
        exact_count = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i, (start_tc, end_tc) in enumerate(self.detected_scenes):
                tasks.append(executor.submit(_split_worker, (i, start_tc, end_tc)))

            for future in concurrent.futures.as_completed(tasks):
                if self.abort_flag.is_set():
                    executor.shutdown(wait=False, cancel_futures=True)
                    logger.warning("Parallel split aborted.")
                    return

                try:
                    counts = future.result()
                    completed_count += 1
                    if counts and (counts[0] == 0 or counts[0] == counts[1]):
                        exact_count += 1
                    pct = 85 + (completed_count / total_scenes) * 15
                    self.progress_queue.put_nowait((pct, f"Splitting scenes Parallel ({completed_count}/{total_scenes})"))
                except Exception as e:
                    logger.error(f"Worker task failed: {e}")
                    # We continue despite errors in single scenes

        logger.info(
            "Parallel splitting finished. Processed %d/%d scenes (%d verified frame-exact).",
            completed_count, total_scenes, exact_count,
        )

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
