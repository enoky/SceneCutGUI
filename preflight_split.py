"""Report how _split_video will handle a given source before committing to a run.

Usage:  python preflight_split.py "path/to/video.mkv" [expected_scene_count]

Prints the decode/cut strategy that will be chosen and, for the slow path,
estimates what that costs so it can be caught before a multi-hour split.
"""
import subprocess
import sys
import time
from pathlib import Path

from scene_cut_gui import (
    SEEK_MARGIN_SEC,
    can_hwdecode,
    has_nvenc_hevc,
    probe_stream_props,
)


def track_report(path: str) -> None:
    """List every track, so demuxer complaints can be tied to a specific stream."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=index,codec_type,codec_name,pix_fmt,width,height,nb_frames",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, check=False,
    )
    print("Tracks:")
    for line in (result.stdout or "").strip().splitlines():
        print(f"  [{line.split(',')[0]}] {','.join(line.split(',')[1:])}")

    # A track FFmpeg refuses to decode shows up here as a decode failure.
    for kind, label in (("v:0", "video"), ("a:0", "audio")):
        probe = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-map", f"0:{kind}",
             "-frames:v", "1", "-frames:a", "1", "-f", "null", "-"],
            capture_output=True, text=True, check=False,
        )
        state = "decodes OK" if probe.returncode == 0 else \
            f"FAILS -> {(probe.stderr or '').strip().splitlines()[-1:]}"
        print(f"  first {label} track: {state}")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    scene_count = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if not Path(path).exists():
        print(f"No such file: {path}")
        return 2

    print(f"=== {path} ===")
    track_report(path)

    props = probe_stream_props(path)
    fps_num, fps_den = props["fps_num"], props["fps_den"]
    have_rate = bool(fps_num and fps_den)
    use_seek = have_rate and props["is_cfr"]
    nvenc = has_nvenc_hevc()

    print("\nProbe:")
    print(f"  frame rate   : {fps_num}/{fps_den}"
          f"{f' ({fps_num/fps_den:.6f} fps)' if have_rate else '  <-- UNUSABLE'}")
    print(f"  start_time   : {props['start_time']:.6f}s")
    print(f"  CFR          : {props['is_cfr']}")
    print(f"  audio        : {props['has_audio']}")
    for key in ("color_range", "color_space", "color_transfer", "color_primaries"):
        if props.get(key):
            print(f"  {key:<13}: {props[key]}")

    print("\nPipeline that will be used:")
    print(f"  encode : {'hevc_nvenc 10-bit (GPU)' if nvenc else 'libx265 10-bit (CPU!)'}")
    t0 = time.time()
    hwdec = nvenc and can_hwdecode(path)
    print(f"  decode : {'NVDEC (GPU)' if hwdec else 'CPU'}"
          f"   [test decode took {time.time()-t0:.1f}s]")
    print(f"  cutting: {'keyframe pre-seek (fast)' if use_seek else 'FULL DECODE PER CLIP (slow)'}")

    if not use_seek:
        print("\n  !! Every clip will re-decode this file from frame 0.")
        if scene_count:
            print(f"  !! At {scene_count} scenes that is roughly "
                  f"{scene_count} full-length decodes.")
        print("  !! Remux to a constant-frame-rate MP4 first to get the fast path:")
        print(f'       ffmpeg -i "{path}" -map 0:v:0 -map 0:a:0? -c copy '
              f'-fps_mode passthrough "{Path(path).with_suffix(".remux.mkv").name}"')
        print("     (or re-encode if the source is genuinely variable frame rate)")
    else:
        print(f"\n  Pre-seek margin: {SEEK_MARGIN_SEC}s before each cut.")
        print("  Cost per clip is flat regardless of source length.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
