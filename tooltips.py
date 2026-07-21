"""
Tooltip Definitions for Scene Cut Detection GUI
-----------------------------------------------

This file contains a centralized dictionary of tooltips used throughout
the SceneDetect GUI application. Keeping them in a separate file
improves organization and makes them easier to manage and translate.
"""

TOOLTIPS: dict[str, str] = {
    # Video input and backend
    "video_path": "Path to the video file to analyze for scene changes.",
    "output_dir": "Path to the folder where all output files (CSV, images, etc.) will be saved.",
    "backend": "Video decoding backend to use (opencv, pyav, moviepy). Note: AI validation and CUDA splitting require an OpenCV-compatible video.",
    
    # Detector selection
    "detector_type": "The algorithm used for detecting scene changes.",
    
    # ContentDetector parameters
    "threshold": "How different two frames must be to be a cut. Lower values are more sensitive and will find more scenes.",
    "min_scene_len": "Minimum scene length in frames. Scenes shorter than ~0.30s are preserved by default to avoid losing very short cuts.",
    "weights_hue": "How much to weigh changes in color (e.g., red vs. blue) when comparing frames.",
    "weights_sat": "How much to weigh changes in color intensity (e.g., dull vs. vibrant) when comparing frames.",
    "weights_lum": "How much to weigh changes in brightness (e.g., light vs. dark) when comparing frames.",
    "weights_edges": "How much to weigh changes in edges and lines when comparing frames.",
    "luma_only": "If checked, compares frames by brightness only, ignoring color. Faster but may be less accurate.",
    "kernel_size": "Applies a blur to reduce noise before comparing frames. Use 0 for automatic (recommended).",
    
    # OmniShotCut parameters
    "omnishot_mode": "clean_shot: keep only hard cuts (transitions like dissolves/fades are absorbed into the previous scene). default: keep every detected shot, including transitions as their own segments.",
    "omnishot_overlap": "Frames shared between consecutive inference windows. Higher values cost more time but reduce missed cuts at window edges. Default: 20.",
    "omnishot_confidence": "Minimum confidence a cut must have to be kept (0-1), based on how precisely the model can pin down the cut's exact frame. This is OmniShotCut's equivalent of a threshold: higher values keep fewer, more certain cuts, and rejected cuts merge into the following scene. 0 keeps everything (default). It bites quickly - around 0.5 typically drops a third of cuts - so start near 0.3 and raise it slowly. The log lists the confidence spread for your video after each run.",

    # AdaptiveDetector parameters
    "adaptive_threshold": "Sensitivity for the adaptive detector. Higher values require bigger changes to trigger a cut, finding fewer scenes.",
    "window_width": "Number of surrounding frames to check when adapting the threshold. Helps handle high-motion scenes.",
    "min_content_val": "Ignores changes in very dark or blank frames to avoid false detections during fades.",
    
    # ThresholdDetector parameters
    "fade_bias": "Adjusts sensitivity for detecting gradual fades. Positive values find more fades, negative values find fewer.",
    "add_final_scene": "If checked, ensures the very end of the video is always marked as a scene boundary.",
    
    # HashDetector parameters
    "size": "The level of detail used for comparing frame hashes. Larger sizes are more precise but slower.",
    "lowpass": "Amount of smoothing applied before hashing. Helps ignore minor visual noise.",
    
    # AI Validation
    "ai_validate": "Use DINOv3 AI validation to confirm cuts and filter flashes/fast motion. Requires a CUDA-enabled GPU.",
    "ai_window": "Number of frames before and after a cut to analyze for AI validation.",
    "flash_sensitivity": "Sensitivity for detecting flash spikes versus scene cuts. Lower values (15-25) detect flashes more aggressively. Default: 15.",
    
    # Output and statistics
    "stats_enabled": "Enable logging of detection metrics to a CSV file.",
    "export_csv": "Generate a CSV file listing detected scenes.",
    "export_html": "Generate an HTML report of detected scenes.",
    "export_sc": "Generate a .sc scene cut file for use in other applications.",
    "save_images": "Extract and save thumbnail images for each detected scene.",
    "num_images": "Number of images to extract per detected scene.",
    "frame_margin": "Number of frames to skip at the start/end of a scene when extracting images.",
    "split_ffmpeg": "Use FFmpeg to split the video into a separate file for each scene.",
    
    # FFmpeg Settings
    "ffmpeg_codec": "Scene clips are always written as 10-bit HEVC (Main10, p010le). The full pipeline runs on the NVIDIA GPU: NVDEC decode, CUDA 8->10-bit conversion, NVENC encode, with no frame copied to system memory. Falls back to CPU decoding for sources NVDEC cannot handle, and to libx265 10-bit if this FFmpeg build has no NVENC.",
    "ffmpeg_preset": "NVENC preset. P1 is fastest (lowest quality), P7 is slowest (best quality). P5/P6 is a good balance.",
    "ffmpeg_cq": "Constant QP level for the 10-bit HEVC encode. Lower values mean higher quality and larger file sizes. 16-24 is a reasonable range for 10-bit HEVC.",
    "split_ffmpeg_accuracy": "Cuts land on the exact frame the detector chose, never a keyframe-rounded boundary, and clips are re-timed to strict CFR at the source's exact frame rate. On constant-frame-rate sources a keyframe pre-seek keeps the cost per clip flat, so sources with thousands of cuts stay fast; every clip's frame count is verified afterwards.",
    
    # Buttons
    "start_detection": "Start the full process: detect scenes and generate all selected outputs.",
}

