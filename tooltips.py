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
    "min_scene_len": "The smallest allowed scene length in frames. Prevents creating very short, choppy scenes.",
    "weights_hue": "How much to weigh changes in color (e.g., red vs. blue) when comparing frames.",
    "weights_sat": "How much to weigh changes in color intensity (e.g., dull vs. vibrant) when comparing frames.",
    "weights_lum": "How much to weigh changes in brightness (e.g., light vs. dark) when comparing frames.",
    "weights_edges": "How much to weigh changes in edges and lines when comparing frames.",
    "luma_only": "If checked, compares frames by brightness only, ignoring color. Faster but may be less accurate.",
    "kernel_size": "Applies a blur to reduce noise before comparing frames. Use 0 for automatic (recommended).",
    
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
    "ai_validate": "Use a DINOv3 model to validate cuts and filter out flashes/fast motion. Requires a CUDA-enabled GPU.",
    "ai_window": "Number of frames before and after a cut to analyze for AI validation.",
    "flash_sensitivity": "Luma delta threshold for flash detection (15-80). Lower values detect more subtle flashes. Default: 30.",
    
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
    "ffmpeg_codec": "Video codec for FFmpeg splitting. NVENC options require an NVIDIA GPU.",
    "ffmpeg_preset": "NVENC preset. P1 is fastest (lowest quality), P7 is slowest (best quality). P5/P6 is a good balance.",
    "ffmpeg_cq": "Constant Quality level for NVENC. Lower values mean higher quality and larger file sizes. 18-28 is a reasonable range.",
    
    # Buttons
    "start_detection": "Start the full process: detect scenes and generate all selected outputs.",
}

