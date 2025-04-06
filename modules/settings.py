import os

import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))

DEFAULT_CACHE_DIR = f"{PROJECT_DIR}/output"
DEFAULT_DB_DIR = f"{PROJECT_DIR}/db"

# v1
DEFAULT_DB_V1_DIR = f"{PROJECT_DIR}/db"
DEFAULT_CLIP_FEATURE_DIR = f"{DEFAULT_DB_V1_DIR}/clip_features"
DEFAULT_MAP_KEYFRAMES_DIR = f"{DEFAULT_DB_V1_DIR}/map-keyframes"
DEFAULT_KEYFRAMES_DIR = f"{DEFAULT_DB_V1_DIR}/keyframes"
DEFAULT_FAISS_INDEX_V1_DIR = f"{DEFAULT_DB_V1_DIR}/faiss-index"
# v2
DEFAULT_DB_V2_DIR = f"{PROJECT_DIR}/db/v2"
DEFAULT_CLIP_FEATURE_V2_DIR = f"{DEFAULT_DB_V2_DIR}/clip_features"
DEFAULT_MAP_KEYFRAMES_V2_DIR = f"{DEFAULT_DB_V2_DIR}/map-keyframes"
DEFAULT_KEYFRAMES_V2_DIR = f"{DEFAULT_DB_V2_DIR}/keyframes"
DEFAULT_FAISS_INDEX_V2_DIR = f"{DEFAULT_DB_V2_DIR}/faiss-index"

# general
DEFAULT_MEDIA_INFO_DIR = f"{PROJECT_DIR}/db/media-info"
DEFAULT_THUMBNAILS_DIR = f"{PROJECT_DIR}/db/thumbnails"
DEFAULT_VIDEO_DIR = f"{PROJECT_DIR}/db/video"
DEFAULT_THUMBNAIL_RESOLUTION = 244

DEVICE = (
    "cuda" if torch.cuda.is_available()  # fmt: skip
    else "mps" if torch.backends.mps.is_available() # fmt: skip
    else "cpu" # fmt: skip
)

CLIP_MODEL_CHOICES = [
    # add more choices here
    # (model_name, dataset)
    ("ViT-L-14", "datacomp_xl_s13b_b90k"),
    ("ViT-H-14-378-quickgelu", "dfn5b"),
    ("ViT-B/32", ""),
]

CLIP_MODEL_NAMES = [name for name, _ in CLIP_MODEL_CHOICES]
DEFAULT_CLIP_MODEL_CHOICE = CLIP_MODEL_CHOICES[0][0]

DEFAULT_YOUTUBE_ID = "xYJ63OTMDL4"
DEFAULT_VIDEO_NAME = "NO VIDEO"
DEFAULT_EMBEDDED_HTML = """
<!DOCTYPE html>
<html>
    <body>
        <p align="center">Video name:
            <span id="current_video_name" style="font-size: 20px; color: red;">{video_name}</span>
            FPS:
            <span id="fps" style="font-size: 20px; color: red;">{fps}</span>
        </p>
        <p align="center">Start time: <span style="font-size: 20px; color: red;">{start_time}</span> (seconds)</p>
        <p align="center">
            <iframe src="https://www.youtube.com/embed/{youtube_id}?start={start_time}" 
                    style=" width: 100%; height: 350px;" 
                    allowfullscreen>
            </iframe>
        </p>
    </body>
</html>
"""
DEFAULT_FFMPEG_FRAMERATE = 25
