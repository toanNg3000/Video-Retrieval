import json
import os
import re
from typing import List

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from PIL import Image

from modules.settings import (
    DEFAULT_MEDIA_INFO_DIR,
    DEFAULT_THUMBNAIL_RESOLUTION,
)

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))


def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as fp:
        lines = fp.read()
        return lines


def get_youtube_id(video_name: str) -> str:
    media_info_video_path = os.path.join(DEFAULT_MEDIA_INFO_DIR, f"{video_name}.json")
    with open(media_info_video_path, "r", encoding="utf-8") as fhandle:
        d = json.load(fhandle)
        url: str = d["watch_url"]  # "https://youtube.com/watch?v=1yHly8dYhIQ"
        youtube_id = url.index("watch?v=") + len("watch?v=")
        youtube_id = url[youtube_id:]
    return youtube_id


def get_video_name_from_youtube_id(youtube_id: str) -> str:
    media_info_video_paths = [
        os.path.join(DEFAULT_MEDIA_INFO_DIR, f)
        for f in os.listdir(DEFAULT_MEDIA_INFO_DIR)
    ]
    for media_info_video_path in media_info_video_paths:
        video_name = os.path.basename(media_info_video_path)
        video_name = video_name.replace(".json", "")
        with open(media_info_video_path, "r", encoding="utf-8") as fhandle:
            d = json.load(fhandle)
            url: str = d["watch_url"]  # "https://youtube.com/watch?v=1yHly8dYhIQ"
            id_to_compare = url.index("watch?v=") + len("watch?v=")
            id_to_compare = url[id_to_compare:]
            if id_to_compare == youtube_id:
                return video_name
    return "NO_VIDEO"


def parse_youtube_debug_info(debug_info: str):
    debug_info = json.loads(debug_info)
    t = debug_info["cmt"]
    youtube_id = debug_info["debug_videoId"]
    t = float(t)
    return youtube_id, t


def get_video_name_from_preview_video_html(html_doc: str) -> str:
    soup = BeautifulSoup(html_doc, "html.parser")
    result = soup.select_one('span[id="current_video_name"]')
    return result.text


def cosine_similarity(a, b):
    # Ensure the vectors are numpy arrays
    a = np.array(a)
    b = np.array(b)

    # Compute the dot product
    dot_product = np.dot(a, b)

    # Compute the norms (magnitudes) of the vectors
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # Compute cosine similarity
    similarity = dot_product / (norm_a * norm_b)

    return similarity


def sanitize_model_name(model_name):
    """
    Replaces any special characters in the model_name with "-".
    """
    return re.sub(r"[^a-zA-Z0-9-]", "-", model_name)


def get_keyframes_from_video_name(
    video_name: str,
    db_dir: str,
    use_thumbnail_if_possible: bool = True,
) -> List[str]:

    keyframes_dir = os.path.join(db_dir, "keyframes")
    thumbnails_dir = os.path.join(db_dir, "thumbnails")
    map_keyframes_dir = os.path.join(db_dir, "map-keyframes")

    keyframe_dir = os.path.join(keyframes_dir, video_name)
    thumbnail_dir = os.path.join(thumbnails_dir, video_name)
    map_keyframe_path = os.path.join(map_keyframes_dir, f"{video_name}.csv")
    map_keyframe = pd.read_csv(map_keyframe_path, usecols=["pts_time"])
    keyframe_paths_labels_pair = []
    for i, filename in enumerate(sorted(os.listdir(keyframe_dir))):

        keyframe_path = os.path.join(keyframe_dir, filename)
        thumbnail_path = os.path.join(thumbnail_dir, filename)
        if use_thumbnail_if_possible and os.path.exists(thumbnail_path):
            keyframe_path = thumbnail_path

        time_in_second = map_keyframe.iloc[i].to_list()[0]
        timestamp = int(time_in_second)
        h = timestamp // 3600
        m = timestamp // 60
        s = timestamp % 60
        timestamp = f"{h:02d}:{m:02d}:{s:02d}"
        filename = filename.replace(".jpg", "")
        keyframe_paths_labels_pair.append(
            (keyframe_path, f"{filename}-{timestamp}-{time_in_second:.2f}-{video_name}")
        )
    return keyframe_paths_labels_pair


def create_thumbnail(
    image_path,
    resolution=DEFAULT_THUMBNAIL_RESOLUTION,
    thumbnails_dir=None,
    verbose=False,
):
    # if thumbnails_dir is None:
    #     thumbnails_dir = DEFAULT_THUMBNAILS_DIR

    image_name = os.path.basename(image_path)
    video_name = os.path.basename(os.path.dirname(image_path))
    thumbnails_video_dir = os.path.join(thumbnails_dir, video_name)
    os.makedirs(thumbnails_video_dir, exist_ok=True)
    thumbnail_path = os.path.join(thumbnails_video_dir, f"{image_name}")

    if os.path.exists(thumbnail_path):
        if verbose:
            print(f"thumbnail for '{video_name}/{image_name}' already exist. Skipping.")
        return

    image = Image.open(image_path)
    image = resize_image(image, resolution)

    image.save(thumbnail_path)


def resize_image(image: Image, resolution: int):
    w, h = image.size
    ratio = w / h
    w = int(ratio * resolution)
    h = int(resolution)
    image = image.resize((w, h))
    return image


def get_fps_from_video_name(video_name: str, db_dir: str) -> float:
    map_keyframes_dir = os.path.join(db_dir, "map-keyframes")
    df = pd.read_csv(os.path.join(map_keyframes_dir, f"{video_name}.csv"))
    fps = df["fps"].iloc[0]
    return float(fps)


def split_image_into_3(im: Image):
    """Split image into 3 sections.

    Args:
        im (Image): the image with the format (H, W, C).

    Returns:
        Tuple[Image, Image, Image]: Left, Middle, Right images
    """
    h, w, *_ = im.shape
    mid_point = w // 2
    mid_left = mid_point - (mid_point // 2)
    mid_right = mid_left + h

    left = im[:, 0:h]
    middle = im[:, mid_left:mid_right]
    right = im[:, -h:]
    return left, middle, right


def cut_frames(start_idx, end_idx):
    idx1 = start_idx + int((end_idx - start_idx) / 3)
    idx2 = start_idx + int((end_idx - start_idx) / 3 * 2)
    frame_indices = [start_idx, idx1, idx2, end_idx]
    return frame_indices


def format_query_result(
    query_results,
    db_dir: str,
    use_thumbnail_if_possible: bool = True,
):
    results = []
    for video_keyframe, score, time_in_second in query_results:
        timestamp = int(time_in_second)
        h = timestamp // 3600
        m = timestamp // 60
        s = timestamp % 60

        keyframes_dir = os.path.join(db_dir, "keyframes")
        thumbnails_dir = os.path.join(db_dir, "thumbnails")

        keyframe_path = os.path.join(keyframes_dir, video_keyframe)
        thumbnail_path = os.path.join(thumbnails_dir, video_keyframe)

        if use_thumbnail_if_possible and os.path.exists(thumbnail_path):
            keyframe_path = thumbnail_path

        results.append(
            (
                keyframe_path,
                f"{video_keyframe}_{score:.3f}_{time_in_second}_{h:02d}:{m:02d}:{s:02d}",
            )
        )
    return results


def get_faiss_id(list_keyframe_names, faiss_dict: dict):
    # Create a reverse mapping of values to keys
    reverse_dict = {v: k for k, v in faiss_dict.items()}

    results_id = []

    for keyframe_name in list_keyframe_names:
        if keyframe_name not in reverse_dict:
            print(f"Invalid keyframe: {keyframe_name}")
        else:
            results_id.append(reverse_dict[keyframe_name])

    # Return a list of ids (keys) for each value in the input list
    return results_id

def build_submit_kis_data(video_name: str, time_in_ms):
    return {
        "answerSets": [
            {
                "answers": [
                    {
                        "mediaItemName": video_name,
                        "start": time_in_ms,
                        "end": time_in_ms,
                    }
                ]
            }
        ]
    }


def build_submit_qa_data(video_name: str, time_in_ms, answer: str):
    return {
        "answerSets": [
            {
                "answers": [
                    {
                        "text": f"{answer}-{video_name}-{time_in_ms}",
                    }
                ]
            }
        ]
    }