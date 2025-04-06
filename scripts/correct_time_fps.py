# %%
import os
import ffmpeg
import pandas as pd
import numpy as np
import joblib
from modules.settings import DEFAULT_VIDEO_DIR, DEVICE, PROJECT_DIR
from tqdm import tqdm
# %%
n_jobs = 4
verbose = 1

map_keyframes_v1_dir = f"{PROJECT_DIR}/db/map-keyframes"
map_keyframes_v2_dir = f"{PROJECT_DIR}/db/v2/map-keyframes"


# %%
def get_video_info(filename):
    probe = ffmpeg.probe(filename)
    video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
    return video_info


def check_if_v2_full(v1, v2):
    fnames_v1 = os.listdir(v1)
    fnames_v2 = os.listdir(v2)
    missing_list = []
    for fname in fnames_v1:
        if fname not in fnames_v2:
            missing_list.append(fname)
    return missing_list


# %%
jobs = []
d = {}

mk_v1_csv_paths = [f for f in os.listdir(map_keyframes_v1_dir)]  # fmt: skip
mk_v2_csv_paths = [f for f in os.listdir(map_keyframes_v2_dir)]  # fmt: skip

missing_list = check_if_v2_full(map_keyframes_v1_dir, map_keyframes_v2_dir)
if len(missing_list) == 0:
    print("[DEBUG] V2 is full")
else:
    print("[DEBUG] V2 is missing")
    print(missing_list)

for csv_name in tqdm(sorted(os.listdir(map_keyframes_v2_dir))):
    
    csv_name = os.path.splitext(csv_name)[0]
    csv_v2_path = os.path.join(map_keyframes_v2_dir, f"{csv_name}.csv")
    csv_v1_path = os.path.join(map_keyframes_v1_dir, f"{csv_name}.csv")
    video_path = os.path.join(DEFAULT_VIDEO_DIR, f"{csv_name}.mp4")

    video_info = get_video_info(video_path)
    duration = float(video_info["duration"])

    df_v2 = pd.read_csv(csv_v2_path)
    df_v1 = pd.read_csv(csv_v1_path)

    correct_fps = df_v1["fps"].iloc[0]

    frame_indices = df_v2["frame_idx"]
    last_frame_idx = frame_indices.iloc[-1] + 1
    frame_offset = duration / last_frame_idx

    correct_pts_times = frame_indices * frame_offset
    df_v2["pts_time"] = correct_pts_times.round(3)
    df_v2["frame_idx_extracted"] = df_v2["frame_idx"].copy()
    df_v2["fps_extracted"] = df_v2["fps"]
    
    df_v2["fps"] = correct_fps
    df_v2["frame_idx"] = df_v2["fps"] * df_v2["pts_time"]
    df_v2["frame_idx"] = df_v2["frame_idx"].round(0).astype(np.int64)
    

    df_v2.to_csv(csv_v2_path, index=False)
