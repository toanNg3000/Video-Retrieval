import os
from concurrent import futures
import subprocess

import joblib
import numpy as np
import pandas as pd
import PIL.Image
from tqdm import tqdm

from modules.settings import DEFAULT_VIDEO_DIR, DEVICE, PROJECT_DIR
from modules.transition_detector import TransitionDetector
from util.helper import cut_frames


fps = 25
threshold = 0.5
verbose = 1
map_keyframes_dir = f"{PROJECT_DIR}/db/v2/map-keyframes"
output_dir = f"{PROJECT_DIR}/db/v2/keyframes"
ratio = 16 / 9
resolution = 244
log_path = f"{PROJECT_DIR}/samples/extract_keyframe_log.csv"
w, h = int(resolution * ratio), resolution

n_jobs = 4
verbose = 1

transition_detector = TransitionDetector(
    device=DEVICE,
    framerate=fps,
    threshold=threshold,
    verbose=verbose,
)


def extract_thumbnail_by_time_args(video_path, t, out_path):
    try:
        args = transition_detector.extract_frame_by_time(
            filename=video_path,
            t=t,
            image_size=(w, h),
            return_command=True,
            out_filepath=out_path,
        )
        proc = subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            cwd=None,
        )
        out, err = proc.communicate()
        # for line in proc.stderr:

        # print(out)
        # print(err)

    except Exception as e:
        print(e)
        with open(log_path, "a") as f:
            f.write(f"{video_path}, {t}\n")

jobs = []
d = {}

csv_paths = [
    os.path.join(map_keyframes_dir, f) for f in os.listdir(map_keyframes_dir)
]


for csv_path in sorted(csv_paths):
    video_name = os.path.basename(csv_path)
    video_name = os.path.splitext(video_name)[0]
    video_path = os.path.join(DEFAULT_VIDEO_DIR, f"{video_name}.mp4")
    image_dir = os.path.join(output_dir, video_name)

    os.makedirs(image_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    frame_indices = df["frame_idx"]
    # pts_time = df["pts_time"]

    video_info = transition_detector.get_video_info(video_path)
    duration = float(video_info["duration"])
    last_frame_idx = frame_indices.iloc[-1] + 1
    frame_offset = duration / last_frame_idx
    for i, frame_idx in enumerate(frame_indices):
        image_path = os.path.join(image_dir, f"{i+1:04d}.jpg")

        if os.path.exists(image_path):
            if video_name not in d:
                d[video_name] = 1
            else:
                d[video_name] += 1

            if d[video_name] == df["n"].iloc[-1]:
                with open(f"{PROJECT_DIR}/samples/extracted_videos.csv", "a") as fh:
                    fh.write(f"{video_name}\n")

        else:
            t = frame_idx * frame_offset
            jobs.append(
                joblib.delayed(extract_thumbnail_by_time_args)(
                    video_path,
                    t,
                    image_path,
                )
            )

print(f"Number of Jobs: {len(jobs)}")
joblib.Parallel(n_jobs=2, verbose=verbose)(jobs)



# def main():
#     pass
# #     def extract_thumbnail_by_time(video_path, t, out_path):
# #         try:
# #             img = transition_detector.extract_frame_by_time(
# #                 filename=video_path,
# #                 t=t,
# #                 image_size=(w, h),
# #             )
# #             img = PIL.Image.fromarray(img)
# #             img.save(out_path)
# #         except:
# #             with open(log_path, "a") as f:
# #                 f.write(f"{video_path}, {t}\n")


# if __name__ == "__main__":
#     main()
