import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from modules.settings import DEVICE, DEFAULT_VIDEO_DIR, PROJECT_DIR
from modules.transition_detector import TransitionDetector
from util.helper import cut_frames


def main():
    fps = 25
    threshold = 0.5
    verbose = 1
    start = None
    duration = None
    output_dir = f"{PROJECT_DIR}/db/v2/map-keyframes"

    os.makedirs(output_dir, exist_ok=True)

    transition_detector = TransitionDetector(
        device=DEVICE,
        framerate=fps,
        threshold=threshold,
        verbose=verbose,
    )

    filepaths = sorted(
        [os.path.join(DEFAULT_VIDEO_DIR, f) for f in os.listdir(DEFAULT_VIDEO_DIR)]
    )
    for filepath in tqdm(filepaths):
        filename = os.path.basename(filepath)
        filename = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{filename}.csv")

        if os.path.exists(output_path):
            print(f"{filename} already exists, skipping.")
            continue

        scenes, n_frames, framerate = transition_detector.predict_video_from_file(
            filename=filepath,
            start=start,
            duration=duration,
        )

        scenes = np.array([cut_frames(s, e) for s, e in scenes])
        scenes = scenes.flatten()

        results = []
        for i, frame_idx in enumerate(scenes):
            n = i + 1
            pts_time = frame_idx / fps
            results.append((n, pts_time, fps, frame_idx))

        headers = ["n", "pts_time", "fps", "frame_idx"]
        df = pd.DataFrame(results, columns=headers)
        df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
