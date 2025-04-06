import os
from typing import Tuple

import ffmpeg
import numpy as np
import torch

from modules.nets.transnetv2 import TransNetV2
from modules.settings import DEVICE, PROJECT_DIR

DEFAULT_TRANSNETV2_MODEL_PATH = os.path.join(
    PROJECT_DIR, "modules/checkpoints/transnetv2/transnetv2-pytorch-weights.pth"
)


class TransitionDetector:
    def __init__(
        self,
        model_path: str = None,
        device=None,
        framerate: int = 25,
        threshold: float = 0.5,
        verbose: int = 0,
        use_hwaccel: bool = False,
    ):
        self.model = TransNetV2()
        self.device = device if device else DEVICE
        self.framerate = framerate
        self.threshold = threshold
        self.verbose = verbose
        self.use_hwaccel = use_hwaccel
        self.IMAGE_SIZE_FOR_PREDICTION = (48, 27)

        self.model_path = model_path if model_path else DEFAULT_TRANSNETV2_MODEL_PATH
        state_dict = torch.load(self.model_path)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _input_iterator(self, frames):
        """Return windows of size 100 with padding to ensure correct input for the model."""
        no_padded_frames_start = 25
        no_padded_frames_end = (
            25 + 50 - (len(frames) % 50 if len(frames) % 50 != 0 else 50)
        )

        start_frame = np.expand_dims(frames[0], 0)
        end_frame = np.expand_dims(frames[-1], 0)
        padded_inputs = np.concatenate(
            [start_frame] * no_padded_frames_start
            + [frames]
            + [end_frame] * no_padded_frames_end,
            0,
        )

        ptr = 0
        while ptr + 100 <= len(padded_inputs):
            out = padded_inputs[ptr : ptr + 100]
            ptr += 50
            yield out[np.newaxis]

    def predictions_to_scenes(self, predictions: np.ndarray, threshold: float = None):
        """Convert frame predictions to scene boundaries."""
        threshold = threshold if threshold else self.threshold
        predictions = (predictions > threshold).astype(np.uint8)
        scenes = []
        t, t_prev, start = -1, 0, 0
        for i, t in enumerate(predictions):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t
        if t == 0:
            scenes.append([start, i])

        # Handle the case where all predictions are 1
        if len(scenes) == 0:
            return np.array([[0, len(predictions) - 1]], dtype=np.int32)

        return np.array(scenes, dtype=np.int32)

    def predict_raw(self, video):
        # video shape: [batch, 27, 48, 3]
        """Run raw inference on the video and return frame predictions."""
        self.model.to(self.device)
        with torch.no_grad():
            predictions = []
            for inp in self._input_iterator(video):
                video_tensor = torch.from_numpy(inp).to(self.device)
                single_frame_pred, all_frame_pred = self.model(video_tensor)

                single_frame_pred = torch.sigmoid(single_frame_pred).cpu().numpy()
                all_frame_pred = torch.sigmoid(all_frame_pred["many_hot"]).cpu().numpy()

                predictions.append(
                    (single_frame_pred[0, 25:75, 0], all_frame_pred[0, 25:75, 0])
                )
                if self.verbose != 0:
                    print(
                        f"\r[TransNetV2] Processing video frames {min(len(predictions) * 50, len(video))}/{len(video)}",
                        end="",
                    )

            single_frame_pred = np.concatenate(
                [single_ for single_, all_ in predictions]
            )
            all_frames_pred = np.concatenate([all_ for single_, all_ in predictions])

            return (
                video.shape[0],
                single_frame_pred[: len(video)],
                all_frames_pred[: len(video)],
            )

    def predict_video_from_file(
        self,
        filename: str,
        start=None,
        duration=None,
        framerate=None,
    ):
        """Predict scenes in the given video."""
        framerate = framerate if framerate else self.framerate
        video, _ = self.load_video(
            filename,
            start,
            duration,
            image_size=self.IMAGE_SIZE_FOR_PREDICTION,
            framerate=framerate,
        )

        n_frames, single_frame_pred, _ = self.predict_raw(video)
        scenes = self.predictions_to_scenes(single_frame_pred)
        return scenes, n_frames, framerate

    def predict_video_from_array(self, video):
        assert video.shape[1:] == (27, 48, 3), "Invalid video shape"
        n_frames, single_frame_pred, _ = self.predict_raw(video)
        scenes = self.predictions_to_scenes(single_frame_pred)
        return scenes, n_frames

    def load_video(
        self,
        filename: str,
        start=None,
        duration=None,
        image_size: Tuple[int, int] = None,
        framerate=None,
        remove_audio=True,
        use_hwaccel=None,
    ):
        framerate = framerate if framerate else self.framerate
        use_hwaccel = use_hwaccel if use_hwaccel else self.use_hwaccel
        ffmpeg_out_args = {
            "format": "rawvideo",
            "pix_fmt": "rgb24",
        }
        if start:
            ffmpeg_out_args["ss"] = start
        if duration:
            ffmpeg_out_args["t"] = duration
        if framerate:
            ffmpeg_out_args["r"] = framerate
        if image_size:
            w, h = image_size
        else:
            video_info = self.get_video_info(filename)
            w = int(video_info["width"])
            h = int(video_info["height"])
        if remove_audio:
            ffmpeg_out_args["an"] = None

        ffmpeg_out_args["s"] = f"{w}x{h}"

        if self.verbose != 0:
            print(f"\r\n[TransitionDetector] Loading Video {filename} ...", end=" ")

        ffmpeg_in_args = {"r": framerate}
        if use_hwaccel and self.device == "cuda":
            ffmpeg_in_args["hwaccel"] = "cuda"
            if h % 2 != 0:
                ffmpeg_out_args["s"] = f"{w}x{h+1}"
            else:
                ffmpeg_out_args["s"] = f"{w}x{h}"

        video_stream, _ = (
            ffmpeg.input(filename, **ffmpeg_in_args)
            .output("pipe:", **ffmpeg_out_args)
            .run(capture_stdout=True, capture_stderr=True)
        )

        if self.verbose != 0:
            print("done.")

        if use_hwaccel and self.device == "cuda" and h % 2 != 0:
            video = np.frombuffer(video_stream, np.uint8).reshape([-1, h + 1, w, 3])
            video = video[:, :-1, :, :]
        else:
            video = np.frombuffer(video_stream, np.uint8).reshape([-1, h, w, 3])

        return video, framerate

    def convert_frame_to_time(self, frame: int, framerate: int = None):
        framerate = framerate if framerate else self.framerate
        return frame / framerate

    def extract_frame(
        self,
        filename: str,
        frame,
        image_size: Tuple[int, int] = None,
        framerate=None,
    ):
        framerate = framerate if framerate else self.framerate
        ffmpeg_output_args = dict(
            format="rawvideo",
            pix_fmt="rgb24",
            vframes=1,
            r=framerate,
        )

        if image_size:
            w, h = image_size
            ffmpeg_output_args["s"] = f"{w}x{h}"
        else:
            video_info = self.get_video_info(filename)
            w = int(video_info["width"])
            h = int(video_info["height"])

        out, _ = (
            ffmpeg.input(filename, r=framerate)
            .filter_("select", "gte(n,{})".format(frame))
            .output("pipe:", **ffmpeg_output_args)
            .run(capture_stdout=True, capture_stderr=True)
        )
        return np.frombuffer(out, np.uint8).reshape([h, w, 3])

    def extract_frame_by_time(
        self,
        filename: str,
        t: float,
        image_size: Tuple[int, int] = None,
        framerate=None,
        return_command=False,
        out_filepath: str = None,
    ):
        framerate = framerate if framerate else self.framerate
        ffmpeg_output_args = dict(
            # format="rawvideo",
            # pix_fmt="rgb24",
            vframes=1,
        )

        if image_size:
            w, h = image_size
            ffmpeg_output_args["s"] = f"{w}x{h}"
        else:
            video_info = self.get_video_info(filename)
            w = int(video_info["width"])
            h = int(video_info["height"])

        stream = ffmpeg.input(filename, ss=f"{t:.3f}")

        if return_command:
            if not out_filepath:
                raise ValueError("User must provide `out_filename` when `return_command` is True")
            stream = stream.output(out_filepath, **ffmpeg_output_args)
            return stream.compile()

        out, _ = stream.output("pipe:", **ffmpeg_output_args).run(
            capture_stdout=True, capture_stderr=True
        )

        return np.frombuffer(out, np.uint8).reshape([h, w, 3])

    def get_video_info(self, filename):
        probe = ffmpeg.probe(filename)
        video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
        return video_info
