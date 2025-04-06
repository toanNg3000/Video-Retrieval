from __future__ import absolute_import

import gc
from typing import List

import clip
import open_clip
import torch

from modules.settings import CLIP_MODEL_CHOICES, DEFAULT_CLIP_MODEL_CHOICE, DEVICE


class ClipModel:
    def __init__(self, model_name=DEFAULT_CLIP_MODEL_CHOICE, device=None):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.preprocessor = None
        self.device = device if device else DEVICE

        self.set_model(model_name)

    def __del__(self):
        print(f"[DEBUG-clip-model]: Deleting {self.model_name}")
        del self.model
        del self.tokenizer
        del self.preprocessor

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def tokenize(self, inputs: List[str]):
        if not self.tokenizer:
            raise ValueError(
                "ClipModel is not initialized or tokenizer does not exist."
            )

        tokens = self.tokenizer(inputs)
        tokens = tokens.to(self.device)
        return tokens

    def preprocess(self, image_arr):
        if not self.preprocessor:
            raise ValueError(
                "ClipModel is not initialized or preprocessor does not exist."
            )

        result = self.preprocessor(image_arr)
        result = result.to(self.device)
        return result

    @torch.no_grad
    @torch.inference_mode
    def encode_image(self, image):
        if not self.model:
            raise ValueError("ClipModel is not initialized or model does not exist.")

        return self.model.encode_image(image)

    @torch.no_grad
    @torch.inference_mode
    def encode_text(self, tokenized_text):
        if not self.model:
            raise ValueError("ClipModel is not initialized or model does not exist.")

        return self.model.encode_text(tokenized_text)

    @torch.no_grad
    @torch.inference_mode
    def __call__(self, input_1, input_2):
        if not self.model:
            raise ValueError("ClipModel is not initialized or model does not exist.")

        return self.model(input_1, input_2)

    def set_model(self, model_name: str):
        if "ViT-B/32" == model_name:
            print(f"[DEBUG-clip-model]: Deleting {self.model_name}")
            del self.model
            del self.tokenizer
            del self.preprocessor

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.model_name = model_name
            self.model, self.preprocessor = clip.load(model_name, device=self.device)
            self.tokenizer = clip.tokenize
            return

        for _model_name, dataset in CLIP_MODEL_CHOICES:
            if model_name == _model_name:
                print(f"[DEBUG-clip-model]: Deleting {self.model_name}")
                del self.model
                del self.tokenizer
                del self.preprocessor

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[DEBUG-clip-model]: Loading {self.model_name}")
                self.model_name = model_name
                self.model, _, self.preprocessor = (
                    open_clip.create_model_and_transforms(
                        model_name=model_name,
                        pretrained=dataset,
                        device=self.device,
                    )
                )
                self.tokenizer = open_clip.get_tokenizer(model_name)
                return

        raise ValueError(
            f"`model_name` does not exist. Must be one of {CLIP_MODEL_CHOICES}. Got '{model_name}'."
        )
