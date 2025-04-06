# from sentence_transformers import SentenceTransformer, util
from __future__ import absolute_import

import gc
import glob
import json
import os
from time import gmtime, strftime
from typing import List, Union

import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from modules.clip_model import ClipModel
from modules.settings import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CLIP_FEATURE_DIR,
    DEFAULT_MAP_KEYFRAMES_DIR,
    DEFAULT_KEYFRAMES_DIR,
    DEFAULT_MEDIA_INFO_DIR,
    DEVICE,
    PROJECT_DIR,
)
from util.helper import cosine_similarity, sanitize_model_name

from . import vi_translator


class Retriever:
    def __init__(
        self,
        clip_model_name: str = None,
        db_dir: str = None,
        top_k: int = 100,
        save_cache: bool = False,
        cache_dir: str = None,
        device=None,
    ):
        self.clip_model = ClipModel(clip_model_name, device)
        self.top_k = top_k

        self.db_dir = db_dir
        self.clip_feature_db = os.path.join(
            db_dir,
            "clip_features",
            sanitize_model_name(clip_model_name),
        )
        self.map_keyframes_db = os.path.join(db_dir, "map-keyframes")
        self.keyframes_db = os.path.join(db_dir, "keyframes")

        self.save_cache = save_cache
        self.cache_dir = cache_dir if cache_dir else DEFAULT_CACHE_DIR
        self.device = device if device else DEVICE

    def __del__(self):
        del self.clip_model

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad
    @torch.inference_mode
    def __call__(
        self,
        query_embedding,
        top_k: int = None,
        save_cache: bool = None,
        device=None,
        blacklist: List[str] = None,
    ):
        return self.retrieve(query_embedding, top_k, save_cache, device, blacklist)

    def get_db_path(self):
        # clip_model_name = self.clip_model.model_name
        return self.clip_feature_db

    @torch.no_grad
    @torch.inference_mode
    def retrieve(
        self,
        query_embedding,
        top_k=None,
        save_cache: bool = None,
        device=None,
        blacklist: List[str] = None,
    ) -> list:

        device = device if device else self.device
        top_k = top_k if top_k else self.top_k
        save_cache = save_cache if save_cache else self.save_cache
        blacklist = blacklist if isinstance(blacklist, list) else []

        query_embedding = query_embedding.to(device)
        results = []
        clip_feature_paths = []
        for f in os.listdir(self.clip_feature_db):
            video_pack = f.split("_")[0]
            if video_pack not in blacklist:
                clip_feature_paths.append(os.path.join(self.clip_feature_db, f))

            os.path.join(self.clip_feature_db, f)

        clip_feature_paths = sorted(clip_feature_paths)
        buffer_embeddings = None
        for path in tqdm(clip_feature_paths):
            video_name = os.path.basename(path)
            video_name = video_name.replace(".npy", "")

            # retrieve map-keyframe
            map_keyframe_path = os.path.join(self.map_keyframes_db, f"{video_name}.csv")
            map_keyframe = pd.read_csv(map_keyframe_path, usecols=["pts_time"])

            clip_features = np.load(path)
            keyframe_names = [
                f for f in os.listdir(f"{self.keyframes_db}/{video_name}")
            ]
            keyframe_names = sorted(keyframe_names)

            # parallel by stacking
            n_frames = clip_features.shape[0]
            clip_features = torch.tensor(clip_features).to(device)

            if buffer_embeddings is None:
                buffer_embeddings = query_embedding.repeat(n_frames, 1)
            else:
                n_text_embs = buffer_embeddings.shape[0]
                if n_text_embs < n_frames:
                    n_to_repeat = n_frames - n_text_embs
                    buffer_embeddings = torch.vstack(
                        [buffer_embeddings, query_embedding.repeat(n_to_repeat, 1)]
                    )

            scores = torch.cosine_similarity(
                buffer_embeddings[:n_frames], clip_features, dim=1
            )
            scores = scores.detach().to("cpu").numpy()

            for i, score in enumerate(scores):
                time_in_second = map_keyframe.iloc[i].to_list()[0]
                full_key_frame_path = (
                    f"{video_name}/{keyframe_names[i]}"  # L01_V001/001.jpg
                )
                results.append((full_key_frame_path, score, time_in_second))

            del clip_features
            del scores

        del buffer_embeddings

        results = sorted(results, key=lambda x: x[1], reverse=True)

        if top_k > 0:
            results = results[:top_k]

        if save_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            df = pd.DataFrame(results, columns=["video_name", "score"])
            df = df.sort_values("score", ascending=False)
            df = df.head(top_k)
            time_str = strftime("%Y_%m_%d_%H_%M_%S")

            df.to_csv(f"{self.cache_dir}/{time_str}.csv")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # print(results)
        return results

    def retrieve_text(
        self,
        query_text_eng,
        top_k=None,
        save_cache: bool = None,
        device=None,
        blacklist: List[str] = None,
    ) -> list:

        token_eng = self.clip_model.tokenize([query_text_eng]).to(device)
        query_embedding = self.clip_model.encode_text(token_eng)

        return self.retrieve(query_embedding, top_k, save_cache, device, blacklist)

    def retrieve_2_text(
        self,
        query_text_eng_1,
        query_text_eng_2,
        top_k=None,
        next_keyframe_duration: float = 12,
        first_k_limit: int = 2000,
        save_cache: bool = None,
        device=None,
        blacklist: List[str] = None,
    ):
        """
        - loop though keyframe_1 in a video
        - get next keyframes_x of keyframe_1 based on the next_keyframe_duration
        - compute similarity between query_text_eng_1 amd keyframe_1 -> score_1
        - compute similarity between query_text_eng_2 and keyframes_x(s)
        - get max similarity of the similairy(query_text_eng_2, keyframes_x) -> score_2
        return log(2+cos1) + log(2+cos2) as the result"""

        device = device if device else self.device
        top_k = top_k if top_k else self.top_k
        save_cache = save_cache if save_cache else self.save_cache

        token_eng = self.clip_model.tokenize(
            [query_text_eng_1, query_text_eng_2],
        ).to(device)
        query_embedding_1, query_embedding_2, *_ = self.clip_model.encode_text(
            token_eng
        )

        query_embedding_1 = query_embedding_1.to(device)

        # result for keyframe 1
        # keyframe_path = db/keyframes/L0_V001/001.jpg
        # TODO: refactor to not use the retrieve method (due to performance,
        # we shouldn't load the same thing multiple times)

        temp_results = self.retrieve(
            query_embedding_1, blacklist=blacklist, top_k=first_k_limit
        )  # -> (keyframe_path, score, time_in_second)

        # load all map_keyframe, so we don't have to reload anytime we need to search
        all_map_keyframes = {}
        all_map_keyframes_path = glob.glob(f"{self.map_keyframes_db}/*.csv")

        for map_keyframe_path in all_map_keyframes_path:
            video_idx = os.path.basename(map_keyframe_path)
            video_idx = video_idx.replace(".csv", "")

            all_map_keyframes.update({video_idx: pd.read_csv(map_keyframe_path)})

        results = []
        for keyframe_path, score_1, time_in_second in temp_results:
            video_id = keyframe_path.split("/")[-2]

            # load the map_keyframe
            map_keyframe = all_map_keyframes[video_id]

            next_cut_off_time = time_in_second + next_keyframe_duration

            # video_name / keyframe.jpg
            # map-keyframe
            # n | pts
            # 1 | 0.0
            # 2 | 5.0
            # 3 | 5.7
            next_keyframes_idx = (
                map_keyframe[
                    (map_keyframe.pts_time > time_in_second)
                    & (map_keyframe.pts_time <= next_cut_off_time)
                ]["n"]
                .sort_values()
                .to_list()
            )

            # clip feature in an numpy array has the starting index is 1
            clip_feature_idx = [kf_idx - 1 for kf_idx in next_keyframes_idx]

            # if next_keyframes_idx is [], we have already reached the end of the video => there is no temporal search here => return 0 score
            if len(next_keyframes_idx) == 0:
                final_score = 0
                results.append((keyframe_path, final_score, time_in_second))
                continue

            # extract the clip feature for the video
            clip_feature_paths = f"{self.clip_feature_db}/{video_id}.npy"

            clip_features = np.load(clip_feature_paths)

            next_frames_clip_feature = torch.tensor(clip_features[clip_feature_idx]).to(
                device
            )

            score_2 = max(
                torch.cosine_similarity(
                    next_frames_clip_feature, query_embedding_2, dim=1
                )
            )

            score_2 = score_2.detach().to("cpu").numpy()

            # the score ranges from -1 to 1, so the 2 + score ranges from 1 to 3
            # final_score = np.log(2 + score_1) + np.log(2 + score_2)

            # the score ranges from -1 to 1, so the (score + 1)/2 ranges from 0 to 1
            final_score = ((score_1 + 1) / 2) * ((score_2 + 1) / 2)

            results.append((keyframe_path, final_score, time_in_second))

        results = sorted(results, key=lambda x: x[1], reverse=True)

        if top_k > 0:
            results = results[:top_k]

        if save_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            df = pd.DataFrame(results, columns=["video_name", "score"])
            df = df.sort_values("score", ascending=False)
            df = df.head(top_k)
            time_str = strftime("%Y_%m_%d_%H_%M_%S")

            df.to_csv(f"{self.cache_dir}/{time_str}.csv")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    def retrieve_image(
        self,
        image,
        top_k=None,
        save_cache: bool = None,
        device=None,
        blacklist: List[str] = None,
    ) -> list:

        query_embedding = self.clip_model.encode_image(image)
        return self.retrieve(query_embedding, top_k, save_cache, device, blacklist)

    def retrieve_2_images(
        self,
        image_1,
        image_2,
        top_k=None,
        next_keyframe_duration: float = 12,
        first_k_limit: int = 2000,
        blacklist: List[str] = None,
    ):

        device = self.device

        query_embedding_1 = self.clip_model.encode_image(image_1).to(device)
        query_embedding_2 = self.clip_model.encode_image(image_2).to(device)

        # query_embedding_2 = self.clip_model.encode_image(image_2).to(device)
        # query_embedding_2 = query_embedding_2.to(device)

        temp_results = self.retrieve(
            query_embedding_1, blacklist=blacklist, top_k=first_k_limit, device=device
        )  # (keyframe_path, score, time_in_second)

        # load all map_keyframe, so we don't have to reload anytime we need to search
        all_map_keyframes = {}

        all_map_keyframes_path = glob.glob(f"{self.map_keyframes_db}/*.csv")

        for map_keyframe_path in all_map_keyframes_path:
            video_idx = os.path.basename(map_keyframe_path)
            video_idx = video_idx.replace(".csv", "")

            all_map_keyframes.update({video_idx: pd.read_csv(map_keyframe_path)})

        results = []

        for keyframe_path, score_1, time_in_second in temp_results:
            video_id = keyframe_path.split("/")[-2]

            # load the map_keyframe
            map_keyframe = all_map_keyframes[video_id]

            next_cut_off_time = time_in_second + next_keyframe_duration

            next_keyframes_idx = (
                map_keyframe[
                    (map_keyframe.pts_time > time_in_second)
                    & (map_keyframe.pts_time <= next_cut_off_time)
                ]["n"]
                .sort_values()
                .to_list()
            )

            # clip feature in an numpy array has the starting index is 1
            clip_feature_idx = [kf_idx - 1 for kf_idx in next_keyframes_idx]

            # if next_keyframes_idx is [], we have already reached the end of the video => there is no temporal search here => return 0 score
            if len(next_keyframes_idx) == 0:
                final_score = 0
                results.append((keyframe_path, final_score, time_in_second))
                continue

            # extract the clip feature for the video
            clip_feature_paths = f"{self.clip_feature_db}/{video_id}.npy"

            clip_features = np.load(clip_feature_paths)

            next_frames_clip_feature = torch.tensor(clip_features[clip_feature_idx]).to(
                device
            )

            score_2 = max(
                torch.cosine_similarity(
                    next_frames_clip_feature, query_embedding_2, dim=1
                )
            )

            score_2 = score_2.detach().to("cpu").numpy()

            # the score ranges from -1 to 1, so the 2 + score ranges from 1 to 3
            # final_score = np.log(2 + score_1) + np.log(2 + score_2)

            # the score ranges from -1 to 1, so the (score + 1)/2 ranges from 0 to 1
            final_score = ((score_1 + 1) / 2) * ((score_2 + 1) / 2)


            results.append((keyframe_path, final_score, time_in_second))

        results = sorted(results, key=lambda x: x[1], reverse=True)

        if top_k > 0:
            results = results[:top_k]

        return results

    def preprocess_image(self, image):
        return self.clip_model.preprocess(image)

    def set_model(self, clip_model_name: str):
        print(f"[DEBUG-retriever]: Deleteing model {self.clip_model.model_name}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[DEBUG-retriever]: Loading new model {clip_model_name}")
        self.clip_model.set_model(clip_model_name)
        self.clip_feature_db = os.path.join(
            self.db_dir,
            "clip_features",
            sanitize_model_name(clip_model_name),
        )

    def set_clip_feature_db(self, clip_feature_db: str):
        if not clip_feature_db:
            raise ValueError()
        self.clip_feature_db = clip_feature_db

    def set_map_keyframes_db(self, map_keyframes_db: str):
        if not map_keyframes_db:
            raise ValueError()
        self.map_keyframes_db = map_keyframes_db

    def set_keyframes_db(self, keyframes_db: str):
        if not keyframes_db:
            raise ValueError()
        self.keyframes_db = keyframes_db

    def set_db_dir(self, db_dir: str):
        if not db_dir:
            raise ValueError()
        self.db_dir = db_dir
        self.set_clip_feature_db(
            os.path.join(
                db_dir,
                "clip_features",
                sanitize_model_name(self.clip_model.model_name),
            )
        )
        self.set_keyframes_db(os.path.join(db_dir, "keyframes"))
        self.set_map_keyframes_db(os.path.join(db_dir, "map-keyframes"))


class FaissRetriever(Retriever):
    def __init__(
        self,
        clip_model_name: str = None,
        db_dir: str = None,
        top_k: int = 100,
        save_cache: bool = False,
        cache_dir: str = None,
        device=None,
    ):
        super().__init__(
            clip_model_name=clip_model_name,
            db_dir=db_dir,
            top_k=top_k,
            save_cache=save_cache,
            cache_dir=cache_dir,
            device=device,
        )
        self.faiss_index_db = os.path.join(db_dir, "faiss-index")
        self.indexer: faiss.IndexFlatIP = faiss.read_index(
            f"{self.faiss_index_db}/faiss_clip.bin"
        )
        with open(f"{self.faiss_index_db}/idx2keyframe.json") as fp:
            self.idx2keyframe = json.load(fp)

        with open(f"{self.faiss_index_db}/pack2idx.json") as fp:
            self.pack2idx = json.load(fp)

        # self.video2idx = json.load(f"{faiss_index_db}/video2idx.json")

    def retrieve(
        self,
        query_embedding: Union[np.ndarray, torch.Tensor],
        top_k=None,
        save_cache: bool = None,
        device=None,
        blacklist: List[str] = None,
    ) -> list:

        device = device if device else self.device
        top_k = top_k if top_k else self.top_k
        save_cache = save_cache if save_cache else self.save_cache
        blacklist = blacklist if isinstance(blacklist, list) else None

        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()
        if isinstance(query_embedding, np.ndarray):
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding[np.newaxis, ...]

        params = None
        if blacklist:
            ids_to_avoid = []
            for video_pack in blacklist:
                start, end = self.pack2idx[video_pack]
                ids_to_avoid.extend(list(range(start, end + 1)))

            selection = faiss.IDSelectorNot(faiss.IDSelectorBatch(ids_to_avoid))
            params = faiss.SearchParameters(sel=selection)

        scores, keyframe_ids = self.indexer.search(
            query_embedding, top_k, params=params
        )
        # outputs Dimensions are [n_queries, k]
        # since we only input 1 query:
        scores = scores[0]
        keyframe_ids = keyframe_ids[0]

        # results are padded with -1 if results doesn't have enough `top_k``
        valid_ids = np.flatnonzero(keyframe_ids != -1)
        scores = scores[valid_ids]
        keyframe_ids = keyframe_ids[valid_ids]

        full_key_frame_paths = [self.idx2keyframe[str(i)] for i in keyframe_ids]

        results = []  # full_keyframe_path, score, time_in_seconds
        for i, full_key_frame_path in enumerate(full_key_frame_paths):
            video_name, keyframe_name = full_key_frame_path.split("/")
            keyframe_name, _ = os.path.splitext(keyframe_name)

            map_keyframe_path = os.path.join(self.map_keyframes_db, f"{video_name}.csv")
            map_keyframe = pd.read_csv(map_keyframe_path, usecols=["pts_time"])

            n = int(keyframe_name) - 1
            time_in_second = map_keyframe.iloc[n].to_list()[0]
            results.append((full_key_frame_paths[i], scores[i], time_in_second))

        if save_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            df = pd.DataFrame(
                results, columns=["video_name", "score", "time_in_second"]
            )
            time_str = strftime("%Y_%m_%d_%H_%M_%S")
            df.to_csv(f"{self.cache_dir}/{time_str}.csv")

        return results

    def retrieve_faiss_next_search(
        self,
        query_text_next_search: str,
        top_k=None,
        save_cache: bool = None,
        device=None,
        faiss_idx: List[int] = None,
    ) -> list:
        device = device if device else self.device
        top_k = top_k if top_k else self.top_k
        save_cache = save_cache if save_cache else self.save_cache

        token_eng = self.clip_model.tokenize([query_text_next_search])
        query_embedding = self.clip_model.encode_text(token_eng)

        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()
        if isinstance(query_embedding, np.ndarray):
            if len(query_embedding.shape) == 1:
                query_embedding[np.newaxis, ...]

        params = None
        selection = faiss.IDSelectorBatch(faiss_idx)
        params = faiss.SearchParameters(sel=selection)

        scores, keyframe_ids = self.indexer.search(
            query_embedding, top_k, params=params
        )
        scores = scores[0]
        keyframe_ids = keyframe_ids[0]

        # results are padded with -1 if results doesn't have enough `top_k``
        valid_ids = np.flatnonzero(keyframe_ids != -1)
        scores = scores[valid_ids]
        keyframe_ids = keyframe_ids[valid_ids]

        full_key_frame_paths = [self.idx2keyframe[str(i)] for i in keyframe_ids]
        results = []
        for i, full_key_frame_path in enumerate(full_key_frame_paths):
            video_name, keyframe_name = full_key_frame_path.split("/")
            keyframe_name, _ = os.path.splitext(keyframe_name)

            map_keyframe_path = os.path.join(self.map_keyframes_db, f"{video_name}.csv")
            map_keyframe = pd.read_csv(map_keyframe_path, usecols=["pts_time"])

            n = int(keyframe_name) - 1
            time_in_second = map_keyframe.iloc[n].to_list()[0]
            results.append((full_key_frame_paths[i], scores[i], time_in_second))

        if save_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            df = pd.DataFrame(
                results, columns=["video_name", "score", "time_in_second"]
            )
            time_str = strftime("%Y_%m_%d_%H_%M_%S")
            df.to_csv(f"{self.cache_dir}/{time_str}.csv")
        return results

    def set_db_dir(self, db_dir: str):
        if not db_dir:
            raise ValueError()
        self.db_dir = db_dir
        self.set_clip_feature_db(
            os.path.join(
                db_dir,
                "clip_features",
                sanitize_model_name(self.clip_model.model_name),
            )
        )
        self.set_keyframes_db(os.path.join(db_dir, "keyframes"))
        self.set_map_keyframes_db(os.path.join(db_dir, "map-keyframes"))
        self.faiss_index_db = os.path.join(db_dir, "faiss-index")
        self.set_indexer(self.faiss_index_db)

    def set_indexer(self, faiss_index_db: str):
        self.indexer: faiss.IndexFlatIP = faiss.read_index(
            f"{faiss_index_db}/faiss_clip.bin"
        )
        with open(f"{faiss_index_db}/idx2keyframe.json") as fp:
            self.idx2keyframe = json.load(fp)

        with open(f"{faiss_index_db}/pack2idx.json") as fp:
            self.pack2idx = json.load(fp)


if __name__ == "__main__":
    sample_queries_dir = f"{PROJECT_DIR}/samples/pack-pretest"
    DEFAULT_CLIP_FEATURE_DIR = f"{PROJECT_DIR}/db/clip_features"

    retriever = Retriever(
        clip_model_name="ViT-B/32",
        clip_feature_db=f"{DEFAULT_CLIP_FEATURE_DIR}/ViT-B-32",
    )

    sample_queries_paths = [
        os.path.join(sample_queries_dir, f) for f in os.listdir(sample_queries_dir)
    ]

    sample_query_vi = ""
    with open(sample_queries_paths[0], "r", encoding="utf-8") as fhandle:
        sample_query_vi = fhandle.read()

    sample_query_eng = vi_translator.translate_vi2en(sample_query_vi)

    scores = retriever(sample_query_eng, top_k=100, save_cache=True)
    print(scores)
