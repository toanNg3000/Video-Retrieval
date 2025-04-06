import pickle
import os
import faiss
import json
import re
import numpy as np
import pandas as pd

import google.generativeai as genai

from modules.settings import PROJECT_DIR

genai.configure(api_key=os.environ["API_KEY"])

model = genai.GenerativeModel("gemini-1.5-flash")

with open(f"{PROJECT_DIR}/util/tag_list", "rb") as fp:
    TAG_LIST = pickle.load(fp)


class TagRetrieval:

    def __init__(self, db_dir, top_k=100):
        """
        faiss_tag_indexing_path: f"{PROJECT_DIR}/db/tfidf_tag_embedding/tag_faiss_indexing.bin"
        tag_tfidf_vectorizer_path: f"{PROJECT_DIR}/db/tfidf_tag_embedding/tag_tfidf_vectorizer.pkl"
        db_dir: f"{PROJECT_DIR}/db"
        top_k: int = 100
        """
        self._db_dir = db_dir
        self._faiss_tag_indexing_path = f"{self.db_dir}/tfidf_tag_embedding/tag_faiss_indexing.bin"
        self._tag_tfidf_vectorizer_path = f"{self.db_dir}/tfidf_tag_embedding/tag_tfidf_vectorizer.pkl"
        self._faiss_tag_index = faiss.read_index(self.faiss_tag_indexing_path)
        self.map_keyframes_db = f"{self.db_dir}/map-keyframes"
        self._top_k = top_k

        with open(self.tag_tfidf_vectorizer_path, 'rb') as f:
            self._ifidf_vectorizer = pickle.load(f)

        with open(f"{os.path.dirname(self.faiss_tag_indexing_path)}/tag_idx2keyframe.json") as fp:
            self.tag_idx2keyframe = json.load(fp)
            
        

    def __call__(self, tags_query: str, top_k: int = None):
        top_k = top_k if top_k else self.top_k
        return self.retrieval(tags_query, top_k)

    @property
    def faiss_tag_indexing_path(self):
        return self._faiss_tag_indexing_path

    @property
    def db_dir(self):
        return self._db_dir


    @property
    def tag_tfidf_vectorizer_path(self):
        return self._tag_tfidf_vectorizer_path

    @property
    def faiss_tag_index(self):
        return self._faiss_tag_index

    @property
    def ifidf_vectorizer(self):
        return self._ifidf_vectorizer

    @property
    def top_k(self):
        return self._top_k

    @top_k.setter
    def top_k(self, x):

        if not isinstance(x, int):
            raise ValueError("top_k must be an integer")
        if x < 1:
            raise ValueError("top_k must be greater or equal to 1")

        self._top_k = x

    @classmethod
    def recommend_tag(cls, input_query):


        prompt = f"""
        Given the tags: {' '.join(TAG_LIST)}

        Select all related tags (objects, scene, actions, colors, etc) describing the query from this list based on the following query: {input_query}.

        Format the result as ###tag1 tag2 ...###.     
        """

        response = model.generate_content(prompt).text

        def get_tag_list(input_string):

            result = re.search(r"###(.*?)###", input_string)

            if result:
                gemini_tags = result.group(1).strip()
                return ' '.join([tag for tag in gemini_tags.split(" ") if tag in TAG_LIST])
            return 'Can Not Find Corresponding Tag'

        return get_tag_list(response)

    def transform_input(self, tag_query: str):
        tag_query_vector = self.ifidf_vectorizer.transform([tag_query]).toarray().astype(np.float32)
        return tag_query_vector

    def retrieval(self, tags_query: str, top_k: int = None):

        scores, keyframe_ids = self.faiss_tag_index.search(self.transform_input(tags_query), top_k)

        scores = scores[0]
        keyframe_ids = keyframe_ids[0]
        full_key_frame_paths = [self.tag_idx2keyframe[str(i)] for i in keyframe_ids]

        results = []  # full_keyframe_path, score, time_in_seconds
        for i, full_key_frame_path in enumerate(full_key_frame_paths):
            video_name, keyframe_name = full_key_frame_path.split("/")[-2], full_key_frame_path.split("/")[-1]
            keyframe_name, extension = os.path.splitext(keyframe_name)

            map_keyframe_path = os.path.join(self.map_keyframes_db, f"{video_name}.csv")
            map_keyframe = pd.read_csv(map_keyframe_path, usecols=["pts_time"])

            n = int(keyframe_name) - 1
            time_in_second = map_keyframe.iloc[n].to_list()[0]
            results.append((f"{video_name}/{keyframe_name}{extension}", scores[i], time_in_second))

        return results


if __name__ == '__main__':

    test_db_dir = f"{PROJECT_DIR}/db"

    tag_retrieval = TagRetrieval(test_db_dir, top_k=100)

    input_query = "city city_skyline city_view night night_view sea sky skyline sun sunset water"
    test_tag_retrieval = tag_retrieval(tags_query=input_query, top_k=10)

    print(f"Test tag retrieval result: {test_tag_retrieval}")

    # res = model.generate_content("Write a story about a magic backpack.")

    """Bắt đầu cảnh 1 đồng lúa, có một chiếc xe đang cắt lúa màu đỏ và trắng. Trên xe có hai người, người ngồi cao 
    hơn đội chiếc nón màu đỏ. Tiếp đó là một nhóm người đang mang vác bao tải."""

    test_recommend_tag = TagRetrieval.recommend_tag(
        "At the beginning of the scene in a rice field, there is a red and white rice cutting cart. There are two "
        "people on the cart, the taller one is wearing a red hat. Next is a group of people carrying sacks.")

    print(test_recommend_tag)
    # output: rice_field rice_cutting_cart red_hat sack

    ### rice_field rice_cutting_cart sack red_hat group ###
