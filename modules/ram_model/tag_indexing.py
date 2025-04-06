import json
import os
import faiss
import pickle
import glob
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from modules.settings import PROJECT_DIR

with open(f'{PROJECT_DIR}/util/tag_list', 'rb') as fp:
    IFIDF_VOCAB = pickle.load(fp)
    print(f"Length of the initial vocabulary: {len(IFIDF_VOCAB)}")
    # configure the vocabulary is lower case
    IFIDF_VOCAB = set(i.lower() for i in IFIDF_VOCAB)
    print(f"Length of the vocabulary: {len(IFIDF_VOCAB)}")


def load_context(input_data_path=None, keyframes_dir=None):
    """
    keyframes_dir: .../db/keyframes/L01_V001/001.jpg
    input_data_path: .../ram_plus_encoded/L01_V001.txt --> the list of encoded image from the keyframes_dir
    """

    _context = []

    data_paths = glob.glob(input_data_path + '/*.txt')
    data_paths.sort(reverse=False, key=lambda x: x[-12:-4])

    for data_path in data_paths:  # ex: .../L01_V001.txt
        with open(data_path, 'r', encoding='utf-8') as f:
            data = f.readlines()
            data = [item.strip() for item in data]
            _context += data

    return _context


def _sorting_keyframe_path(path: str):
    """
    path: .../L01_V002/003.jpg
    """
    video_id, keyframe_id = path.split("/")[-2], path.split("/")[-1]

    part1 = video_id.split("_")[0][1:]  # "01"
    part2 = video_id.split("_")[1][1:]  # "002"
    part3 = keyframe_id.split('.')[0]  # "003"

    # print(part1 + part2 + part3)

    return part1 + part2 + part3


def write_tag_idx2keyframe(keyframes_dir=None, tfidf_tag_embedding_path=None):
    """
    keyframes_dir: .../db/keyframes/
    """

    tag_idx2keyframe = {}

    keyframes_paths = glob.glob(f"{keyframes_dir}/*/*.jpg")
    keyframes_paths.sort(key=_sorting_keyframe_path)

    for idx, keyframe_path in enumerate(keyframes_paths):
        tag_idx2keyframe[idx] = keyframe_path

    with open(f"{tfidf_tag_embedding_path}/tag_idx2keyframe.json", 'w') as f:
        json.dump(tag_idx2keyframe, f)

    return tag_idx2keyframe


def ifidf_indexing(input_data_path, faiss_indexing_output_dir, tfidf_vectorizer_output_dir):
    context = load_context(input_data_path)

    ifidf_transform = TfidfVectorizer(input='content', ngram_range=(1, 1), token_pattern=r"(?u)\b[\w\d]+\b",
                                      vocabulary=IFIDF_VOCAB)
    context_matrix = ifidf_transform.fit_transform(context).toarray()
    context_matrix.astype(np.float32)

    index = faiss.IndexFlatIP(len(IFIDF_VOCAB))

    assert context_matrix.shape[1] == len(IFIDF_VOCAB), "Mismatch between TF-IDF matrix columns and vocabulary size!"

    # check if dot product is better or cosine similarity

    # for i in range(context_matrix.shape[0]):
    index.add(context_matrix)

    os.makedirs(faiss_indexing_output_dir, exist_ok=True)

    faiss.write_index(index, f"{faiss_indexing_output_dir}/tag_faiss_indexing.bin")

    print(f"Faiss indexing path: {faiss_indexing_output_dir}/tag_faiss_indexing.bin")

    os.makedirs(tfidf_vectorizer_output_dir, exist_ok=True)

    with open(os.path.join(tfidf_vectorizer_output_dir, "tag_tfidf_vectorizer.pkl"),
              'wb') as f:
        pickle.dump(ifidf_transform, f)

        print(f"vectorizer path is {os.path.join(tfidf_vectorizer_output_dir, 'tag_tfidf_vectorizer.pkl')}")


if __name__ == '__main__':
    input_ram_encoded_path = f"{PROJECT_DIR}/db/ram_plus_encoded"
    tfidf_faiss_tag_embedding_path = f"{PROJECT_DIR}/db/tfidf_tag_embedding"
    tfidf_vectorizer_path = f"{PROJECT_DIR}/db/tfidf_tag_embedding"

    # create & write faiss indexer and tfidf_vectorizer
    ifidf_indexing(input_ram_encoded_path, tfidf_faiss_tag_embedding_path, tfidf_vectorizer_path)

    write_tag_idx2keyframe(f"{PROJECT_DIR}/db/keyframes", tfidf_vectorizer_path)

    # res = _sorting_keyframe_path(f'{PROJECT_DIR}/db/keyframes/L01_V001/001.jpg')
    #
    # print(res)

