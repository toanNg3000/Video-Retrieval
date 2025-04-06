import os
import faiss
import numpy as np
from tqdm import tqdm
import json
from modules.settings import (
    PROJECT_DIR,
    DEFAULT_CLIP_FEATURE_DIR,
    DEFAULT_CLIP_FEATURE_V2_DIR,
    DEFAULT_DB_V1_DIR,
    DEFAULT_DB_V2_DIR,
    DEFAULT_FAISS_INDEX_V2_DIR
)

def faiss_indexing(model: str, feature_length: int, db_dir: str):
    """
    - Indexing clip features in the index object
    - Create an dictionary for mapping from the index of the index object to the keyframe
    """

    features_dir = os.path.join(db_dir, "clip_features", model)
    keyframes_dir = os.path.join(db_dir, "keyframes")

    print(features_dir)
    print(db_dir)

    idx2keyframe = {}
    video2idx = {}
    pack2idx = {}

    i = 0

    index = faiss.IndexFlatIP(feature_length)

    for feature_file in tqdm(
        sorted(os.listdir(features_dir))
    ):  # feature: 'L01_V001.npy'

        # for feature_path in tqdm(sorted(glob.glob(os.path.join(features_dir, data_part) +'/*.npy'))):
        video_name, _ = os.path.splitext(feature_file)  # remove .npy, L01_V001

        keyframe_names = os.listdir(f"{keyframes_dir}/{video_name}")
        keyframe_names = sorted(keyframe_names)

        feature_path = os.path.join(features_dir, feature_file)
        feats = np.load(feature_path)

        start_idx = i
        for idx, feat in enumerate(feats):
            feat = feat.astype(np.float32).reshape(1, -1)
            faiss.normalize_L2(feat)  # Normalize for cosine similarity
            index.add(feat)  # Add feature to the index
            # create a starting index 0
            # update
            idx2keyframe[i] = f"{video_name}/{keyframe_names[idx]}"
            i += 1
        video2idx[video_name] = (start_idx, i - 1)

    for video_name in video2idx:
        pack_name, _ = video_name.split("_")
        start, end = video2idx[video_name]
        if pack_name not in pack2idx:
            pack2idx[pack_name] = [start, end]
        else:
            pack2idx[pack_name][1] = end
        
    # Write the index to a binary file
    index_path = os.path.join(db_dir, "faiss-index", "faiss_clip.bin")
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)

    # Save idx2keyframe mapping as JSON
    idx2keyframe_path = os.path.join(db_dir, "faiss-index", "idx2keyframe.json")
    with open(idx2keyframe_path, "w") as f:
        json.dump(idx2keyframe, f, indent=4)

    video2idx_path = os.path.join(db_dir, "faiss-index", "video2idx.json")
    with open(video2idx_path, "w") as f:
        json.dump(video2idx, f, indent=4)

    pack2idx_path = os.path.join(db_dir, "faiss-index", "pack2idx.json")
    with open(pack2idx_path, "w") as f:
        json.dump(pack2idx, f, indent=4)
    return index

if __name__ == '__main__':
    MODEL = "ViT-L-14"
    feature_length = 768
    index = faiss_indexing(MODEL, feature_length, DEFAULT_DB_V2_DIR)


