import torch
from torchreid.reid.utils.feature_extractor import FeatureExtractor
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from collections import defaultdict

class OSNetReID:
    extractor = None

    @classmethod
    def _init_extractor(cls):
        if cls.extractor is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cls.extractor = FeatureExtractor(
                model_name='osnet_x1_0',
                device=device
            )

# Todo: esp, min_sampleを引数で設定できるようにする
    @classmethod
    def cluster_imgs(cls, img_paths):
        cls._init_extractor()
        feats = cls.extractor(img_paths) 
        dist_mat = 1 - cosine_similarity(feats)
        dist_mat = np.clip(dist_mat, 0, None) 
        cluster = DBSCAN(eps=0.3, min_samples=2, metric='precomputed')
        labels = cluster.fit_predict(dist_mat)
        groups = defaultdict(list)
        for path, label in zip(img_paths, labels):
            groups[label].append(path)
        return groups
    
if __name__ == "__main__":

    import glob
    import os
    import sys
    from img_utils.img_utils import load_img_paths_from_dir

    input_dir = sys.argv[1]
    img_paths = load_img_paths_from_dir(input_dir)

    groups = OSNetReID.cluster_imgs(img_paths)

    for label, paths in groups.items():
        print(f"Cluster :{label}")
        for path in paths:
            print(f"  {path}")

