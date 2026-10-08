from networkx.algorithms import shortest_paths
import torch
from torchreid.reid.utils.feature_extractor import FeatureExtractor
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx


class OSNet:
    extractor = None


    @classmethod
    def _init_extractor(cls):
        if cls.extractor is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cls.extractor = FeatureExtractor(
                model_name='osnet_x1_0',
                device=device
            )

    @classmethod
    def dbscan_by_auto_eps(cls, img_paths, min_samples=2):
        cls._init_extractor()
        feats = np.asarray(cls.extractor(img_paths))
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        feats_normalized = feats / norms
        eps = cls.find_optimal_eps(feats_normalized)
        cluster = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        labels = cluster.fit_predict(feats_normalized)
        groups = defaultdict(list)
        for path, label in zip(img_paths, labels):
            groups[label].append(path)
        return groups, eps

    @staticmethod
    def find_optimal_eps(feats_norm, k=2):
        neigh = NearestNeighbors(n_neighbors=k+1, metric='euclidean') # n_neighbors:自分自身を含む何番目まで遠い距離を探すか（1はじまり） 自分自身を含むためk+1
        neigh.fit(feats_norm)
        distances, _ = neigh.kneighbors(feats_norm)
        k_distances = np.sort(distances[:, k]) # 何番目の距離を取り出すか．0はじまりのためk

        acc = np.diff(k_distances, n=2)
        # max_idx = np.argmax(np.abs(acc))
        max_idx = np.argmax(acc)
        eps = k_distances[max_idx+1]
        # 谷を直接epsに設定すると，谷部分に相当する点がクラスタに含まれなくなってしまうため，小さな値を足し合わせる
        eps = eps + 1e-5
        return eps

    @classmethod
    def clustering_by_mst(cls, img_paths, max_size, min_size=3):
        cls._init_extractor()
        feats = np.asarray(cls.extractor(img_paths))
        labels, components = cls.mst_constrained_clustering(feats, max_size, min_size)
        groups = defaultdict(list)
        for path, label in zip(img_paths, labels):
            groups[label].append(path)
        return groups

    @staticmethod
    def mst_constrained_clustering(feats, max_size, min_size=3):
        # 1. 特徴量の正規化 (L2 Normalization)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        feats_normalized = feats / norms
        # 2. 距離行列の計算と最小全域木(MST)の構築
        dist_matrix = squareform(pdist(feats_normalized))
        mst = minimum_spanning_tree(dist_matrix).toarray()
        # NetworkXグラフへの変換
        n_samples = feats.shape[0]
        G = nx.Graph()
        G.add_nodes_from(range(n_samples))
        # エッジ情報の抽出
        edges = []
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                if mst[i, j] > 0:
                    G.add_edge(i, j, weight=mst[i, j])
                    edges.append((i, j, mst[i, j]))
        # 3. エッジを距離（重み）の降順にソート（長いエッジ＝密度の谷 から切断するため）
        edges.sort(key=lambda x: x[2], reverse=True)
        # 4. 切断プロセス
        for u, v, weight in edges:
            components = list(nx.connected_components(G)) # 現在の全クラスター（連結成分）を取得
            # 終了条件: 全てのグループが max_size 以下になったら処理を終える
            if all(len(comp) <= max_size for comp in components):
                break
            comp_containing_edge = next(c for c in components if u in c and v in c) # 対象のエッジ (u, v) が含まれるグループを取得
            # グループの要素数が閾値 (max_size) を超えている場合のみエッジを切断
            if len(comp_containing_edge) > max_size:
                G.remove_edge(u, v)
        # 5. ラベル付け
        final_components = list(nx.connected_components(G))
        labels = np.full(n_samples, -1, dtype=int) # 初期値 -1 (ノイズ)
    
        cluster_id = 0
        for comp in final_components:
            # 規定値 (min_size) 以上のグループのみ正式なクラスターとする
            if len(comp) >= min_size:
                for node in comp:
                    labels[node] = cluster_id
                cluster_id += 1
            
        return labels, final_components

    @staticmethod
    def compute_k_reciprocal_jaccard_distance(feats, k=5):
        n_samples = len(feats) 
        dist_mat = euclidean_distances(feats) 
        # k近傍 (k-NN) のインデックスを取得
        knn_indices = np.argsort(dist_mat, axis=1)[:, :k]
        # 相互k近傍 (k-reciprocal neighbors) のセットを構築
        k_reciprocal_sets = []
        for i in range(n_samples):
            forward_knn = knn_indices[i]
            reciprocal_set = []
            for j in forward_knn:
                if i in knn_indices[j]: # i -> j だけでなく j -> i も成り立っていれば採用
                    reciprocal_set.append(j)
            k_reciprocal_sets.append(set(reciprocal_set))
        
        # Jaccard距離行列の計算
        jaccard_dist_mat = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                if i == j:
                    jaccard_dist_mat[i, j] = 0.0
                    continue
                set_i = k_reciprocal_sets[i]
                set_j = k_reciprocal_sets[j]
                intersection = len(set_i.intersection(set_j)) # 共通の友達の数（積集合）
                union = len(set_i.union(set_j)) # 全員の友達の数（和集合）
                # Jaccard距離 = 1 - Jaccard係数
                if union == 0:
                    jaccard_dist_mat[i, j] = 1.0 # 共通の友達がいない場合は最大距離
                else:
                    jaccard_dist_mat[i, j] = 1.0 - (intersection / union)
                
        return jaccard_dist_mat       

    @classmethod
    def dbscan_by_k_reciprocal_jaccard(cls, img_paths, eps, min_samples=2, k=5):
        cls._init_extractor()
        feats = np.asarray(cls.extractor(img_paths))
        dist_mat = cls.compute_k_reciprocal_jaccard_distance(feats, k=k)
        cluster = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        labels = cluster.fit(dist_mat).labels_
        groups = defaultdict(list)
        for path, label in zip(img_paths, labels):
            groups[label].append(path)
        return groups

    @classmethod
    def find_top_n_similar_pairs(cls, img_paths, top_n=-1, similarity_threshold=0):
        cls._init_extractor()
        # 特徴抽出と正規化
        feats = np.asarray(cls.extractor(img_paths))
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        feats_normalized = feats / norms
        # 全画像間の類似度行列 (コサイン類似度) を計算
        sim_matrix = np.dot(feats_normalized, feats_normalized.T)
        # 重複（i, j と j, i）と自己類似（i, i）を避けるため、上三角行列のインデックスを取得
        # k=1 にすることで対角成分（自身との類似度）を除外
        n_samples = len(img_paths)
        triu_indices = np.triu_indices(n_samples, k=1)
        # 上三角成分の類似度を取得
        flat_similarities = sim_matrix[triu_indices]
        
        if top_n == -1:
            # 閾値以上のインデックスのみ抽出
            candidate_indices = np.where(flat_similarities >= similarity_threshold)[0]
            # 類似度の降順でソート
            sorted_indices = candidate_indices[np.argsort(flat_similarities[candidate_indices])[::-1]]
        else:
            # 指定された top_n が総ペア数を超えないように調整
            actual_top_n = min(top_n, len(flat_similarities))
            # 類似度が高い順（降順）にソートしたインデックスを取得
            sorted_indices = np.argsort(flat_similarities)[::-1][:actual_top_n]
        
        # 対応する画像パスのペアと類似度を返却
        top_pairs = []
        for idx in sorted_indices:
            i = triu_indices[0][idx]
            j = triu_indices[1][idx]
            top_pairs.append((img_paths[i], img_paths[j], float(flat_similarities[idx])))
            
        return top_pairs

    @classmethod
    def find_top_n_similar_pairs_to_query(cls, gallery_img_paths, query_img_path, top_n=-1, similarity_threshold=0):
        cls._init_extractor()
        # extract features of gallery and query images 
        feats_g = np.asarray(cls.extractor(gallery_img_paths))
        feats_q = np.asarray(cls.extractor([query_img_path]))
        # normalize features
        norms_g = np.linalg.norm(feats_g, axis=1, keepdims=True)
        norms_g = np.maximum(norms_g, 1e-12)
        feats_g_norm = feats_g / norms_g
        norms_q = np.linalg.norm(feats_q, axis=1, keepdims=True)
        norms_q = np.maximum(norms_q, 1e-12)
        feats_q_norm = feats_q / norms_q
        # calculate cos similarity. the shape is (N,).  
        sim_values = np.dot(feats_g_norm, feats_q_norm.T).flatten()
        sorted_indices = np.argsort(sim_values)[::-1]
        sim_results = []
        for idx in sorted_indices:
            score = float(sim_values[idx])
            if score < similarity_threshold: break
            sim_results.append((gallery_img_paths[idx], score))
            if top_n != -1 and len(sim_results) == top_n: break
        return sim_results