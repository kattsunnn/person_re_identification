import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def visualize_reid_features_tsne(features: np.ndarray, person_ids: np.ndarray, n_dims: int = 2):
    """
    Re-IDの高次元特徴量をPCA + t-SNEで圧縮し、2Dまたは3Dでプロットする関数
    
    Parameters:
    - features: shape (N, 512) の特徴ベクトル行列
    - person_ids: shape (N,) 各画像の人物ID（色分け用）
    - n_dims: 2 または 3 （プロットする次元数）
    """
    print(f"--- 1. 前処理・データ確認 ---")
    print(f"入力データ形状: サンプル数 = {features.shape[0]}, 次元数 = {features.shape[1]}")
    
    # --- ステップ2: PCAによる中間次元削減 (512D -> 50D) ---
    # ノイズを除去し、t-SNEの計算を高速化・安定化する
    print(f"--- 2. PCAによる次元削減 (512D -> 50D) ---")
    pca = PCA(n_components=50, random_state=42)
    features_pca = pca.fit_transform(features)
    
    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"PCA累積寄与率 (上位50次元): {explained_var:.2%}")

    # --- ステップ3 & 5: t-SNEによる非線形マッピング (50D -> 2D or 3D) ---
    print(f"--- 3. t-SNEによる {n_dims}次元へのマッピング実行中... ---")
    tsne = TSNE(
        n_components=n_dims,
        perplexity=30,      # 近傍の考慮サイズ（データ数に応じて10〜50で調整）
        learning_rate='auto',
        init='pca',         # 初期配置にPCA結果を使い、局所解へのハマりを防ぐ
        random_state=42,
        n_iter=1000
    )
    features_projected = tsne.fit_transform(features_pca)
    print("t-SNE投影完了！")

    # --- ステップ4 & 5: グラフの描画 ---
    fig = plt.figure(figsize=(10, 8))
    
    if n_dims == 2:
        # 2次元プロットの場合
        ax = fig.add_subplot(111)
        scatter = ax.scatter(
            features_projected[:, 0], features_projected[:, 1],
            c=person_ids, cmap='tab10', alpha=0.7, s=30
        )
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.set_title("Re-ID Feature Space Visualization (2D t-SNE)")
        
    elif n_dims == 3:
        # 3次元プロットの場合
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            features_projected[:, 0], features_projected[:, 1], features_projected[:, 2],
            c=person_ids, cmap='tab10', alpha=0.7, s=30
        )
        ax.set_xlabel("t-SNE Dim 1")
        ax.set_ylabel("t-SNE Dim 2")
        ax.set_zlabel("t-SNE Dim 3")
        ax.set_title("Re-ID Feature Space Visualization (3D t-SNE)")
    
    else:
        raise ValueError("n_dims は 2 または 3 を指定してください。")

    # カラーバーの追加
    cbar = plt.colorbar(scatter)
    cbar.set_label("Person ID")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

# ==========================================
# 実行テスト用コード（ダミーデータでの動作確認）
# ==========================================
if __name__ == "__main__":
    # 例: 1000サンプル、512次元のRe-ID特徴量ベクトルを模したダミーデータ
    np.random.seed(42)
    dummy_features = np.random.randn(1000, 512)
    
    # 例: 10人分の人物ID (0 〜 9)
    dummy_person_ids = np.random.randint(0, 10, size=1000)

    # 2次元で可視化を実行してみる
    visualize_reid_features_tsne(dummy_features, dummy_person_ids, n_dims=2)