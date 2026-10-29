from pathlib import Path
import sys
import img_utils as iu
from person_re_identification.osnet import OSNet

if len(sys.argv) < 2:
    print("Usage: python scripts/find_top_n_similar_pair.py <folder_path> [top_n] [similarity_threshold]")
    sys.exit(1)

folder_path = sys.argv[1]
top_n = int(sys.argv[2]) if len(sys.argv) > 2 else -1
similarity_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0

img_paths = iu.glob_img_paths(folder_path)

if len(img_paths) < 2:
    print(f"Error: At least 2 images are required to find pairs. Found {len(img_paths)} in '{folder_path}'.")
    sys.exit(1)

similar_pairs = OSNet.find_top_n_similar_pairs(
    img_paths, top_n=top_n, similarity_threshold=similarity_threshold
)

print(f"Similar pairs ({len(similar_pairs)} found):")
for path1, path2, score in similar_pairs:
    print(f"{Path(path1).name} <-> {Path(path2).name}: {score:.4f}")
