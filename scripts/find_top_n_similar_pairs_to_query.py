from pathlib import Path
import sys
import img_utils as iu
from person_re_identification.osnet import OSNet

if len(sys.argv) < 3:
    print("Usage: python scripts/find_top_n_similar_pairs_to_query.py <gallery_folder_path> <query_img_path> [top_n]")
    sys.exit(1)

gallery_folder_path = sys.argv[1]
query_img_path = sys.argv[2]
top_n = int(sys.argv[3]) if len(sys.argv) > 3 else -1

gallery_img_paths = iu.glob_img_paths(gallery_folder_path)

matching_paths = OSNet.find_top_n_similar_pairs_to_query(
    gallery_img_paths, query_img_path, top_n=top_n
)

print(f"Matching paths ({len(matching_paths)} found):")
for path, score in matching_paths:
    print(f"{Path(path).name}: {score:.4f}")
