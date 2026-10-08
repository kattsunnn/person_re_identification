import sys
import img_utils as iu
from person_re_identification.osnet import OSNet

if len(sys.argv) < 3:
    print("Usage: python scripts/test.py <gallery_folder_path> <query_img_path>")
    sys.exit(1)

gallery_folder_path = sys.argv[1]
query_img_path = sys.argv[2]

gallery_img_paths = iu.glob_img_paths(gallery_folder_path)

matching_paths = OSNet.find_top_n_similar_pairs_to_query(
    gallery_img_paths, query_img_path
)

print("Matching paths:")
for path, score in matching_paths:
    print(f"{path}: {score:.4f}")
