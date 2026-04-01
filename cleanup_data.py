import os
import json
import shutil

# Paths
DATA_ROOT = "data/drivaernet_real"
MESH_DIR = os.path.join(DATA_ROOT, "meshes")
SPLIT_PATH = os.path.join(DATA_ROOT, "split.json")
PROCESSED_DIR = os.path.join(DATA_ROOT, "processed")

def cleanup_to_300():
    if not os.path.exists(SPLIT_PATH):
        print(f"Error: {SPLIT_PATH} not found.")
        return

    with open(SPLIT_PATH, 'r') as f:
        split = json.load(f)

    # Flatten all IDs
    all_ids = split['train'] + split['val'] + split['test']
    total_current = len(all_ids)
    print(f"Current total designs: {total_current}")

    if total_current <= 300:
        print("Dataset already at or below 300 designs.")
        return

    # Keep 300, delete the rest
    to_keep = set(all_ids[:300])
    to_delete = set(all_ids[300:])
    
    print(f"Keeping 300 designs, deleting {len(to_delete)} designs.")

    # 1. Delete .vtp files
    count_vtp = 0
    for design_id in to_delete:
        vtp_path = os.path.join(MESH_DIR, f"{design_id}.vtp")
        if os.path.exists(vtp_path):
            os.remove(vtp_path)
            count_vtp += 1
    print(f"Deleted {count_vtp} .vtp files from {MESH_DIR}")

    # 2. Update split.json
    new_train_n = int(0.7 * 300)
    new_val_n = int(0.15 * 300)
    # new_test_n = 300 - new_train_n - new_val_n
    
    kept_list = all_ids[:300]
    new_split = {
        "train": kept_list[:new_train_n],
        "val": kept_list[new_train_n : new_train_n + new_val_n],
        "test": kept_list[new_train_n + new_val_n :]
    }
    
    with open(SPLIT_PATH, 'w') as f:
        json.dump(new_split, f, indent=2)
    print(f"Updated {SPLIT_PATH} with 300 designs.")

    # 3. Clear processed directory (safer than trying to map cached .pt files)
    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)
        os.makedirs(PROCESSED_DIR)
        print(f"Cleared {PROCESSED_DIR} to force re-processing.")

if __name__ == "__main__":
    cleanup_to_300()
