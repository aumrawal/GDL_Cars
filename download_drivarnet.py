import requests
import os
import re
import json
from collections import defaultdict

def download_drivaernet_pipeline_data():
    # 1. Pipeline Directory Setup (Based on f1_base.yaml)
    data_root = "./DrivAerNet++"
    mesh_dir = os.path.join(data_root, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)
    
    server = 'https://dataverse.harvard.edu'
    persistent_id = 'doi:10.7910/DVN/PCZYL4'
    metadata_url = f'{server}/api/datasets/:persistentId/?persistentId={persistent_id}'
    
    print("Fetching dataset metadata from Harvard Dataverse...")
    response = requests.get(metadata_url)
    
    if response.status_code != 200:
        print(f"Failed to fetch metadata. Status code: {response.status_code}")
        return

    files = response.json().get('data', {}).get('latestVersion', {}).get('files', [])
    if not files:
        print("No files found in the dataset metadata.")
        return

    # 2. Group Files by Design ID
    designs = defaultdict(dict)
    
    for f in files:
        file_info = f['dataFile']
        filename = file_info['filename']
        
        # Extract a design ID (assuming a numeric ID of 3+ digits in the filename)
        match = re.search(r'(\d{3,})', filename)
        if not match:
            continue
            
        design_id = match.group(1)
        lower_name = filename.lower()
        
        # Categorize the file type
        if 'mesh' in lower_name or '.vtp' in lower_name or '.stl' in lower_name:
            designs[design_id]['mesh'] = file_info
        elif 'wss' in lower_name or 'shear' in lower_name:
            designs[design_id]['wss'] = file_info
        elif 'press' in lower_name or 'cp' in lower_name:
            designs[design_id]['pressure'] = file_info
            
    # 3. Filter and select exactly 300 designs
    # (If DrivAerNet++ provides all fields unified inside a single .vtp file, 
    # it will just map to the 'mesh' key. The script handles both single and separate files).
    target_design_ids = sorted(list(designs.keys()))[:300]
    print(f"Found and selected {len(target_design_ids)} unique designs.\n")

    # 4. Download Process
    for i, d_id in enumerate(target_design_ids):
        print(f"[{i+1}/300] Processing Design ID: {d_id}")
        
        for file_type, file_info in designs[d_id].items():
            file_id = file_info['id']
            file_name = file_info['filename']
            download_url = f'{server}/api/access/datafile/{file_id}'
            
            # Save to the /meshes folder expected by f1_base.yaml
            file_path = os.path.join(mesh_dir, file_name)
            
            if os.path.exists(file_path):
                print(f"  -> {file_name} already exists, skipping.")
                continue
                
            print(f"  -> Downloading {file_type}: {file_name}...")
            
            file_resp = requests.get(download_url, stream=True)
            if file_resp.status_code == 200:
                with open(file_path, 'wb') as f_out:
                    for chunk in file_resp.iter_content(chunk_size=8192):
                        f_out.write(chunk)
            else:
                print(f"  -> Failed to download {file_name} (Status: {file_resp.status_code})")

    # 5. Generate split.json for the Pipeline
    print("\nGenerating split.json based on f1_base.yaml...")
    total_designs = len(target_design_ids)
    
    train_end = int(0.7 * total_designs)
    val_end = train_end + int(0.15 * total_designs)
    
    split_dict = {
        "train": target_design_ids[:train_end],
        "val": target_design_ids[train_end:val_end],
        "test": target_design_ids[val_end:]
    }
    
    split_path = os.path.join(data_root, "split.json")
    with open(split_path, "w") as f_json:
        json.dump(split_dict, f_json, indent=4)
        
    print(f"Pipeline ready! Data and split.json saved in '{data_root}'.")

if __name__ == "__main__":
    download_drivaernet_pipeline_data()