import pandas as pd
import json
import os
import shutil
import sys

# Force UTF-8 for windows terminal output
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def normalize_politifact(input_file, output_dir):
    """Normalizes PolitiFact data to the master schema."""
    if not os.path.exists(input_file):
        print(f"Warning: {input_file} not found.")
        return []
    
    data = pd.read_json(input_file, lines=True)
    normalized = []
    
    # Label Map for PolitiFact
    label_map = {
        'pants-fire': 1, 'false': 1, 'mostly-false': 1, 'barely-true': 1,
        'half-true': 0, 'mostly-true': 0, 'true': 0,
        'unknown': -1
    }
    
    for _, row in data.iterrows():
        label = label_map.get(row.get('verdict', 'unknown'), -1)
        if label == -1: continue 
        
        raw_path = row.get('local_img_path') or row.get('local_image_path')
        new_path = None
        
        if raw_path and os.path.exists(raw_path):
            filename = os.path.basename(raw_path)
            new_path = os.path.join(output_dir, filename)
            shutil.copy2(raw_path, new_path)
            # Store relative path for the dataset
            new_path = f"data/images/{filename}"

        normalized.append({
            "id": row.get('id', row.get('post_id')),
            "text": row.get('text'),
            "image_path": new_path,
            "label": label,
            "domain": 0, 
            "source": "politifact",
            "raw_verdict": row.get('verdict')
        })
    return normalized

def normalize_factcheck(input_file):
    """Normalizes FactCheck.org data to the master schema."""
    if not os.path.exists(input_file):
        print(f"Warning: {input_file} not found.")
        return []
        
    data = pd.read_json(input_file, lines=True)
    normalized = []
    
    for _, row in data.iterrows():
        verdict = row.get('verdict', 'unknown')
        label = 1 if 'false' in str(verdict).lower() or 'misleading' in str(verdict).lower() else 0
        
        raw_path = row.get('local_image_path')
        new_path = None
        if isinstance(raw_path, str) and os.path.exists(raw_path):
            filename = os.path.basename(raw_path)
            new_path = os.path.join('data/images', filename)
            shutil.copy2(raw_path, new_path)
            new_path = f"data/images/{filename}"

        normalized.append({
            "id": row.get('post_id'),
            "text": row.get('text') or row.get('title'),
            "image_path": new_path,
            "label": label,
            "domain": 1, 
            "source": "factcheck",
            "raw_verdict": verdict
        })
    return normalized

def main():
    master_data = []
    
    # Process existing files
    print("Normalizing PolitiFact...")
    master_data.extend(normalize_politifact('dataset/political_data.jsonl', 'data/images'))
    
    print("Normalizing FactCheck...")
    master_data.extend(normalize_factcheck('dataset/output.jsonl'))
    
    # Save unified data
    output_file = 'data/master_data.jsonl'
    with open(output_file, 'w') as f:
        for entry in master_data:
            json.dump(entry, f)
            f.write('\n')
            
    print(f"✅ Success! Master dataset created with {len(master_data)} entries at {output_file}")

if __name__ == "__main__":
    main()
