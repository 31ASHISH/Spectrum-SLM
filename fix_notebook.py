import json, os

with open('spectrum_slm_kaggle.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

lines = [
    "# Cell 4 - Configure Kaggle paths (AUTO-DETECTS dataset location)",
    "import os",
    "",
    "def find_dataset_root(base='/kaggle/input'):",
    "    if not os.path.isdir(base):",
    "        return None",
    "    for ds in os.listdir(base):",
    "        ds_path = os.path.join(base, ds)",
    "        if not os.path.isdir(ds_path):",
    "            continue",
    "        # Direct match",
    "        if os.path.isdir(os.path.join(ds_path, 'Secondary_User')):",
    "            return ds_path",
    "        # One level deeper (extra folder from zip)",
    "        for sub in os.listdir(ds_path):",
    "            sub_path = os.path.join(ds_path, sub)",
    "            if os.path.isdir(sub_path) and os.path.isdir(os.path.join(sub_path, 'Secondary_User')):",
    "                return sub_path",
    "    return None",
    "",
    "print('Scanning /kaggle/input/ ...')",
    "if os.path.isdir('/kaggle/input'):",
    "    for d in os.listdir('/kaggle/input'):",
    "        print(f'  /kaggle/input/{d}/')",
    "        for sub in os.listdir(f'/kaggle/input/{d}')[:8]:",
    "            print(f'    {sub}/')",
    "",
    "DATA_ROOT = find_dataset_root()",
    "",
    "if DATA_ROOT is None:",
    "    print('ERROR: Could not find Secondary_User/ !')",
    "    print('Add your dataset via Add Input on the right panel.')",
    "else:",
    "    print(f'Dataset root found: {DATA_ROOT}')",
    "",
    "SU_DIR  = f'{DATA_ROOT}/Secondary_User'",
    "NEW_DIR = f'{DATA_ROOT}/files-20260414T094743Z-3-001'",
    "OUT_DIR = '/kaggle/working/checkpoints'",
    "os.makedirs(f'{OUT_DIR}/phase1', exist_ok=True)",
    "os.makedirs(f'{OUT_DIR}/phase2', exist_ok=True)",
    "",
    "print(f'SU_DIR  : {SU_DIR}')",
    "print(f'  exists = {os.path.isdir(SU_DIR)}')",
    "print(f'NEW_DIR : {NEW_DIR}')",
    "print(f'  exists = {os.path.isdir(NEW_DIR)}')",
]

nb['cells'][4]['source'] = "\n".join(lines)

with open('spectrum_slm_kaggle.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)

# Validate
with open('spectrum_slm_kaggle.ipynb', 'r', encoding='utf-8') as f:
    loaded = json.load(f)
print(f'Notebook updated. Cells: {len(loaded["cells"])}')
sz = os.path.getsize('spectrum_slm_kaggle.ipynb')
print(f'File size: {sz/1000:.1f} KB')
print('JSON: VALID')
