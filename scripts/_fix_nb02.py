"""Fix 02_volumetric notebook: GPU compat + dianisay paths."""
import json
from pathlib import Path

NB = 'notebooks/02_volumetric_ablation_kaggle.ipynb'

with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

patched = 0
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']

    # Fix GPU device
    old = "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"
    new = """assert torch.cuda.is_available(), (
    'FATAL: No GPU! Select T4 x2 in Kaggle Settings > Accelerator.'
)
_cc = torch.cuda.get_device_capability(0)
assert _cc[0] * 10 + _cc[1] >= 70, (
    f'GPU {torch.cuda.get_device_name(0)} (sm_{_cc[0]}{_cc[1]}) too old. Select T4 x2.'
)
device = torch.device('cuda')"""
    if old in src:
        src = src.replace(old, new)
        patched += 1
        print('[PATCHED] GPU compat check')

    # Add dianisay paths to REPO_PATHS
    old_paths = """REPO_PATHS = [
    '..',
    '.',
    *glob.glob('/kaggle/input/bioprinting-pipeline-honeycomb*'),
    *glob.glob('/kaggle/input/diana-bioprinting*'),
    *glob.glob('/kaggle/input/*/'),
]"""
    new_paths = """REPO_PATHS = [
    '..',
    '.',
    *glob.glob('/kaggle/input/datasets/dianisay/bioprinting-pipeline-honeycomb*'),
    *glob.glob('/kaggle/input/datasets/dianisay/diana-bioprinting*'),
    *glob.glob('/kaggle/input/bioprinting-pipeline-honeycomb*'),
    *glob.glob('/kaggle/input/diana-bioprinting*'),
    *glob.glob('/kaggle/input/datasets/dianisay/*/'),
    *glob.glob('/kaggle/input/*/'),
]"""
    if old_paths in src:
        src = src.replace(old_paths, new_paths)
        patched += 1
        print('[PATCHED] REPO_PATHS with dianisay')

    cell['source'] = src

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'\nTotal patches: {patched}')
