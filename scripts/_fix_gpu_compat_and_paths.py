"""Fix two critical issues in extended notebooks:
1. P100 GPU is incompatible with PyTorch 2.10+cu128 — add compatibility check
2. Kaggle dataset paths need /datasets/dianisay/ prefix
"""
import json
from pathlib import Path

NOTEBOOKS = [
    'notebooks/01-ablation-study-polar-kaggle-extended.ipynb',
    'notebooks/01-ablation-study-detr-kaggle-extended.ipynb',
    'notebooks/01-ablation-study-autoregressive-kaggle-extended.ipynb',
]

# Fix 1: Replace the simple GPU assertion with a compute-capability check
OLD_GPU_ASSERT = "assert torch.cuda.is_available(), 'FATAL: No GPU! Set Accelerator to T4 x2 in Kaggle Settings.'\ndevice = torch.device('cuda')"

NEW_GPU_CHECK = """assert torch.cuda.is_available(), 'FATAL: No GPU! Set Accelerator to T4 x2 in Kaggle Settings.'
_cc = torch.cuda.get_device_capability(0)
_cc_num = _cc[0] * 10 + _cc[1]
assert _cc_num >= 70, (
    f'GPU compute capability {_cc[0]}.{_cc[1]} is too old for this PyTorch build. '
    f'Detected: {torch.cuda.get_device_name(0)}. '
    f'P100 (sm_60) is NOT supported by PyTorch 2.10+cu128. '
    f'Please select T4 x2 (sm_75) in Kaggle Settings > Accelerator.'
)
device = torch.device('cuda')
print(f'GPU OK: {torch.cuda.get_device_name(0)} (sm_{_cc_num})')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')"""

# Fix 2: Update BASE_PATH to handle the /datasets/dianisay/ structure
OLD_BASE_PATH = "BASE_PATH = Path('/kaggle/input/wound-datasets/Datasets')"

NEW_BASE_PATH = """# Kaggle mounts datasets at /kaggle/input/datasets/<username>/<dataset-slug>/
# Try both the new API path and legacy path
_candidates = [
    Path('/kaggle/input/datasets/dianisay/wound-datasets/Datasets'),
    Path('/kaggle/input/wound-datasets/Datasets'),
    Path('/kaggle/input/datasets/dianisay/wound-datasets'),
    Path('/kaggle/input/wound-datasets'),
]
BASE_PATH = next((p for p in _candidates if p.exists()), _candidates[0])
print(f'BASE_PATH: {BASE_PATH} (exists={BASE_PATH.exists()})')"""

# Fix 3: Update pretrained checkpoint search paths
OLD_PRETRAINED_SEARCH = """'/kaggle/input/ablation-basic-{vname}/results/ablation/{vname}/best.pth',
        '/kaggle/input/ablation-basic-{vname}/best.pth',"""

NEW_PRETRAINED_SEARCH = """'/kaggle/input/ablation-basic-{vname}/results/ablation/{vname}/best.pth',
        '/kaggle/input/ablation-basic-{vname}/best.pth',
        '/kaggle/input/datasets/dianisay/ablation-basic-{vname}/results/ablation/{vname}/best.pth',
        '/kaggle/input/datasets/dianisay/ablation-basic-{vname}/best.pth',"""


def patch_notebook(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    patched = 0
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']

        # Fix GPU assertion
        if OLD_GPU_ASSERT in src:
            src = src.replace(OLD_GPU_ASSERT, NEW_GPU_CHECK)
            patched += 1
            print(f'  [PATCHED] GPU compute capability check')

        # Fix BASE_PATH
        if OLD_BASE_PATH in src:
            src = src.replace(OLD_BASE_PATH, NEW_BASE_PATH)
            patched += 1
            print(f'  [PATCHED] BASE_PATH with dianisay fallback')

        # Fix pretrained search paths (handle both vname formats)
        for vname in ['polar', 'detr', 'autoregressive']:
            old_p = OLD_PRETRAINED_SEARCH.replace('{vname}', vname)
            new_p = NEW_PRETRAINED_SEARCH.replace('{vname}', vname)
            if old_p in src:
                src = src.replace(old_p, new_p)
                patched += 1
                print(f'  [PATCHED] Pretrained paths for {vname}')

        cell['source'] = src

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    return patched


if __name__ == '__main__':
    for nb_rel in NOTEBOOKS:
        nb_path = Path(nb_rel)
        if not nb_path.exists():
            print(f'SKIP: {nb_rel}')
            continue
        print(f'\nPatching: {nb_rel}')
        n = patch_notebook(nb_path)
        print(f'  Total: {n}')
    print('\nDone.')
