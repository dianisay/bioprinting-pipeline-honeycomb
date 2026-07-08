"""Fix GPU compatibility in ALL notebooks (basic + extended).
The P100 (sm_60) is incompatible with PyTorch 2.10+cu128.
Must use T4 x2 (sm_75) or newer.
"""
import json
from pathlib import Path

ALL_NOTEBOOKS = [
    'notebooks/01-ablation-study-polar-kaggle.ipynb',
    'notebooks/01-ablation-study-detr-kaggle.ipynb',
    'notebooks/01-ablation-study-autoregressive-kaggle.ipynb',
    'notebooks/01-ablation-study-polar-kaggle-extended.ipynb',
    'notebooks/01-ablation-study-detr-kaggle-extended.ipynb',
    'notebooks/01-ablation-study-autoregressive-kaggle-extended.ipynb',
]

# Pattern: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE_PATTERNS = [
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
    'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")',
]

GPU_COMPAT_CODE = """assert torch.cuda.is_available(), (
    'FATAL: No GPU detected! Select T4 x2 in Kaggle Settings > Accelerator.'
)
_cc = torch.cuda.get_device_capability(0)
assert _cc[0] * 10 + _cc[1] >= 70, (
    f'GPU {torch.cuda.get_device_name(0)} (sm_{_cc[0]}{_cc[1]}) is too old. '
    f'PyTorch 2.10+cu128 needs sm_70+. Select T4 x2 in Kaggle Settings.'
)
device = torch.device('cuda')
print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB')"""


def patch_notebook(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    patched = 0
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']

        for pat in DEVICE_PATTERNS:
            if pat in src:
                src = src.replace(pat, GPU_COMPAT_CODE)
                cell['source'] = src
                patched += 1
                print(f'  [PATCHED] GPU compat check')
                break

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    return patched


if __name__ == '__main__':
    for nb_rel in ALL_NOTEBOOKS:
        nb_path = Path(nb_rel)
        if not nb_path.exists():
            print(f'SKIP: {nb_rel}')
            continue
        print(f'\n{nb_rel}')
        n = patch_notebook(nb_path)
        if n == 0:
            print('  (already patched or pattern not found)')
        else:
            print(f'  Patches: {n}')
    print('\nDone.')
