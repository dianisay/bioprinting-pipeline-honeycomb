"""Add a hard GPU check that stops the notebook if no GPU is detected."""
import json
from pathlib import Path

NOTEBOOKS = [
    'notebooks/01-ablation-study-polar-kaggle-extended.ipynb',
    'notebooks/01-ablation-study-detr-kaggle-extended.ipynb',
    'notebooks/01-ablation-study-autoregressive-kaggle-extended.ipynb',
]

GPU_CHECK_CODE = '''# ═══════════════════════════════════════════════════════════════════════════════
# CRITICAL: GPU VERIFICATION — Do NOT proceed on CPU
# ═══════════════════════════════════════════════════════════════════════════════
import torch
assert torch.cuda.is_available(), """
╔═══════════════════════════════════════════════════════════════════╗
║  FATAL: No GPU detected! This notebook REQUIRES a GPU.          ║
║                                                                   ║
║  In Kaggle: Settings → Accelerator → GPU T4 x2                  ║
║  DO NOT run on CPU — training will timeout after 12 hours.       ║
╚═══════════════════════════════════════════════════════════════════╝
"""
device = torch.device('cuda')
print(f'GPU verified: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
'''

# Find the existing device setup and replace it
OLD_DEVICE_PATTERNS = [
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
    'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")',
]


def patch_notebook(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    patched = 0

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']

        for pattern in OLD_DEVICE_PATTERNS:
            if pattern in src:
                new_src = src.replace(
                    pattern,
                    "assert torch.cuda.is_available(), 'FATAL: No GPU! Set Accelerator to T4 x2 in Kaggle Settings.'\ndevice = torch.device('cuda')"
                )
                cell['source'] = new_src
                patched += 1
                print(f'  [PATCHED] Hard GPU assertion in cell {i}')
                break

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
        print(f'  Patches: {n}')
    print('\nDone.')
