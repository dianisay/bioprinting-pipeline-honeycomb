"""Add recursion guard to WoundDataset.__getitem__ and data quality diagnostic."""
import json
from pathlib import Path

NOTEBOOKS = [
    'notebooks/01-ablation-study-polar-kaggle-extended.ipynb',
    'notebooks/01-ablation-study-detr-kaggle-extended.ipynb',
    'notebooks/01-ablation-study-autoregressive-kaggle-extended.ipynb',
]

# Fix the recursion issue — add depth parameter
OLD_GETITEM_SIG = '''    def __getitem__(self, idx):
        img_path, mask_path, precomputed_label = self.samples[idx]'''

NEW_GETITEM_SIG = '''    def __getitem__(self, idx, _depth=0):
        if _depth > 10:
            # Exhausted retries — return a synthetic-like fallback
            r = 0.15
            angles = np.linspace(0, 2*np.pi, self.num_radii, endpoint=False)
            dummy_img = torch.zeros(3, self.image_size, self.image_size)
            return {
                'image': dummy_img,
                'centroid': torch.tensor([0.5, 0.5]),
                'radii': torch.tensor(np.full(self.num_radii, r, dtype=np.float32)),
                'points': torch.tensor(np.stack([0.5+r*np.cos(angles), 0.5+r*np.sin(angles)], 1).astype(np.float32)),
            }
        img_path, mask_path, precomputed_label = self.samples[idx]'''

# Fix the skip calls to pass depth
OLD_SKIP_MASK = '''            if label is None:
                # Bad mask — try a different sample instead of poisoning with fake data
                alt_idx = (idx + 1) % len(self.samples)
                return self.__getitem__(alt_idx)'''

NEW_SKIP_MASK = '''            if label is None:
                alt_idx = (idx + 1) % len(self.samples)
                return self.__getitem__(alt_idx, _depth + 1)'''

OLD_SKIP_NOMASK = '''        else:
            # No mask available — skip to next sample
            alt_idx = (idx + 1) % len(self.samples)
            return self.__getitem__(alt_idx)'''

NEW_SKIP_NOMASK = '''        else:
            alt_idx = (idx + 1) % len(self.samples)
            return self.__getitem__(alt_idx, _depth + 1)'''

# Data quality diagnostic to add after dataset building
DATA_QUALITY_CELL = '''# ── Data Quality Check ──────────────────────────────────────────────────────
# Verify how many real masks produce valid polar labels
valid_count = 0
invalid_count = 0
mask_stats = []

for img_path, mask_path, precomputed in train_ds.samples[:min(200, len(train_ds.samples))]:
    if mask_path is not None:
        mask_pil = Image.open(mask_path).convert('L').resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
        mask = np.array(mask_pil)
        mask_stats.append({'max': mask.max(), 'min': mask.min(), 'mean': mask.mean()})
        lbl = mask_to_polar(mask, NUM_RADII, IMAGE_SIZE)
        if lbl is not None:
            valid_count += 1
        else:
            invalid_count += 1

print(f'Data Quality Report (first {min(200, len(train_ds.samples))} samples):')
print(f'  Valid polar labels : {valid_count}')
print(f'  Invalid (skipped)  : {invalid_count}')
print(f'  Precomputed (synth): {sum(1 for _,_,p in train_ds.samples if p is not None)}')
if mask_stats:
    maxes = [s['max'] for s in mask_stats]
    print(f'  Mask value range   : [{min(s["min"] for s in mask_stats)}, {max(maxes)}]')
    print(f'  Masks with max<=1  : {sum(1 for m in maxes if m <= 1)} (need normalization)')
    print(f'  Masks with max>127 : {sum(1 for m in maxes if m > 127)} (standard format)')
print(f'  Total train samples: {len(train_ds)}')
'''


def patch_notebook(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    patched = 0
    build_cell_idx = None

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']

        # Fix __getitem__ signature
        if OLD_GETITEM_SIG in src:
            src = src.replace(OLD_GETITEM_SIG, NEW_GETITEM_SIG)
            patched += 1
            print(f'  [PATCHED] __getitem__ depth guard')

        # Fix skip calls
        if OLD_SKIP_MASK in src:
            src = src.replace(OLD_SKIP_MASK, NEW_SKIP_MASK)
            patched += 1
            print(f'  [PATCHED] mask skip with depth')

        if OLD_SKIP_NOMASK in src:
            src = src.replace(OLD_SKIP_NOMASK, NEW_SKIP_NOMASK)
            patched += 1
            print(f'  [PATCHED] no-mask skip with depth')

        cell['source'] = src

        # Find the "Build dataset" cell
        if 'Build dataset' in src or ('train_ds = WoundDataset' in src and 'USE_REAL_DATA' in src):
            build_cell_idx = i

    # Insert data quality cell after build dataset
    if build_cell_idx is not None:
        quality_cell = {
            'cell_type': 'code',
            'metadata': {'trusted': True},
            'source': DATA_QUALITY_CELL,
            'execution_count': None,
            'outputs': []
        }
        nb['cells'].insert(build_cell_idx + 1, quality_cell)
        patched += 1
        print(f'  [INSERTED] Data quality diagnostic cell at index {build_cell_idx + 1}')

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    return patched


if __name__ == '__main__':
    for nb_rel in NOTEBOOKS:
        nb_path = Path(nb_rel)
        if not nb_path.exists():
            print(f'SKIP (not found): {nb_rel}')
            continue
        print(f'\nPatching: {nb_rel}')
        n = patch_notebook(nb_path)
        print(f'  Total patches: {n}')
    print('\nDone.')
