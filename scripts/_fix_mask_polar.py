"""Fix mask_to_polar and WoundDataset in all extended notebooks.

Problems fixed:
1. mask_to_polar fails for masks with {0,1} values instead of {0,255}
2. Ray-casting can produce zero radii for concave/fragmented masks
3. Fallback circle pollutes training data — now skips bad samples
4. Added minimum radius validation
"""
import json
from pathlib import Path

NOTEBOOKS = [
    'notebooks/01-ablation-study-polar-kaggle-extended.ipynb',
    'notebooks/01-ablation-study-detr-kaggle-extended.ipynb',
    'notebooks/01-ablation-study-autoregressive-kaggle-extended.ipynb',
]

OLD_MASK_TO_POLAR = '''# ─────────────────────────────────────────────────────────────────────────────
# MASK TO POLAR CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

def mask_to_polar(mask, num_radii=NUM_RADII, image_size=IMAGE_SIZE):
    \"\"\"Convert binary mask to polar representation.\"\"\"
    ys, xs = np.where(mask > 127)
    if len(xs) < 50:
        return None
    cx = float(xs.mean()) / mask.shape[1]
    cy = float(ys.mean()) / mask.shape[0]
    cxp = cx * mask.shape[1]
    cyp = cy * mask.shape[0]

    angles = np.linspace(0, 2*np.pi, num_radii, endpoint=False)
    radii  = np.zeros(num_radii, dtype=np.float32)

    max_r = max(mask.shape) * 0.6
    for i, ang in enumerate(angles):
        dx, dy = np.cos(ang), np.sin(ang)
        for step in range(1, int(max_r)):
            px = int(cxp + dx * step)
            py = int(cyp + dy * step)
            if px < 0 or px >= mask.shape[1] or py < 0 or py >= mask.shape[0]:
                radii[i] = step / max(mask.shape)
                break
            if mask[py, px] < 127:
                radii[i] = step / max(mask.shape)
                break
        else:
            radii[i] = max_r / max(mask.shape)

    pts_x = np.clip(cx + radii * np.cos(angles) * max(mask.shape) / mask.shape[1], 0, 1)
    pts_y = np.clip(cy + radii * np.sin(angles) * max(mask.shape) / mask.shape[0], 0, 1)

    return {
        'centroid': np.array([cx, cy], dtype=np.float32),
        'radii'   : radii,
        'points'  : np.stack([pts_x, pts_y], axis=1).astype(np.float32),
    }

print('mask_to_polar defined.')'''

NEW_MASK_TO_POLAR = '''# ─────────────────────────────────────────────────────────────────────────────
# MASK TO POLAR CONVERSION (robust version)
# Handles {0,1} and {0,255} mask conventions, validates output quality
# ─────────────────────────────────────────────────────────────────────────────

def mask_to_polar(mask, num_radii=NUM_RADII, image_size=IMAGE_SIZE):
    \"\"\"Convert binary mask to polar representation.
    
    Handles multiple mask formats: {0,1}, {0,255}, grayscale.
    Returns None if mask is too small or produces invalid polar labels.
    \"\"\"
    # Normalize mask to binary {0, 255} regardless of input format
    if mask.max() <= 1:
        mask_bin = (mask > 0).astype(np.uint8) * 255
    else:
        mask_bin = (mask > 127).astype(np.uint8) * 255

    ys, xs = np.where(mask_bin > 0)
    if len(xs) < 50:
        return None

    cx = float(xs.mean()) / mask_bin.shape[1]
    cy = float(ys.mean()) / mask_bin.shape[0]
    cxp = cx * mask_bin.shape[1]
    cyp = cy * mask_bin.shape[0]

    # Verify centroid is inside mask (handle concave shapes)
    cpx_int, cpy_int = int(cxp), int(cyp)
    cpx_int = np.clip(cpx_int, 0, mask_bin.shape[1] - 1)
    cpy_int = np.clip(cpy_int, 0, mask_bin.shape[0] - 1)
    if mask_bin[cpy_int, cpx_int] == 0:
        # Centroid outside mask — use median instead of mean
        cx = float(np.median(xs)) / mask_bin.shape[1]
        cy = float(np.median(ys)) / mask_bin.shape[0]
        cxp = cx * mask_bin.shape[1]
        cyp = cy * mask_bin.shape[0]

    angles = np.linspace(0, 2*np.pi, num_radii, endpoint=False)
    radii  = np.zeros(num_radii, dtype=np.float32)

    max_r = max(mask_bin.shape) * 0.7
    for i, ang in enumerate(angles):
        dx, dy = np.cos(ang), np.sin(ang)
        found = False
        for step in range(1, int(max_r)):
            px = int(cxp + dx * step)
            py = int(cyp + dy * step)
            if px < 0 or px >= mask_bin.shape[1] or py < 0 or py >= mask_bin.shape[0]:
                radii[i] = step / max(mask_bin.shape)
                found = True
                break
            if mask_bin[py, px] == 0:
                radii[i] = step / max(mask_bin.shape)
                found = True
                break
        if not found:
            radii[i] = max_r / max(mask_bin.shape)

    # Validate: reject if too many radii are near-zero (bad mask)
    min_radius = 3.0 / max(mask_bin.shape)  # at least 3 pixels
    valid_radii = radii > min_radius
    if valid_radii.sum() < num_radii * 0.5:
        return None

    # Clamp minimum radius to avoid degenerate points
    radii = np.maximum(radii, min_radius)

    pts_x = np.clip(cx + radii * np.cos(angles), 0, 1)
    pts_y = np.clip(cy + radii * np.sin(angles), 0, 1)

    return {
        'centroid': np.array([cx, cy], dtype=np.float32),
        'radii'   : radii,
        'points'  : np.stack([pts_x, pts_y], axis=1).astype(np.float32),
    }

print('mask_to_polar defined (robust version).')'''

# Also fix the WoundDataset __getitem__ to SKIP bad samples instead of using fake circles
OLD_GETITEM_FALLBACK = '''            label = mask_to_polar(mask, self.num_radii, self.image_size)
            if label is None:
                r = 0.2
                angles = np.linspace(0, 2*np.pi, self.num_radii, endpoint=False)
                label = {
                    'centroid': np.array([0.5, 0.5], dtype=np.float32),
                    'radii'   : np.full(self.num_radii, r, dtype=np.float32),
                    'points'  : np.stack([0.5 + r*np.cos(angles),
                                          0.5 + r*np.sin(angles)], 1).astype(np.float32),
                }'''

NEW_GETITEM_FALLBACK = '''            label = mask_to_polar(mask, self.num_radii, self.image_size)
            if label is None:
                # Bad mask — try a different sample instead of poisoning with fake data
                alt_idx = (idx + 1) % len(self.samples)
                return self.__getitem__(alt_idx)'''

# Also fix the second fallback (no mask_path)
OLD_NOMASK_FALLBACK = '''        else:
            r = 0.2
            angles = np.linspace(0, 2*np.pi, self.num_radii, endpoint=False)
            label = {
                'centroid': np.array([0.5, 0.5], dtype=np.float32),
                'radii'   : np.full(self.num_radii, r, dtype=np.float32),
                'points'  : np.stack([0.5 + r*np.cos(angles),
                                      0.5 + r*np.sin(angles)], 1).astype(np.float32),
            }'''

NEW_NOMASK_FALLBACK = '''        else:
            # No mask available — skip to next sample
            alt_idx = (idx + 1) % len(self.samples)
            return self.__getitem__(alt_idx)'''


def patch_notebook(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    patched = 0
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']

        # Patch mask_to_polar
        if 'def mask_to_polar' in src and 'mask > 127' in src:
            new_src = src.replace(OLD_MASK_TO_POLAR, NEW_MASK_TO_POLAR)
            if new_src != src:
                cell['source'] = new_src
                patched += 1
                print(f'  [PATCHED] mask_to_polar function')

        # Patch __getitem__ fallback
        if 'mask_to_polar(mask' in src and 'r = 0.2' in src:
            new_src = src.replace(OLD_GETITEM_FALLBACK, NEW_GETITEM_FALLBACK)
            if new_src != src:
                src = new_src
            new_src = src.replace(OLD_NOMASK_FALLBACK, NEW_NOMASK_FALLBACK)
            if new_src != src:
                cell['source'] = new_src
                patched += 1
                print(f'  [PATCHED] WoundDataset fallback (skip instead of fake circle)')

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
        print(f'  Total patches applied: {n}')
    print('\nDone.')
