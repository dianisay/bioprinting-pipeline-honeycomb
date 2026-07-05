from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import logging

from PIL import Image, ImageDraw
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


log = logging.getLogger(__name__)

VALID_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
VALID_LABEL_EXTS = {'.png', '.bmp', '.npy', '.npz'}


def _list_files(dirpath: Path, exts: set) -> List[Path]:
    if not dirpath or not dirpath.exists():
        return []
    return sorted([p for p in dirpath.iterdir() if p.is_file() and p.suffix.lower() in exts])


def _pair_images_and_labels(imgs: List[Path], lbls: List[Path]) -> Tuple[List[Path], List[Optional[Path]]]:
    """Return aligned lists (images, label-or-None) by matching stems.

    If a label with the same stem isn't found for an image, the label is set to None.
    """
    lbl_map: Dict[str, Path] = {p.stem: p for p in lbls}
    paired_imgs: List[Path] = []
    paired_lbls: List[Optional[Path]] = []
    for im in imgs:
        paired_imgs.append(im)
        lbl = lbl_map.get(im.stem)
        paired_lbls.append(lbl)
    return paired_imgs, paired_lbls


def _resolve_split_dirs(base_path: Path, split_name: str) -> Tuple[Path, Path]:
    """Find (images_dir, labels_dir) for a given split under base_path.

    Looks for the common structures base_path/<split>/images and base_path/<split>/labels.
    If base_path doesn't directly contain the split, we look into immediate subdirectories
    (useful when base_path is a container with dataset subfolders).
    Returns (images_dir, labels_dir) where either may be Path(not_exist).
    """
    images_dir = base_path / split_name / 'images'
    labels_dir = base_path / split_name / 'labels'
    if images_dir.exists():
        return images_dir, labels_dir

    # try some alternate names
    alt_images = [base_path / split_name / 'image', base_path / f'{split_name}_images']
    for p in alt_images:
        if p.exists():
            # labels are expected in a 'labels' sibling
            lbl = p.parent / 'labels'
            return p, lbl

    # if not found, search immediate children for split
    if base_path.exists() and any(base_path.iterdir()):
        for child in base_path.iterdir():
            if not child.is_dir():
                continue
            imgs = child / split_name / 'images'
            lbls = child / split_name / 'labels'
            if imgs.exists():
                return imgs, lbls
    # not found
    return images_dir, labels_dir


class WoundDataset(Dataset):
    """Simple, robust dataset mapping images to masks.

    - If a mask is missing for an image, creates a soft fallback mask (small centred circle) so
      that downstream training loops won't crash.
    - Loads images as RGB tensors (C,H,W) float in [0,1].
    - Loads masks as single-channel (H,W) LongTensor with values 0/1.
    """

    def __init__(self, image_paths: List[Path], label_paths: List[Optional[Path]], transforms=None, image_size: Optional[int] = None):
        assert len(image_paths) == len(label_paths)
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transforms = transforms
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def _load_image(self, p: Path):
        img = Image.open(p).convert('RGB')
        if self.image_size:
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return t

    def _load_mask(self, p: Optional[Path], shape: Tuple[int, int]):
        H, W = shape
        if p is None or not p.exists():
            # create a fallback centered circular mask (radius = 0.15*min_dim)
            r = int(round(0.15 * min(H, W)))
            mask = Image.new('L', (W, H), 0)
            draw = ImageDraw.Draw(mask)
            cx, cy = W // 2, H // 2
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
            arr = np.array(mask, dtype=np.uint8) // 255
            return torch.from_numpy(arr).long()
        # load provided mask and convert to binary
        m = Image.open(p).convert('L')
        if self.image_size:
            m = m.resize((self.image_size, self.image_size), Image.NEAREST)
        arr = np.array(m, dtype=np.uint8)
        # Binarize: all non-zero becomes 1
        arr = (arr > 0).astype(np.uint8)
        return torch.from_numpy(arr).long()

    def __getitem__(self, idx):
        img_p = self.image_paths[idx]
        lbl_p = self.label_paths[idx]
        img_t = self._load_image(img_p)
        mask_t = self._load_mask(lbl_p, (img_t.shape[1], img_t.shape[2]))
        if self.transforms:
            # transforms should handle image and mask jointly; user can supply torchvision/albumentations
            try:
                out = self.transforms(image=img_t, mask=mask_t)
                return out['image'], out['mask']
            except Exception:
                # user transforms may expect numpy arrays
                pass
        return img_t, mask_t


class WoundDataModule(pl.LightningDataModule):
    """DataModule that aggregates multiple datasets with robust path resolution.

    Example:
        from pathlib import Path
        dataset_paths = [Path('/kaggle/input/.../Foot Ulcer Segmentation Challenge'), Path('/kaggle/input/.../Medetec_foot_ulcer_224')]
        dm = WoundDataModule(dataset_paths, batch_size=8, image_size=256)
        dm.setup()
        train_loader = dm.train_dataloader()

    Notes:
    - validation split falls back to test if missing
    - supports containers where base_path contains multiple dataset subfolders
    """

    def __init__(self, dataset_paths: List[Path], batch_size: int = 8, num_workers: int = 4, transforms=None, image_size: Optional[int] = None):
        super().__init__()
        self.dataset_paths = [Path(p) for p in dataset_paths]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms
        self.image_size = image_size

        self.splits = {
            'train': {'images': [], 'labels': []},
            'val': {'images': [], 'labels': []},
            'test': {'images': [], 'labels': []},
        }

    def setup(self, stage: Optional[str] = None):
        for base in self.dataset_paths:
            # Resolve train
            train_imgs_dir, train_lbl_dir = _resolve_split_dirs(base, 'train')
            train_imgs = _list_files(train_imgs_dir, VALID_IMAGE_EXTS)
            train_lbls = _list_files(train_lbl_dir, VALID_LABEL_EXTS)
            imgs, lbls = _pair_images_and_labels(train_imgs, train_lbls)
            self.splits['train']['images'].extend(imgs)
            self.splits['train']['labels'].extend(lbls)

            # Resolve test
            test_imgs_dir, test_lbl_dir = _resolve_split_dirs(base, 'test')
            test_imgs = _list_files(test_imgs_dir, VALID_IMAGE_EXTS)
            test_lbls = _list_files(test_lbl_dir, VALID_LABEL_EXTS)
            imgs, lbls = _pair_images_and_labels(test_imgs, test_lbls)
            self.splits['test']['images'].extend(imgs)
            self.splits['test']['labels'].extend(lbls)

            # Resolve validation (fallback to test if missing)
            val_imgs_dir, val_lbl_dir = _resolve_split_dirs(base, 'validation')
            if not val_imgs_dir.exists():
                val_imgs_dir, val_lbl_dir = test_imgs_dir, test_lbl_dir
            val_imgs = _list_files(val_imgs_dir, VALID_IMAGE_EXTS)
            val_lbls = _list_files(val_lbl_dir, VALID_LABEL_EXTS)
            imgs, lbls = _pair_images_and_labels(val_imgs, val_lbls)
            self.splits['val']['images'].extend(imgs)
            self.splits['val']['labels'].extend(lbls)

        # Final consistency check: ensure each split's image and label list lengths match
        for s in ['train', 'val', 'test']:
            imgs = self.splits[s]['images']
            lbls = self.splits[s]['labels']
            if len(imgs) != len(lbls):
                # pair_images_and_labels should have created same length lists with None for missing labels
                log.info(f"Split {s}: {len(imgs)} images, {len([l for l in lbls if l is not None])} labelled")

        log.info(f"Loaded dataset splits: train={len(self.splits['train']['images'])}, val={len(self.splits['val']['images'])}, test={len(self.splits['test']['images'])}")

    def _make_dataset(self, split: str) -> WoundDataset:
        imgs = self.splits[split]['images']
        lbls = self.splits[split]['labels']
        return WoundDataset(imgs, lbls, transforms=self.transforms, image_size=self.image_size)

    def train_dataloader(self):
        ds = self._make_dataset('train')
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        ds = self._make_dataset('val')
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        ds = self._make_dataset('test')
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
