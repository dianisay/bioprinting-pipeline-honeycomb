"""End-to-end verification: generate data, train 2 epochs, evaluate.

Run this locally to confirm the full pipeline works before Kaggle.
"""

import sys
sys.path.insert(0, ".")

import torch
import numpy as np
import tempfile
import os
from pathlib import Path

from data.synthetic_generator import generate_dataset
from data.dataset import create_dataloaders
from training.train import Trainer
from training.evaluate import evaluate


def test_full_pipeline():
    print("=" * 60)
    print("FULL PIPELINE VERIFICATION (CPU, 2 epochs, tiny dataset)")
    print("=" * 60)

    # 1. Generate tiny synthetic dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        synth_dir = os.path.join(tmpdir, "synthetic")
        results_dir = os.path.join(tmpdir, "results")

        print("\n[1/5] Generating 30 synthetic samples...")
        generate_dataset(synth_dir, num_samples=30, seed=0)

        # 2. Create dataloaders
        print("\n[2/5] Creating dataloaders...")
        train_loader, val_loader, test_loader = create_dataloaders(
            synthetic_dir=synth_dir,
            batch_size=4,
            num_radii=32,
        )

        # 3. Test each decoder type
        decoder_types = ["polar", "detr", "autoregressive"]
        for decoder_type in decoder_types:
            print(f"\n[3/5] Training {decoder_type} decoder (2 epochs)...")
            trainer = Trainer(
                decoder_type=decoder_type,
                d_model=64,
                num_heads=4,
                num_encoder_layers=2,
                num_decoder_layers=2,
                num_points=32,
                lr=1e-3,
                batch_size=4,
                max_epochs=2,
                patience=5,
                device="cpu",
                output_dir=results_dir,
                pretrained_backbone=False,
            )
            history = trainer.train(train_loader, val_loader)

            assert len(history["train_loss"]) == 2
            assert history["train_loss"][0] > 0
            print(f"    -> {decoder_type} OK: loss = {history['train_loss'][-1]:.4f}")

        # 4. Test evaluation
        print("\n[4/5] Running evaluation on polar checkpoint...")
        ckpt_path = os.path.join(results_dir, "polar", "best.pth")
        if os.path.exists(ckpt_path):
            results = evaluate(
                checkpoint_path=ckpt_path,
                synthetic_dir=synth_dir,
                batch_size=4,
            )
            assert "chamfer" in results
            print(f"    -> Evaluation OK: Chamfer = {results['chamfer']['mean']:.4f}")

        # 5. Verify outputs
        print("\n[5/5] Checking output files...")
        for dt in decoder_types:
            assert Path(results_dir, dt, "best.pth").exists(), f"Missing best.pth for {dt}"
            assert Path(results_dir, dt, "history.json").exists(), f"Missing history.json for {dt}"
        print("    -> All checkpoints and histories saved correctly")

    print("\n" + "=" * 60)
    print("ALL PIPELINE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_full_pipeline()
