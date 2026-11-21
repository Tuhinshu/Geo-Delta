"""
PyTorch Training Pipeline for GeoDelta Siamese Cross-Attention Engine on BigEarthNet-S2.
SIH Problem Statement SIH26227 - Ministry of Defence, Government of India.

Fine-tunes the Siamese ResNet-34 Feature Extractor + Cross-Attention Engine + UNet++ Decoder
on real Sentinel-2 Level-2A surface reflectance imagery extracted from BigEarthNet-S2.tar.zst.
"""

import os
import sys
import time
import argparse
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add backend directory to sys.path
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.services.siamese_engine import SiameseCrossAttentionEngine
from app.services.vlm_encoder import RemoteCLIPTextEncoder
from app.training.bigearthnet_dataset import BigEarthNetS2Dataset


class DiceLoss(nn.Module):
    """Computes differentiable soft Dice Loss for imbalanced segmentation masks."""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1.0 - dice


class CombinedChangeLoss(nn.Module):
    """Weighted combination of Binary Cross-Entropy Loss and Soft Dice Loss."""
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_clamped = torch.clamp(pred, 1e-7, 1.0 - 1e-7)
        loss_bce = self.bce(pred_clamped, target)
        loss_dice = self.dice(pred, target)
        return self.bce_weight * loss_bce + self.dice_weight * loss_dice


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """Computes Precision, Recall, F1-Score, and Intersection-over-Union (IoU)."""
    pred_bin = (pred >= threshold).float()
    target_bin = (target >= 0.5).float()

    tp = (pred_bin * target_bin).sum().item()
    fp = (pred_bin * (1.0 - target_bin)).sum().item()
    fn = ((1.0 - pred_bin) * target_bin).sum().item()

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "iou": round(iou, 4)
    }


def train_on_bigearthnet(
    patches_dir: str = r"D:\Geo Delta - SIH\storage\samples\bigearthnet_test\BigEarthNet-S2",
    output_model_path: str = r"D:\Geo Delta - SIH\storage\models\siamese_trained_model.pt",
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 1e-4,
    device_name: str = "cpu"
) -> Dict[str, Any]:
    """
    Executes training on BigEarthNet-S2 Sentinel-2 imagery and saves calibrated weights.
    """
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    print(f"\n================================================================================")
    print(f"  GEODELTA SENTINEL-2 DEEP LEARNING TRAINING (BIGEARTHNET-S2)")
    print(f"================================================================================")
    print(f"  Dataset Directory : {patches_dir}")
    print(f"  Target Checkpoint : {output_model_path}")
    print(f"  Hardware Target   : {device}")
    print(f"  Epochs / Batch    : {epochs} Epochs | Batch Size: {batch_size}")
    print(f"  Learning Rate     : {lr}")
    print(f"================================================================================\n")

    # 1. Initialize BigEarthNet Sentinel-2 Dataset
    dataset = BigEarthNetS2Dataset(
        patches_dir=patches_dir,
        image_size=(256, 256),
        num_pairs_per_patch=3,
        seed=42
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    print(f"[Training Pipeline] Loaded {len(dataset)} Sentinel-2 training pairs.")

    # 2. Initialize Models
    model = SiameseCrossAttentionEngine(in_channels=3, embed_dim=512).to(device)

    # Load existing checkpoint if present to warm-start fine-tuning
    if os.path.isfile(output_model_path):
        try:
            checkpoint = torch.load(output_model_path, map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict, strict=False)
            print(f"[Training Pipeline] Warm-started from existing checkpoint: {output_model_path}")
        except Exception as e:
            print(f"[Training Pipeline] Starting fresh training (checkpoint note: {e})")

    vlm = RemoteCLIPTextEncoder().to(device)
    vlm.eval()

    criterion = CombinedChangeLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    start_time = time.perf_counter()
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for t1, t2, mask, prompts in loader:
            t1 = t1.to(device)
            t2 = t2.to(device)
            mask = mask.to(device)

            with torch.no_grad():
                prompt_text = prompts[0] if isinstance(prompts, (list, tuple)) else str(prompts)
                e_star = vlm.encode_conditioned_prompt(prompt_text).to(device)

            optimizer.zero_grad()
            pred = model(t1, t2, e_star)
            loss = criterion(pred, mask)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / max(batch_count, 1)

        # Validation evaluation pass
        model.eval()
        with torch.no_grad():
            val_t1, val_t2, val_mask, val_prompts = next(iter(loader))
            val_t1 = val_t1.to(device)
            val_t2 = val_t2.to(device)
            val_e_star = vlm.encode_conditioned_prompt(val_prompts[0]).to(device)
            val_pred = model(val_t1, val_t2, val_e_star)
            metrics = compute_metrics(val_pred, val_mask.to(device))

        epoch_record = {
            "epoch": epoch,
            "loss": round(avg_loss, 4),
            "iou": metrics["iou"],
            "f1_score": metrics["f1_score"],
            "precision": metrics["precision"],
            "recall": metrics["recall"]
        }
        history.append(epoch_record)
        print(
            f"  [Epoch {epoch:02d}/{epochs:02d}] "
            f"Loss: {avg_loss:.4f} | "
            f"IoU: {metrics['iou']:.4f} | "
            f"F1: {metrics['f1_score']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f}"
        )

    total_training_time = round(time.perf_counter() - start_time, 2)

    # 3. Save Calibrated Weights
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "dataset_source": "BigEarthNet-S2 (Sentinel-2 Level-2A Multi-Spectral)",
        "in_channels": 3,
        "embed_dim": 512,
        "epochs_trained": epochs,
        "final_loss": history[-1]["loss"],
        "final_iou": history[-1]["iou"],
        "final_f1": history[-1]["f1_score"],
        "trained_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "is_trained": True
    }, output_model_path)

    print(f"\n[Training Complete] Model successfully saved to: {output_model_path}")
    print(f"[Training Complete] Elapsed Time: {total_training_time}s | Final IoU: {history[-1]['iou']}\n")

    return {
        "status": "SUCCESS",
        "model_path": output_model_path,
        "elapsed_seconds": total_training_time,
        "history": history
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GeoDelta on BigEarthNet-S2")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    train_on_bigearthnet(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
