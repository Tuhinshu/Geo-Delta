"""
PyTorch Training & Fine-Tuning Pipeline for GeoDelta Siamese Cross-Attention Network.
Trains the Siamese ResNet-34 Feature Extractor + Cross-Attention + UNet++ Decoder
using a combined Binary Cross-Entropy + Dice Loss for precise satellite change detection.
"""

import os
import sys
import time
import argparse
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add backend directory to sys.path so app modules are resolvable
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.services.siamese_engine import SiameseCrossAttentionEngine
from app.services.vlm_encoder import RemoteCLIPTextEncoder
from app.training.dataset import BitemporalChangeDataset, create_tactical_sample_pairs


class DiceLoss(nn.Module):
    """Computes differentiable soft Dice Loss for imbalanced segmentation masks."""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred and target are in [0.0, 1.0], shape (B, 1, H, W)
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1.0 - dice


class CombinedChangeLoss(nn.Module):
    """Weighted combination of BCE Loss and Soft Dice Loss."""
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
    """Computes Precision, Recall, F1-Score, and IoU."""
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


def train_change_detector(
    data_dir: str,
    output_model_path: str = "storage/models/siamese_trained_model.pt",
    epochs: int = 5,
    batch_size: int = 2,
    lr: float = 1e-4,
    device_name: str = "cpu"
) -> Dict[str, Any]:
    """
    Executes training and saves the calibrated model weights.
    """
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    print(f"[GeoDelta Training] Initializing pipeline on device: {device}")

    # Ensure dataset exists; synthesize tactical pairs if directory empty
    if not os.path.isdir(os.path.join(data_dir, "A")):
        print(f"[GeoDelta Training] Generating tactical defense training pairs in {data_dir}...")
        data_dir = create_tactical_sample_pairs(data_dir)

    dataset = BitemporalChangeDataset(root_dir=data_dir, image_size=(256, 256))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    print(f"[GeoDelta Training] Loaded dataset with {len(dataset)} bitemporal samples.")

    # Initialize models
    # Note: Using 3-channel RGB for universal support (PNG/TIFF)
    model = SiameseCrossAttentionEngine(in_channels=3, embed_dim=512).to(device)
    vlm = RemoteCLIPTextEncoder().to(device)
    vlm.eval()

    criterion = CombinedChangeLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    start_time = time.perf_counter()
    best_iou = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for t1, t2, mask, prompts in loader:
            t1 = t1.to(device)
            t2 = t2.to(device)
            mask = mask.to(device)

            # Compute CLIP text embeddings for prompts in batch
            with torch.no_grad():
                # Single prompt per batch item
                prompt_text = prompts[0] if isinstance(prompts, (list, tuple)) else str(prompts)
                e_star = vlm.encode_conditioned_prompt(prompt_text).to(device)

            optimizer.zero_grad()
            # Forward pass through Siamese network
            pred = model(t1, t2, e_star)
            loss = criterion(pred, mask)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(loader), 1)

        # Validation pass over training samples
        model.eval()
        with torch.no_grad():
            val_t1, val_t2, val_mask, val_prompts = next(iter(loader))
            val_t1 = val_t1.to(device)
            val_val_t2 = val_t2.to(device)
            e_val = vlm.encode_conditioned_prompt(val_prompts[0]).to(device)
            val_pred = model(val_t1, val_val_t2, e_val)
            metrics = compute_metrics(val_pred, val_mask.to(device))

        print(
            f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f} | "
            f"IoU: {metrics['iou']:.4f} | F1: {metrics['f1_score']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}"
        )

        if metrics["iou"] >= best_iou or epoch == epochs:
            best_iou = metrics["iou"]
            os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "in_channels": 3,
                "embed_dim": 512,
                "best_iou": best_iou,
                "metrics": metrics,
                "epochs_trained": epoch,
                "timestamp": time.time()
            }, output_model_path)

    total_time = round(time.perf_counter() - start_time, 2)
    print(f"[GeoDelta Training] Checkpoint successfully saved to {output_model_path} ({total_time}s).")

    return {
        "status": "SUCCESS",
        "output_path": output_model_path,
        "epochs": epochs,
        "best_iou": best_iou,
        "final_metrics": metrics,
        "duration_seconds": total_time
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GeoDelta Siamese Model Trainer")
    parser.add_argument("--data-dir", default="storage/samples/tactical_train", help="Dataset directory")
    parser.add_argument("--save-path", default="storage/models/siamese_trained_model.pt", help="Checkpoint output path")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", default="cpu", help="Compute device (cpu/cuda)")
    args = parser.parse_args()

    res = train_change_detector(
        data_dir=args.data_dir,
        output_model_path=args.save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_name=args.device
    )
    print(res)
