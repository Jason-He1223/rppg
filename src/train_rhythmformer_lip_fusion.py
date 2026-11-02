
#!/usr/bin/env python3

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix
)

DEFAULT_DATA_ROOT = Path(
    "/iacl/pg25/zijie/rppg/rPPG-TalkingFace-Detector/data"
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset
# ============================================================

class LipFusionDataset(Dataset):
    def __init__(
        self, data_root, split, mode,
        compression_prob=0.25
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.mode = mode
        self.compression_prob = (
            compression_prob if split == "train" and mode == "fusion"
            else 0.0
        )

        base = pd.read_csv(
            self.data_root / "multiwindow_manifest.csv"
        )
        lip = pd.read_csv(
            self.data_root / "lip_features_multiwindow"
            / "lip_manifest.csv"
        )

        keys = ["split", "class", "video_id", "window_id"]

        if lip.duplicated(keys).any():
            raise ValueError("Duplicate keys in lip manifest")

        lip = lip[keys + ["original_lip_path"]]
        base = base[base["split"] == split].copy()

        self.df = base.merge(
            lip, on=keys, how="left", validate="one_to_one"
        ).reset_index(drop=True)

        if self.df.empty:
            raise ValueError(f"No samples for split {split}")

        missing = self.df["original_lip_path"].isna()
        if missing.any():
            examples = self.df.loc[missing, "window_id"].head(10).tolist()
            raise ValueError(
                f"{split}: {missing.sum()} windows missing lip features. "
                f"Examples: {examples}"
            )

        self.df["original_lip_path"] = (
            self.df["original_lip_path"].astype(str)
        )

        required = ["original_lip_path"]
        if mode == "fusion":
            required += [
                "original_latent_path",
                "original_waveform_path"
            ]

        for col in required:
            missing_files = [
                p for p in self.df[col]
                if pd.isna(p) or not Path(str(p)).is_file()
            ]
            if missing_files:
                raise FileNotFoundError(
                    f"{split}: {len(missing_files)} missing files "
                    f"for {col}; first: {missing_files[0]}"
                )

        print(
            f"{split}: {len(self.df)} windows | "
            f"{self.df['video_id'].nunique()} videos | "
            f"real={(self.df['label'] == 0).sum()} | "
            f"fake={(self.df['label'] == 1).sum()} | "
            f"H264_prob={self.compression_prob}"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        lip = np.load(
            row["original_lip_path"]
        ).astype(np.float32)

        if lip.shape != (160, 512):
            raise ValueError(
                f"Invalid lip shape {lip.shape}: "
                f"{row['original_lip_path']}"
            )

        if not np.isfinite(lip).all():
            raise ValueError(
                f"Nonfinite lip features: {row['window_id']}"
            )

        y = torch.tensor(float(row["label"]), dtype=torch.float32)
        video_id = str(row["video_id"])
        lip = torch.from_numpy(lip)

        if self.mode == "lip_only":
            return lip, y, video_id

        latent_path = Path(row["original_latent_path"])
        wave_path = Path(row["original_waveform_path"])

        if (
            self.compression_prob > 0
            and random.random() < self.compression_prob
        ):
            lp = row.get("h264_latent_path", "")
            wp = row.get("h264_waveform_path", "")

            if pd.notna(lp) and pd.notna(wp):
                lp, wp = Path(str(lp)), Path(str(wp))
                if lp.is_file() and wp.is_file():
                    latent_path, wave_path = lp, wp

        latent = np.load(latent_path).astype(np.float32)
        waveform = np.load(wave_path).astype(np.float32)

        if latent.shape != (64, 160, 8, 8):
            raise ValueError(
                f"Invalid latent shape: {latent.shape}"
            )

        if waveform.shape != (160,):
            raise ValueError(
                f"Invalid waveform shape: {waveform.shape}"
            )

        waveform = (
            waveform - waveform.mean()
        ) / (waveform.std() + 1e-6)

        return (
            torch.from_numpy(latent),
            torch.from_numpy(waveform),
            lip, y, video_id
        )


# ============================================================
# Lip temporal Transformer + attention pooling
# ============================================================

class LipTemporalEncoder(nn.Module):
    def __init__(
        self, input_dim=512, d_model=128,
        nhead=4, layers=2, dropout=0.30,
        seq_len=160
    ):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model)
        )

        self.pos = nn.Parameter(
            torch.zeros(1, seq_len, d_model)
        )
        nn.init.trunc_normal_(self.pos, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            layer, num_layers=layers
        )

        self.norm = nn.LayerNorm(d_model)

        self.attention = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.output = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [B,160,512]
        x = self.proj(x) + self.pos
        x = self.transformer(x)
        x = self.norm(x)

        scores = self.attention(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1)

        feature = (
            x * weights.unsqueeze(-1)
        ).sum(dim=1)

        return self.output(feature)


# ============================================================
# Existing RhythmFormer latent encoder
# ============================================================

class RhythmFormerLatentEncoder(nn.Module):
    def __init__(
        self, input_dim=64, seq_len=160,
        spatial_size=8, d_model=32,
        nhead=4, temporal_layers=1,
        spatial_layers=1, dim_feedforward=64,
        dropout=0.30
    ):
        super().__init__()

        self.num_regions = spatial_size * spatial_size

        self.input_proj = nn.Linear(input_dim, d_model)

        self.temporal_pos = nn.Parameter(
            torch.zeros(1, seq_len, d_model)
        )
        self.spatial_pos = nn.Parameter(
            torch.zeros(1, self.num_regions, d_model)
        )

        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.temporal_transformer = nn.TransformerEncoder(
            temporal_layer, num_layers=temporal_layers
        )

        self.temporal_signal_head = nn.Linear(d_model, 1)

        spatial_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.spatial_transformer = nn.TransformerEncoder(
            spatial_layer, num_layers=spatial_layers
        )

        self.spatial_projection = nn.Sequential(
            nn.Linear(self.num_regions * d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, x):
        # [B,64,160,8,8]
        B, C, T, H, W = x.shape

        x = x.permute(0, 3, 4, 2, 1).contiguous()
        x = x.view(B, H * W, T, C)
        x = self.input_proj(x)

        x = x.view(B * H * W, T, -1)
        x = x + self.temporal_pos
        x = self.temporal_transformer(x)

        regional_waveforms = (
            self.temporal_signal_head(x)
            .squeeze(-1)
            .view(B, H * W, T)
        )

        x = x.mean(dim=1)
        x = x.view(B, H * W, -1)

        x = x + self.spatial_pos
        x = self.spatial_transformer(x)

        x = x.reshape(B, -1)
        latent_feature = self.spatial_projection(x)

        return latent_feature, regional_waveforms


# ============================================================
# Existing RhythmFormer representation + consistency
# ============================================================

class RhythmFormerFeatureEncoder(nn.Module):
    def __init__(
        self, d_model=32, nhead=4,
        temporal_layers=1, spatial_layers=1,
        ff_dim=64, dropout=0.30
    ):
        super().__init__()

        self.latent_encoder = RhythmFormerLatentEncoder(
            d_model=d_model,
            nhead=nhead,
            temporal_layers=temporal_layers,
            spatial_layers=spatial_layers,
            dim_feedforward=ff_dim,
            dropout=dropout
        )

        self.waveform_encoder = nn.Sequential(
            nn.Linear(160, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model)
        )

        self.consistency_encoder = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, latent, waveform):
        latent_feature, regional = self.latent_encoder(latent)
        waveform_feature = self.waveform_encoder(waveform)

        regional_norm = (
            regional - regional.mean(dim=-1, keepdim=True)
        )
        regional_norm = (
            regional_norm
            / (regional_norm.norm(dim=-1, keepdim=True) + 1e-6)
        )

        wave_norm = (
            waveform - waveform.mean(dim=-1, keepdim=True)
        )
        wave_norm = (
            wave_norm
            / (wave_norm.norm(dim=-1, keepdim=True) + 1e-6)
        )

        correlation = (
            regional_norm * wave_norm.unsqueeze(1)
        ).sum(dim=-1)

        consistency_feature = self.consistency_encoder(
            correlation
        )

        return torch.cat(
            [
                latent_feature,
                waveform_feature,
                consistency_feature
            ],
            dim=1
        )


# ============================================================
# Two model modes
# ============================================================

class LipDetectionModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.mode = args.mode

        self.lip_encoder = LipTemporalEncoder(
            d_model=args.lip_dim,
            nhead=args.lip_heads,
            layers=args.lip_layers,
            dropout=args.dropout
        )

        if self.mode == "fusion":
            self.rhythm_encoder = RhythmFormerFeatureEncoder(
                d_model=args.d_model,
                nhead=args.nhead,
                temporal_layers=args.temporal_layers,
                spatial_layers=args.spatial_layers,
                ff_dim=args.ff_dim,
                dropout=args.dropout
            )
            fusion_dim = args.lip_dim + args.d_model * 3
        else:
            fusion_dim = args.lip_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(64, 1)
        )

    def forward(self, lip, latent=None, waveform=None):
        lip_feature = self.lip_encoder(lip)

        if self.mode == "fusion":
            if latent is None or waveform is None:
                raise ValueError(
                    "Fusion mode requires latent and waveform"
                )

            rhythm_feature = self.rhythm_encoder(
                latent, waveform
            )
            feature = torch.cat(
                [rhythm_feature, lip_feature], dim=1
            )
        else:
            feature = lip_feature

        return self.classifier(feature).squeeze(-1)


# ============================================================
# Batch preparation
# ============================================================

def prepare_batch(batch, mode, device):
    if mode == "lip_only":
        lip, y, video_ids = batch
        lip = lip.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        return lip, None, None, y, video_ids

    latent, waveform, lip, y, video_ids = batch

    return (
        lip.to(device, non_blocking=True),
        latent.to(device, non_blocking=True),
        waveform.to(device, non_blocking=True),
        y.to(device, non_blocking=True),
        video_ids
    )


# ============================================================
# Video-level evaluation
# ============================================================

@torch.no_grad()
def evaluate_video_level(model, loader, device, mode):
    model.eval()

    logits_all = []
    labels_all = []
    ids_all = []

    for batch in loader:
        lip, latent, waveform, y, ids = prepare_batch(
            batch, mode, device
        )

        logits = model(lip, latent, waveform)

        logits_all.extend(logits.cpu().numpy().tolist())
        labels_all.extend(y.cpu().numpy().tolist())
        ids_all.extend(list(ids))

    result = pd.DataFrame({
        "video_id": ids_all,
        "label": labels_all,
        "logit": logits_all
    })
    window_y = result["label"].to_numpy(dtype=np.float32)
    window_logits = result["logit"].to_numpy(dtype=np.float32)
    window_auc = roc_auc_score(window_y, window_logits)

    print(f"Window AUC: {window_auc:.4f}")

    if result.groupby("video_id")["label"].nunique().max() > 1:
        raise RuntimeError(
            "Multiple labels found for one video_id"
        )

    video_df = (
        result.groupby("video_id", as_index=False)
        .agg(
            label=("label", "first"),
            logit=("logit", "mean"),
            n_windows=("logit", "size")
        )
    )

    y = video_df["label"].to_numpy(dtype=np.float32)
    logits = video_df["logit"].to_numpy(dtype=np.float32)

    probs = torch.sigmoid(
        torch.from_numpy(logits)
    ).numpy()

    auc = (
        roc_auc_score(y, probs)
        if len(np.unique(y)) == 2 else float("nan")
    )

    loss = nn.functional.binary_cross_entropy_with_logits(
        torch.from_numpy(logits),
        torch.from_numpy(y)
    ).item()

    video_df["probability"] = probs
    video_df["prediction"] = (probs >= 0.5).astype(int)

    return loss, auc, y, probs, video_df


# ============================================================
# Augmentation
# ============================================================

def augment_batch(lip, latent, waveform, mode):
    if mode == "fusion":
        latent = latent + torch.randn_like(latent) * 0.005

        if random.random() < 0.5:
            h = random.randrange(8)
            w = random.randrange(8)
            latent[:, :, :, h, w] = 0

    if random.random() < 0.5:
        cut_len = random.randint(4, 16)
        start = random.randint(0, 160 - cut_len)
        lip[:, start:start + cut_len, :] = 0

        if mode == "fusion":
            latent[:, :, start:start + cut_len, :, :] = 0
            waveform[:, start:start + cut_len] = 0

    return lip, latent, waveform


# ============================================================
# Training
# ============================================================

def train(args):
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Mode:", args.mode)

    train_ds = LipFusionDataset(
        args.data_root, "train", args.mode,
        compression_prob=args.compression_prob
    )
    val_ds = LipFusionDataset(
        args.data_root, "val", args.mode,
        compression_prob=0
    )
    test_ds = LipFusionDataset(
        args.data_root, "test", args.mode,
        compression_prob=0
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers,
        pin_memory=device.type == "cuda"
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_batch,
        shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda"
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.eval_batch,
        shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda"
    )

    model = LipDetectionModel(args).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    y_train = train_ds.df["label"].to_numpy()
    n_real = int((y_train == 0).sum())
    n_fake = int((y_train == 1).sum())

    if n_real == 0 or n_fake == 0:
        raise ValueError("Training requires both classes")

    pos_weight = torch.tensor(
        [n_real / n_fake],
        device=device,
        dtype=torch.float32
    )

    print(f"Train real={n_real}, fake={n_fake}")
    print(f"BCE pos_weight={pos_weight.item():.4f}")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    checkpoint_dir = (
        Path(args.data_root) / "checkpoints"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = (
        checkpoint_dir
        / f"rhythmformer_lip_{args.mode}_attention.pt"
    )

    results_dir = (
        Path(args.data_root) / "results"
        / f"lip_{args.mode}_attention"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    best_auc = -1.0
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for batch in train_loader:
            lip, latent, waveform, y, _ = prepare_batch(
                batch, args.mode, device
            )

            if args.augment:
                lip, latent, waveform = augment_batch(
                    lip, latent, waveform, args.mode
                )

            optimizer.zero_grad(set_to_none=True)

            logits = model(lip, latent, waveform)
            loss = criterion(logits, y)

            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), 5.0
            )
            optimizer.step()

            losses.append(loss.item())

        val_loss, val_auc, _, _, val_df = (
            evaluate_video_level(
                model, val_loader, device, args.mode
            )
        )

        if np.isfinite(val_auc):
            scheduler.step(val_auc)

        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss {np.mean(losses):.4f} | "
            f"Val Video Loss {val_loss:.4f} | "
            f"Val Video AUC {val_auc:.4f} | "
            f"Videos {len(val_df)} | "
            f"LR {lr:.2e}"
        )

        if np.isfinite(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "val_video_auc": val_auc,
                "model_state_dict": model.state_dict(),
                "args": vars(args)
            }, checkpoint_path)

            val_df.to_csv(
                results_dir / "best_val_predictions.csv",
                index=False
            )

            print("  -> New best video-level AUC")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_epoch < 0:
        raise RuntimeError(
            "No valid validation checkpoint was produced"
        )

    checkpoint = torch.load(
        checkpoint_path, map_location=device
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print("\nLoading best checkpoint...")
    print("Best epoch:", checkpoint["epoch"])
    print(
        "Best validation VIDEO AUC:",
        f"{checkpoint['val_video_auc']:.4f}"
    )

    test_loss, test_auc, y, p, test_df = (
        evaluate_video_level(
            model, test_loader, device, args.mode
        )
    )

    pred = (p >= 0.5).astype(int)

    accuracy = accuracy_score(y, pred)
    f1 = f1_score(y, pred, zero_division=0)
    precision = precision_score(y, pred, zero_division=0)
    recall = recall_score(y, pred, zero_division=0)
    cm = confusion_matrix(y, pred, labels=[0, 1])

    print("\n==========================================")
    print(f"FINAL VIDEO-LEVEL TEST: {args.mode}")
    print("==========================================")
    print(f"Videos:    {len(test_df)}")
    print(f"Windows:   {len(test_ds)}")
    print(f"Loss:      {test_loss:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"AUC:       {test_auc:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print("Confusion matrix:")
    print(cm)

    test_df.to_csv(
        results_dir / "test_video_predictions.csv",
        index=False
    )

    print("\nPerformance by manipulation method:")

    test_df["method"] = test_df["video_id"].apply(
        lambda v: v.split("__", 1)[0]
        if "__" in v else "real"
    )

    method_rows = []
    for method, group in test_df.groupby("method"):
        fake_group = group[group["label"] == 1]
        if len(fake_group) == 0:
            continue

        method_recall = (
            fake_group["prediction"] == 1
        ).mean()

        errors = int(
            (fake_group["prediction"] == 0).sum()
        )

        method_rows.append({
            "method": method,
            "videos": len(fake_group),
            "false_negatives": errors,
            "recall": method_recall
        })

    if method_rows:
        method_df = pd.DataFrame(method_rows)
        method_df = method_df.sort_values(
            "false_negatives", ascending=False
        )
        print(method_df.to_string(index=False))

        method_df.to_csv(
            results_dir / "performance_by_method.csv",
            index=False
        )

    print("\nCheckpoint:", checkpoint_path)
    print("Results:", results_dir)


# ============================================================
# Arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT)
    )

    parser.add_argument(
        "--mode",
        choices=["lip_only", "fusion"],
        required=True
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4
    )

    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument(
        "--temporal-layers", type=int, default=1
    )
    parser.add_argument(
        "--spatial-layers", type=int, default=1
    )
    parser.add_argument("--ff-dim", type=int, default=64)

    parser.add_argument("--lip-dim", type=int, default=128)
    parser.add_argument("--lip-heads", type=int, default=4)
    parser.add_argument("--lip-layers", type=int, default=2)

    parser.add_argument(
        "--dropout", type=float, default=0.30
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action="store_true")

    parser.add_argument(
        "--compression-prob",
        type=float,
        default=0.25,
        help="H264 augmentation probability for RhythmFormer only"
    )

    args = parser.parse_args()

    if not 0 <= args.compression_prob <= 1:
        parser.error("--compression-prob must be between 0 and 1")

    if args.lip_dim % args.lip_heads != 0:
        parser.error("--lip-dim must be divisible by --lip-heads")

    return args


if __name__ == "__main__":
    train(parse_args())
