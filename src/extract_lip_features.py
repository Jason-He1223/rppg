#!/usr/bin/env python3

import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


PROJECT_ROOT = Path(
    "/iacl/pg25/zijie/rppg/rPPG-TalkingFace-Detector"
)

CLIP_LENGTH = 160
CROP_SIZE = 96
BATCH_SIZE = 64
MIN_NATIVE_FRAMES = 60

LIP_LANDMARKS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    185, 40, 39, 37, 0, 267, 269, 270, 409,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    191, 80, 81, 82, 13, 312, 311, 310, 415
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default=str(PROJECT_ROOT / "data")
    )
    parser.add_argument(
        "--manifest",
        default=None
    )
    parser.add_argument(
        "--output-dir",
        default=None
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE
    )
    parser.add_argument(
        "--bitrate-ratio",
        type=float,
        default=0.70
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="all"
    )
    parser.add_argument(
        "--skip-h264",
        action="store_true"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of unique videos; 0 = all"
    )
    return parser.parse_args()


def get_video_path(row, video_root):
    video_id = str(row["video_id"])
    label = str(row["class"])

    if label == "real":
        return video_root / "real" / f"{video_id}.mp4"

    if label == "fake":
        if "__" not in video_id:
            raise ValueError(f"Invalid fake video ID: {video_id}")

        method, filename = video_id.split("__", 1)
        return video_root / "fake" / method / f"{filename}.mp4"

    raise ValueError(f"Unknown class: {label}")


def load_encoder(device):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    model = model.to(device).eval()

    for p in model.parameters():
        p.requires_grad = False

    return model


def get_video_bitrate(video_path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=bit_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True
    )

    bitrate = int(result.stdout.strip())

    if bitrate <= 0:
        raise RuntimeError(f"Invalid bitrate: {video_path}")

    return bitrate


def compress_video(input_path, output_path, bitrate_ratio):
    original_bitrate = get_video_bitrate(input_path)
    target_bitrate = max(
        int(original_bitrate * bitrate_ratio),
        50000
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-i", str(input_path),
        "-c:v", "libopenh264",
        "-b:v", str(target_bitrate),
        "-an",
        str(output_path)
    ]

    subprocess.run(cmd, check=True)

    return original_bitrate, target_bitrate


def load_video(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        frames.append(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded: {video_path}")

    return frames


def crop_lips(frame, face_mesh, previous_box=None):
    h, w = frame.shape[:2]
    result = face_mesh.process(frame)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

        points = np.array([
            [landmarks[i].x * w, landmarks[i].y * h]
            for i in LIP_LANDMARKS
        ])

        x1, y1 = points.min(axis=0)
        x2, y2 = points.max(axis=0)

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        size = max(x2 - x1, y2 - y1) * 1.2
        size = max(size, 16)

        box = (
            int(cx - size / 2),
            int(cy - size / 2),
            int(cx + size / 2),
            int(cy + size / 2)
        )

    elif previous_box is not None:
        box = previous_box

    else:
        return None, None

    x1, y1, x2, y2 = box

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None, previous_box

    crop = frame[y1:y2, x1:x2]

    crop = cv2.resize(
        crop,
        (CROP_SIZE, CROP_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    return crop, box


def extract_lip_crops(frames, face_mesh):
    crops = []
    previous_box = None
    failed = 0

    for frame in frames:
        crop, previous_box = crop_lips(
            frame,
            face_mesh,
            previous_box
        )

        if crop is None:
            failed += 1
            crop = np.zeros(
                (CROP_SIZE, CROP_SIZE, 3),
                dtype=np.uint8
            )

        crops.append(crop)

    return np.stack(crops), failed


def temporal_resample(crops, target_length):
    n = len(crops)

    if n == target_length:
        return crops

    if n == 1:
        return np.repeat(crops, target_length, axis=0)

    positions = np.linspace(0, n - 1, target_length)

    output = []

    for position in positions:
        left = int(np.floor(position))
        right = min(left + 1, n - 1)
        alpha = position - left

        frame = (
            (1.0 - alpha) * crops[left].astype(np.float32)
            + alpha * crops[right].astype(np.float32)
        )

        output.append(
            np.clip(frame, 0, 255).astype(np.uint8)
        )

    return np.stack(output)


@torch.inference_mode()
def extract_features(crops, model, device, batch_size):
    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device=device
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=device
    ).view(1, 3, 1, 1)

    features = []

    for start in range(0, len(crops), batch_size):
        batch = crops[start:start + batch_size]

        x = torch.from_numpy(batch).to(device)
        x = x.permute(0, 3, 1, 2).float() / 255.0

        x = F.interpolate(
            x,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        x = (x - mean) / std

        feat = model(x)

        features.append(
            feat.cpu().numpy()
        )

    return np.concatenate(features, axis=0)


def get_window_crops(crops, start_frame):
    n_frames = len(crops)

    if n_frames < CLIP_LENGTH:
        return temporal_resample(crops, CLIP_LENGTH)

    window = crops[
        start_frame:start_frame + CLIP_LENGTH
    ]

    if len(window) != CLIP_LENGTH:
        raise RuntimeError(
            f"Window length mismatch: {len(window)}"
        )

    return window


def extract_video_features(
    video_path,
    rows,
    output_root,
    label,
    model,
    face_mesh,
    device,
    batch_size
):
    records = []
    n_new = 0
    n_skip = 0

    for _, row in rows.iterrows():
        output_path = (
            output_root
            / label
            / f"{row['window_id']}.npy"
        )

        records.append({
            "window_id": row["window_id"],
            "path": str(output_path)
        })

        if output_path.exists():
            n_skip += 1
        else:
            n_new += 1

    if n_new == 0:
        return records, 0, n_skip, 0

    frames = load_video(video_path)

    if len(frames) < MIN_NATIVE_FRAMES:
        raise RuntimeError(
            f"Too few frames: {len(frames)}"
        )

    crops, failed = extract_lip_crops(
        frames,
        face_mesh
    )

    for _, row in rows.iterrows():
        output_path = (
            output_root
            / label
            / f"{row['window_id']}.npy"
        )

        if output_path.exists():
            continue

        window_crops = get_window_crops(
            crops,
            int(row["start_frame"])
        )

        features = extract_features(
            window_crops,
            model,
            device,
            batch_size
        )

        if features.shape != (CLIP_LENGTH, 512):
            raise ValueError(
                f"Unexpected feature shape: {features.shape}"
            )

        np.save(
            output_path,
            features.astype(np.float16)
        )

    return records, n_new, n_skip, failed


def main():
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    if not 0 < args.bitrate_ratio <= 1:
        raise ValueError("--bitrate-ratio must be in (0,1]")

    data_root = Path(args.data_root)
    video_root = data_root / "videos"

    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else data_root / "multiwindow_manifest.csv"
    )

    output_root = (
        Path(args.output_dir)
        if args.output_dir
        else data_root / "lip_features_multiwindow"
    )

    ratio_tag = int(args.bitrate_ratio * 100)

    h264_root = data_root / (
        f"lip_features_multiwindow_h264_{ratio_tag}"
    )

    for root in [output_root, h264_root]:
        for label in ["real", "fake"]:
            (root / label).mkdir(
                parents=True,
                exist_ok=True
            )

    df = pd.read_csv(manifest_path)

    if args.split != "all":
        df = df[df["split"] == args.split].copy()

    groups = list(
        df.groupby(
            ["split", "class", "video_id"],
            sort=False
        )
    )

    if args.limit > 0:
        groups = groups[:args.limit]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Videos:", len(groups))
    print("H264 enabled:", not args.skip_h264)

    model = load_encoder(device)

    records = []
    failures = []

    with tempfile.TemporaryDirectory(
        prefix="lip_h264_"
    ) as temp_dir:

        temp_dir = Path(temp_dir)

        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:

            for index, (key, rows) in enumerate(groups, 1):
                split, label, video_id = key

                row = rows.iloc[0]
                video_path = get_video_path(
                    row,
                    video_root
                )

                print(
                    f"[{index}/{len(groups)}] "
                    f"{split} | {label} | {video_id}"
                )

                try:
                    if not video_path.exists():
                        raise FileNotFoundError(video_path)

                    original_records, n_new, n_skip, failed = (
                        extract_video_features(
                            video_path,
                            rows,
                            output_root,
                            label,
                            model,
                            face_mesh,
                            device,
                            args.batch_size
                        )
                    )

                    original_paths = {
                        rec["window_id"]: rec["path"]
                        for rec in original_records
                    }

                    h264_paths = {}

                    if split == "train" and not args.skip_h264:
                        h264_rows = rows[
                            rows["h264_latent_path"].notna()
                        ]

                        if len(h264_rows) > 0:
                            compressed_path = (
                                temp_dir / f"video_{index}.mp4"
                            )

                            compress_video(
                                video_path,
                                compressed_path,
                                args.bitrate_ratio
                            )

                            h264_records, _, _, _ = (
                                extract_video_features(
                                    compressed_path,
                                    h264_rows,
                                    h264_root,
                                    label,
                                    model,
                                    face_mesh,
                                    device,
                                    args.batch_size
                                )
                            )

                            h264_paths = {
                                rec["window_id"]: rec["path"]
                                for rec in h264_records
                            }

                            compressed_path.unlink(
                                missing_ok=True
                            )

                    for _, r in rows.iterrows():
                        window_id = r["window_id"]

                        records.append({
                            "split": split,
                            "class": label,
                            "label": int(r["label"]),
                            "identity": r["identity"],
                            "video_id": video_id,
                            "window_id": window_id,
                            "window_idx": int(r["window_idx"]),
                            "start_frame": int(r["start_frame"]),
                            "end_frame": int(r["end_frame"]),
                            "original_lip_path": (
                                original_paths[window_id]
                            ),
                            "h264_lip_path": (
                                h264_paths.get(window_id, "")
                            )
                        })

                    print(
                        f"  Original: {n_new} new, "
                        f"{n_skip} cached"
                    )
                    print(f"  Missing lip detections: {failed}")

                except Exception as e:
                    failures.append({
                        "split": split,
                        "class": label,
                        "video_id": video_id,
                        "video_path": str(video_path),
                        "error": str(e)
                    })

                    print(f"  FAILED: {e}")

                pd.DataFrame(records).to_csv(
                    output_root / "lip_manifest.csv",
                    index=False
                )

                pd.DataFrame(failures).to_csv(
                    output_root / "lip_failures.csv",
                    index=False
                )

    print("\nExtraction complete")
    print("Lip windows:", len(records))
    print("Failed videos:", len(failures))
    print("Manifest:", output_root / "lip_manifest.csv")


if __name__ == "__main__":
    main()