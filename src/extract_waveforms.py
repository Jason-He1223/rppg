#!/usr/bin/env python3
"""
extract_waveforms.py — Batch rPPG waveform extraction using RhythmFormer.

Walks videos/real/ and videos/fake/, extracts a 160-sample rPPG waveform 
per video, and saves each as a .npy file under waveforms/real/ or 
waveforms/fake/.

Fully resumable: already-extracted videos are skipped.
Writes a manifest.csv tracking status of every video.

Usage:
    python3 extract_waveforms.py --data-root ~/data
    python3 extract_waveforms.py --data-root ~/data --workers 4
"""

import argparse
import csv
import sys
import time
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

def parse_args():
    p = argparse.ArgumentParser(description="Batch rPPG waveform extraction")
    p.add_argument("--data-root", default=str(Path.home() / "data"))
    p.add_argument("--repo-dir", default=str(Path.home() / "RhythmFormer"))
    p.add_argument("--n-frames", type=int, default=160)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--expand-coef", type=float, default=1.5)
    p.add_argument("--min-frames", type=int, default=60)
    p.add_argument("--stride", type=int, default=60, help="Stride for sliding-window extraction on real videos (0 = disable)")
    p.add_argument("--flush-every", type=int, default=100, help="Write manifest to disk every N videos")
    p.add_argument("--workers", type=int, default=1, help="Parallel video readers (keep at 1 if GPU is the bottleneck)")
    return p.parse_args()


def load_rhythmformer(repo_dir, device):
    repo = Path(repo_dir)
    weights = repo / "PreTrainedModels" / "UBFC_cross_RhythmFormer.pth"
    assert repo.exists(), f"RhythmFormer repo not found at {repo}. Run setup.sh first."
    assert weights.exists(), f"Weights not found at {weights}"

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from neural_methods.model.RhythmFormer import RhythmFormer
    model = RhythmFormer()

    try:
        state = torch.load(weights, map_location="cpu", weights_only=True)
    except Exception:
        state = torch.load(weights, map_location="cpu", weights_only=False)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", "", 1): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing and not unexpected, \
        f"State dict mismatch — missing: {missing[:3]}, unexpected: {unexpected[:3]}"

    model = model.to(device).eval()
    print(f"RhythmFormer loaded on {device} | "
          f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    return model


def build_face_detector(model_path):

    if not Path(model_path).exists():
        url = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
               "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite")
        print(f"Downloading face detector model...")
        urllib.request.urlretrieve(url, model_path)

    detector = mp_vision.FaceDetector.create_from_options(
        mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            min_detection_confidence=0.5,
        )
    )
    return detector


def video_to_tensor(path, face_detector, n_frames=160, size=128, expand_coef=1.5, min_native_frames=60, start_frame=0):
    """Read video, detect face, crop and normalize. Returns (tensor, fps)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Could not open {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()

    n_native = len(frames)
    if n_native < min_native_frames:
        raise ValueError(f"Video too short: {n_native} frames (min {min_native_frames})")

    # Select frames for this window
    if start_frame > 0:
        # Windowed extraction: slice exact segment
        frames = frames[start_frame:start_frame + n_frames]
    elif n_native < n_frames:
        # Short video: upsample via linspace to preserve temporal phase
        idx = np.linspace(0, n_native - 1, n_frames).round().astype(int)
        frames = [frames[i] for i in idx]
    else:
        frames = frames[:n_frames]

    # Face detection on first 30 frames
    bbox = None
    for frame in frames[:30]:
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = face_detector.detect(mp_img)
        if not result.detections:
            continue
        det = max(result.detections, key=lambda d: d.categories[0].score)
        bb = det.bounding_box
        h, w = frame.shape[:2]
        cx = bb.origin_x + bb.width / 2
        cy = bb.origin_y + bb.height / 2
        side = max(bb.width, bb.height) * expand_coef
        x1 = int(max(0, cx - side / 2))
        y1 = int(max(0, cy - side / 2))
        x2 = int(min(w, cx + side / 2))
        y2 = int(min(h, cy + side / 2))
        bbox = (x1, y1, x2, y2)
        break

    if bbox is None:
        raise ValueError("No face detected in first 30 frames")

    x1, y1, x2, y2 = bbox
    crops = np.stack([
        cv2.resize(f[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_AREA)
        for f in frames
    ]).astype(np.float32)

    mean, std = crops.mean(), crops.std()
    if std < 1e-6:
        raise ValueError("Constant pixels — degenerate crop")
    crops = (crops - mean) / std

    tensor = torch.from_numpy(crops).permute(0, 3, 1, 2).unsqueeze(0).contiguous()
    return tensor, fps


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Data root: {data_root}")
    print(f"Device: {device}")

    # Paths
    video_root = data_root / "videos"
    wave_root  = data_root / "waveforms"
    (wave_root / "real").mkdir(parents=True, exist_ok=True)
    (wave_root / "fake").mkdir(parents=True, exist_ok=True)
    manifest_path = data_root / "manifest.csv"
    face_model_path = data_root / "blaze_face_short_range.tflite"

    # Load models
    rhythmformer = load_rhythmformer(args.repo_dir, device)
    face_detector = build_face_detector(face_model_path)

    # Collect all video paths
    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
    jobs = []
    for label in ("real", "fake"):
        vdir = video_root / label
        if not vdir.exists():
            print(f"WARNING: {vdir} does not exist (skipping)")
            continue
        for vp in sorted(vdir.rglob("*")):
            if not (vp.is_file() and vp.suffix.lower() in VIDEO_EXTS):
                continue
            # Prefix with parent dir name when video is in a subdirectory
            # (e.g. fake/AniTalker/id0_0000.mp4 → AniTalker__id0_0000)
            if vp.parent != vdir:
                base_id = f"{vp.parent.name}__{vp.stem}"
            else:
                base_id = vp.stem

            if label == "real" and args.stride > 0:
                # Sliding-window: one job per window for class balancing
                cap = cv2.VideoCapture(str(vp))
                n_native = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                starts = list(range(0, max(1, n_native - args.n_frames + 1), args.stride))
                for k, start in enumerate(starts):
                    video_id = f"{base_id}_w{k}"
                    out_path = wave_root / label / f"{video_id}.npy"
                    jobs.append((label, video_id, vp, out_path, start))
            else:
                out_path = wave_root / label / f"{base_id}.npy"
                jobs.append((label, base_id, vp, out_path, 0))

    print(f"\nFound {len(jobs)} total videos")

    # Load existing manifest to skip already-done videos
    done = set()
    manifest_rows = []
    if manifest_path.exists():
        with open(manifest_path) as f:
            for row in csv.DictReader(f):
                manifest_rows.append(row)
                if row["status"] == "ok":
                    done.add(row["video_id"])
        print(f"Already extracted: {len(done)} | Remaining: {len(jobs)-len(done)}")

    # Extraction
    t0 = time.time()
    n_ok, n_fail = 0, 0
    new_rows = []

    for i, (label, video_id, vp, out_path, start_frame) in enumerate(jobs):
        if video_id in done:
            continue

        try:
            if not out_path.exists():
                tensor, fps = video_to_tensor(
                    vp, face_detector,
                    n_frames=args.n_frames,
                    size=args.size,
                    expand_coef=args.expand_coef,
                    min_native_frames=args.min_frames,
                    start_frame=start_frame
                )
                tensor = tensor.to(device)
                with torch.no_grad():
                    waveform = rhythmformer(tensor).squeeze(0).cpu().numpy()
                np.save(out_path, waveform)

            new_rows.append({
                "video_id": video_id, "class": label,
                "status": "ok", "error": ""
            })
            n_ok += 1
            done.add(video_id)

        except Exception as e:
            new_rows.append({
                "video_id": video_id, "class": label,
                "status": "fail", "error": str(e)[:200]
            })
            n_fail += 1

        # Progress
        processed = n_ok + n_fail
        if processed % 50 == 0 or processed == 1:
            elapsed = time.time() - t0
            rate = processed / elapsed
            remaining = (len(jobs) - len(done)) / max(rate, 1e-6)
            print(f"  [{processed:>5}/{len(jobs)}] ok={n_ok} fail={n_fail} "
                  f"| {rate:.1f} vid/s | ETA {remaining/60:.0f} min")

        # Flush manifest periodically
        if len(new_rows) >= args.flush_every:
            manifest_rows.extend(new_rows)
            new_rows = []
            with open(manifest_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["video_id", "class", "status", "error"])
                w.writeheader()
                w.writerows(manifest_rows)

    # Final flush
    manifest_rows.extend(new_rows)
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "class", "status", "error"])
        w.writeheader()
        w.writerows(manifest_rows)

    elapsed = time.time() - t0
    print(f"\n=== Extraction complete ===")
    print(f"OK: {n_ok}")
    print(f"Failed: {n_fail}")
    print(f"Time: {elapsed/60:.1f} min")
    print(f"Manifest: {manifest_path}")
    print(f"\nNext step: python3 build_split.py --data-root {args.data_root}")


if __name__ == "__main__":
    main()
