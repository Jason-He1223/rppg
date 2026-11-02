#!/usr/bin/env python3

import sys
import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch


# ============================================================
# Existing preprocessing / RhythmFormer utilities
# ============================================================

PROJECT_ROOT = Path(
    "/iacl/pg25/zijie/rppg/rPPG-TalkingFace-Detector"
)

SRC_ROOT = PROJECT_ROOT / "src"

sys.path.insert(
    0,
    str(SRC_ROOT)
)

import extract_waveforms as ew


RHYTHMFORMER_REPO = Path(
    "/iacl/pg25/zijie/rppg/RhythmFormer"
)


# ============================================================
# Defaults
# ============================================================

CLIP_LENGTH = 160
STRIDE = 80

SIZE = 128
EXPAND_COEF = 1.5
MIN_NATIVE_FRAMES = 60


# ============================================================
# RhythmFormer latent extractor
# ============================================================

class RhythmFormerLatent(torch.nn.Module):
    """
    Frozen RhythmFormer returning the spatial latent before
    RhythmFormer's spatial averaging.

    Input:
        [B, 160, 3, 128, 128]

    Output:
        [B, 64, 160, 8, 8]
    """

    def __init__(self, rhythmformer):

        super().__init__()

        self.model = rhythmformer

        for p in self.model.parameters():
            p.requires_grad = False

        self.model.eval()

    @torch.no_grad()
    def forward(self, x):

        N, D, C, H, W = x.shape

        x = self.model.Fusion_Stem(
            x
        )

        x = (
            x.view(
                N,
                D,
                64,
                H // 4,
                W // 4,
            )
            .permute(
                0,
                2,
                1,
                3,
                4,
            )
            .contiguous()
        )

        x = self.model.patch_embedding(
            x
        )

        for stage in self.model.stages:
            x = stage(x)

        return x


# ============================================================
# H.264 compression
# ============================================================

def get_video_bitrate(video_path):

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=bit_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )

    bitrate = int(
        result.stdout.strip()
    )

    if bitrate <= 0:

        raise RuntimeError(
            f"Invalid bitrate for "
            f"{video_path}: {bitrate}"
        )

    return bitrate


def compress_video(
    input_path,
    output_path,
    bitrate_ratio=0.70,
):

    original_bitrate = (
        get_video_bitrate(
            input_path
        )
    )

    target_bitrate = max(
        int(
            original_bitrate
            * bitrate_ratio
        ),
        50000,
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-c:v",
        "libopenh264",
        "-b:v",
        str(target_bitrate),
        "-an",
        str(output_path),
    ]

    subprocess.run(
        cmd,
        check=True,
    )

    return (
        original_bitrate,
        target_bitrate,
    )


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        default=(
            "/iacl/pg25/zijie/rppg/"
            "rPPG-TalkingFace-Detector/data"
        ),
    )

    parser.add_argument(
        "--repo-dir",
        default=str(
            RHYTHMFORMER_REPO
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help=(
            "Number of 160-frame windows "
            "processed simultaneously by "
            "RhythmFormer."
        ),
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=80,
        help=(
            "Sliding-window stride in frames."
        ),
    )

    parser.add_argument(
        "--bitrate-ratio",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "val",
            "test",
            "all",
        ],
        default="all",
    )

    parser.add_argument(
        "--skip-h264",
        action="store_true",
        help=(
            "Do not generate training H264 "
            "features."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Process only N unique videos; "
            "0 = all."
        ),
    )

    return parser.parse_args()


# ============================================================
# Source-video mapping
# ============================================================

def strip_real_window_id(
    video_id
):
    """
    Existing real samples look like:

        id0_0000_w0
        id0_0000_w1
        id0_0000_w2

    All correspond to original video:

        id0_0000.mp4
    """

    if "_w" in video_id:

        base, suffix = (
            video_id.rsplit(
                "_w",
                1,
            )
        )

        if suffix.isdigit():
            return base

    return video_id


def get_source_video(
    row,
    video_root,
):

    label = str(
        row["class"]
    )

    video_id = str(
        row["video_id"]
    )

    if label == "real":

        original_id = (
            strip_real_window_id(
                video_id
            )
        )

        video_path = (
            video_root
            / "real"
            / f"{original_id}.mp4"
        )

    elif label == "fake":

        if "__" not in video_id:

            raise ValueError(
                f"Invalid fake ID: "
                f"{video_id}"
            )

        method, filename = (
            video_id.split(
                "__",
                1,
            )
        )

        original_id = (
            video_id
        )

        video_path = (
            video_root
            / "fake"
            / method
            / f"{filename}.mp4"
        )

    else:

        raise ValueError(
            f"Unknown class: {label}"
        )

    return (
        original_id,
        video_path,
    )


# ============================================================
# Build one row per ORIGINAL VIDEO
# ============================================================

def build_video_manifest(
    split_df,
    video_root,
):

    videos = {}

    for _, row in (
        split_df.iterrows()
    ):

        (
            original_id,
            video_path,
        ) = get_source_video(
            row,
            video_root,
        )

        key = (
            str(row["split"]),
            str(row["class"]),
            original_id,
        )

        if key in videos:
            continue

        videos[key] = {
            "split":
                str(row["split"]),

            "class":
                str(row["class"]),

            "label":
                int(row["label"]),

            "identity":
                row.get(
                    "identity",
                    "",
                ),

            "video_id":
                original_id,

            "video_path":
                str(video_path),
        }

    return pd.DataFrame(
        list(
            videos.values()
        )
    )


# ============================================================
# Get video frame count
# ============================================================

def get_frame_count(
    video_path
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open video: "
            f"{video_path}"
        )

    n_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    cap.release()

    return n_frames


# ============================================================
# Sliding-window positions
# ============================================================

def get_window_starts(
    n_frames,
    stride,
):

    if n_frames < MIN_NATIVE_FRAMES:

        return []

    # Preserve existing behavior for
    # videos shorter than 160:
    # ew.video_to_tensor will interpolate
    # the video to 160 frames.
    if n_frames < CLIP_LENGTH:

        return [0]

    starts = list(
        range(
            0,
            n_frames
            - CLIP_LENGTH
            + 1,
            stride,
        )
    )

    # Always include the final possible
    # complete 160-frame window.
    final_start = (
        n_frames
        - CLIP_LENGTH
    )

    if starts[-1] != final_start:

        starts.append(
            final_start
        )

    return starts


# ============================================================
# Run one RhythmFormer batch
# ============================================================

@torch.inference_mode()
def run_rhythmformer_batch(
    tensors,
    extractor,
    rhythmformer,
    device,
):
    """
    tensors:
        list of [1,160,3,128,128]

    Returns:
        latents:
            [B,64,160,8,8]

        waveforms:
            [B,160]
    """

    batch = torch.cat(
        tensors,
        dim=0,
    )

    batch = batch.to(
        device,
        non_blocking=True,
    )

    # ========================================================
    # Spatial latent
    # ========================================================

    latent = extractor(
        batch
    )

    if (
        latent.ndim != 5
        or tuple(
            latent.shape[1:]
        )
        != (
            64,
            160,
            8,
            8,
        )
    ):

        raise ValueError(
            "Unexpected latent shape: "
            f"{tuple(latent.shape)}"
        )

    # ========================================================
    # IMPORTANT:
    #
    # Preserve behavior of your existing H264 extraction:
    #
    # latent -> float16 -> float32 -> spatial mean
    # -> ConvBlockLast
    #
    # This means the stored latent and waveform generation
    # follow the same numerical path as your current dataset.
    # ========================================================

    latent_np = (
        latent
        .detach()
        .cpu()
        .numpy()
        .astype(np.float16)
    )

    latent_for_waveform = (
        torch.from_numpy(
            latent_np.astype(
                np.float32
            )
        )
        .to(
            device,
            non_blocking=True,
        )
    )

    spatial_mean = (
        latent_for_waveform.mean(
            dim=(-1, -2)
        )
    )

    waveform = (
        rhythmformer.ConvBlockLast(
            spatial_mean
        )
    )

    waveform_np = (
        waveform
        .detach()
        .float()
        .cpu()
        .numpy()
        .reshape(
            latent_np.shape[0],
            -1,
        )
        .astype(np.float32)
    )

    if (
        waveform_np.shape[1]
        != CLIP_LENGTH
    ):

        raise ValueError(
            "Unexpected waveform shape: "
            f"{waveform_np.shape}"
        )

    return (
        latent_np,
        waveform_np,
    )


# ============================================================
# Extract ALL windows for one video version
# ============================================================

def extract_video_windows(
    video_path,
    starts,
    video_id,
    label,
    latent_root,
    waveform_root,
    face_detector,
    extractor,
    rhythmformer,
    device,
    batch_size,
):

    records = []

    pending_tensors = []
    pending_records = []

    n_new = 0
    n_skip = 0

    # ========================================================
    # Flush current batch
    # ========================================================

    def flush_batch():

        nonlocal n_new

        if not pending_tensors:
            return

        print(
            f"      GPU batch: "
            f"{len(pending_tensors)}"
        )

        (
            latents,
            waveforms,
        ) = run_rhythmformer_batch(
            pending_tensors,
            extractor,
            rhythmformer,
            device,
        )

        for i, rec in enumerate(
            pending_records
        ):

            np.save(
                rec["latent_path"],
                latents[i],
            )

            np.save(
                rec["waveform_path"],
                waveforms[i],
            )

            n_new += 1

        pending_tensors.clear()
        pending_records.clear()

    # ========================================================
    # Prepare each window
    # ========================================================

    for window_idx, start_frame in enumerate(
        starts
    ):

        window_id = (
            f"{video_id}"
            f"__mw{window_idx:03d}"
        )

        latent_path = (
            latent_root
            / label
            / f"{window_id}.npy"
        )

        waveform_path = (
            waveform_root
            / label
            / f"{window_id}.npy"
        )

        rec = {
            "window_id":
                window_id,

            "window_idx":
                window_idx,

            "start_frame":
                start_frame,

            "latent_path":
                latent_path,

            "waveform_path":
                waveform_path,
        }

        records.append(
            rec
        )

        # ====================================================
        # Resume
        # ====================================================

        if (
            latent_path.exists()
            and waveform_path.exists()
        ):

            n_skip += 1

            continue

        # ====================================================
        # EXACT SAME preprocessing function as your existing
        # extraction.
        #
        # Only difference:
        # start_frame changes for each temporal window.
        # ====================================================

        tensor, fps = (
            ew.video_to_tensor(
                video_path,
                face_detector,
                n_frames=CLIP_LENGTH,
                size=SIZE,
                expand_coef=EXPAND_COEF,
                min_native_frames=(
                    MIN_NATIVE_FRAMES
                ),
                start_frame=start_frame,
            )
        )

        # Keep on CPU until GPU batch is ready.
        tensor = tensor.cpu()

        pending_tensors.append(
            tensor
        )

        pending_records.append(
            rec
        )

        if (
            len(
                pending_tensors
            )
            >= batch_size
        ):

            flush_batch()

    # ========================================================
    # Final incomplete batch
    # ========================================================

    flush_batch()

    return (
        records,
        n_new,
        n_skip,
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    if args.batch_size < 1:

        raise ValueError(
            "--batch-size must be >= 1"
        )

    if args.stride < 1:

        raise ValueError(
            "--stride must be >= 1"
        )

    data_root = Path(
        args.data_root
    )

    video_root = (
        data_root
        / "videos"
    )

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "============================================"
    )

    print(
        "RhythmFormer multi-window preprocessing"
    )

    print(
        "============================================"
    )

    print(
        "Device:",
        device,
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

        gpu_memory = (
            torch.cuda.get_device_properties(
                0
            ).total_memory
            / (1024 ** 3)
        )

        print(
            f"GPU memory: "
            f"{gpu_memory:.1f} GB"
        )

    print(
        "Batch size:",
        args.batch_size,
    )

    print(
        "Window length:",
        CLIP_LENGTH,
    )

    print(
        "Stride:",
        args.stride,
    )

    print(
        "Split:",
        args.split,
    )

    print(
        "H264 training augmentation:",
        not args.skip_h264,
    )

    print(
        "H264 bitrate ratio:",
        args.bitrate_ratio,
    )

    # ========================================================
    # Load pretrained RhythmFormer
    #
    # IMPORTANT:
    # Reuse the exact known-working loader from
    # extract_waveforms.py.
    # ========================================================

    print(
        "\nLoading pretrained RhythmFormer..."
    )

    rhythmformer = (
        ew.load_rhythmformer(
            args.repo_dir,
            device,
        )
    )

    extractor = (
        RhythmFormerLatent(
            rhythmformer
        )
        .to(device)
        .eval()
    )

    # ========================================================
    # Face detector
    # ========================================================

    print(
        "Loading face detector..."
    )

    face_detector = (
        ew.build_face_detector(
            data_root
            / "blaze_face_short_range.tflite"
        )
    )

    # ========================================================
    # Original dataset split
    # ========================================================

    split_path = (
        data_root
        / "dataset_split.csv"
    )

    if not split_path.exists():

        raise FileNotFoundError(
            split_path
        )

    df = pd.read_csv(
        split_path
    )

    if args.split != "all":

        df = (
            df[
                df["split"]
                == args.split
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

    # ========================================================
    # Collapse existing real windows into original videos
    # ========================================================

    videos = (
        build_video_manifest(
            df,
            video_root,
        )
    )

    if args.limit > 0:

        videos = (
            videos.iloc[
                :args.limit
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

    print(
        f"\nUnique original videos: "
        f"{len(videos)}"
    )

    if len(videos) > 0:

        print(
            videos.groupby(
                [
                    "split",
                    "class",
                ]
            ).size()
        )

    # ========================================================
    # Output directories
    # ========================================================

    ratio_tag = int(
        args.bitrate_ratio
        * 100
    )

    original_latent_root = (
        data_root
        / "rhythmformer_spatial_latents_multiwindow"
    )

    original_waveform_root = (
        data_root
        / "waveforms_multiwindow"
    )

    h264_latent_root = (
        data_root
        / (
            "rhythmformer_spatial_latents_"
            f"multiwindow_h264_{ratio_tag}"
        )
    )

    h264_waveform_root = (
        data_root
        / (
            "waveforms_"
            f"multiwindow_h264_{ratio_tag}"
        )
    )

    for root in [
        original_latent_root,
        original_waveform_root,
        h264_latent_root,
        h264_waveform_root,
    ]:

        for label in [
            "real",
            "fake",
        ]:

            (
                root
                / label
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

    manifest_path = (
        data_root
        / "multiwindow_manifest.csv"
    )

    failure_path = (
        data_root
        / "multiwindow_failures.csv"
    )

    # ========================================================
    # Preserve manifest rows for splits we're not processing
    # ========================================================

    manifest_rows = []

    if (
        manifest_path.exists()
        and args.split != "all"
    ):

        old_manifest = (
            pd.read_csv(
                manifest_path
            )
        )

        old_manifest = (
            old_manifest[
                old_manifest["split"]
                != args.split
            ]
        )

        manifest_rows.extend(
            old_manifest.to_dict(
                "records"
            )
        )

    failures = []

    total_original_new = 0
    total_original_skip = 0

    total_h264_new = 0
    total_h264_skip = 0

    # ========================================================
    # Temporary H264 directory
    # ========================================================

    with tempfile.TemporaryDirectory(
        prefix="rppg_multiwindow_h264_"
    ) as temp_dir:

        temp_dir = Path(
            temp_dir
        )

        # ====================================================
        # Process original videos
        # ====================================================

        for video_number, (_, row) in enumerate(
            videos.iterrows(),
            1,
        ):

            split = str(
                row["split"]
            )

            label = str(
                row["class"]
            )

            video_id = str(
                row["video_id"]
            )

            video_path = Path(
                row["video_path"]
            )

            print(
                "\n"
                "============================================"
            )

            print(
                f"[{video_number}/"
                f"{len(videos)}] "
                f"{split} | "
                f"{label}"
            )

            print(
                video_id
            )

            print(
                video_path
            )

            compressed_path = None

            try:

                if not video_path.exists():

                    raise FileNotFoundError(
                        video_path
                    )

                # =============================================
                # Window positions
                # =============================================

                n_frames = (
                    get_frame_count(
                        video_path
                    )
                )

                starts = (
                    get_window_starts(
                        n_frames,
                        args.stride,
                    )
                )

                if not starts:

                    raise RuntimeError(
                        f"Video has only "
                        f"{n_frames} frames"
                    )

                print(
                    f"  Native frames: "
                    f"{n_frames}"
                )

                print(
                    f"  Windows: "
                    f"{len(starts)}"
                )

                print(
                    f"  Starts: "
                    f"{starts}"
                )

                # =============================================
                # ORIGINAL
                # =============================================

                print(
                    "  Extracting original..."
                )

                (
                    original_records,
                    n_new,
                    n_skip,
                ) = extract_video_windows(
                    video_path=video_path,
                    starts=starts,
                    video_id=video_id,
                    label=label,
                    latent_root=(
                        original_latent_root
                    ),
                    waveform_root=(
                        original_waveform_root
                    ),
                    face_detector=(
                        face_detector
                    ),
                    extractor=extractor,
                    rhythmformer=(
                        rhythmformer
                    ),
                    device=device,
                    batch_size=(
                        args.batch_size
                    ),
                )

                total_original_new += (
                    n_new
                )

                total_original_skip += (
                    n_skip
                )

                print(
                    f"  Original: "
                    f"{n_new} new, "
                    f"{n_skip} cached"
                )

                # =============================================
                # H264
                #
                # IMPORTANT:
                # preserve existing design:
                # compression augmentation is TRAINING ONLY.
                # =============================================

                h264_records = None

                original_bitrate = None
                target_bitrate = None

                if (
                    split == "train"
                    and not args.skip_h264
                ):

                    safe_id = (
                        video_id
                        .replace(
                            "/",
                            "_",
                        )
                        .replace(
                            "\\",
                            "_",
                        )
                    )

                    compressed_path = (
                        temp_dir
                        / (
                            f"{safe_id}"
                            "_h264.mp4"
                        )
                    )

                    print(
                        "  Compressing H264..."
                    )

                    (
                        original_bitrate,
                        target_bitrate,
                    ) = compress_video(
                        video_path,
                        compressed_path,
                        bitrate_ratio=(
                            args.bitrate_ratio
                        ),
                    )

                    print(
                        f"  Bitrate: "
                        f"{original_bitrate / 1000:.0f} "
                        f"-> "
                        f"{target_bitrate / 1000:.0f} "
                        f"kbps"
                    )

                    # Use the SAME temporal
                    # start frames as original.
                    compressed_n_frames = (
                        get_frame_count(
                            compressed_path
                        )
                    )

                    if (
                        compressed_n_frames
                        < MIN_NATIVE_FRAMES
                    ):

                        raise RuntimeError(
                            "Compressed video became "
                            f"too short: "
                            f"{compressed_n_frames}"
                        )

                    print(
                        "  Extracting H264..."
                    )

                    (
                        h264_records,
                        h_new,
                        h_skip,
                    ) = extract_video_windows(
                        video_path=(
                            compressed_path
                        ),
                        starts=starts,
                        video_id=video_id,
                        label=label,
                        latent_root=(
                            h264_latent_root
                        ),
                        waveform_root=(
                            h264_waveform_root
                        ),
                        face_detector=(
                            face_detector
                        ),
                        extractor=extractor,
                        rhythmformer=(
                            rhythmformer
                        ),
                        device=device,
                        batch_size=(
                            args.batch_size
                        ),
                    )

                    total_h264_new += (
                        h_new
                    )

                    total_h264_skip += (
                        h_skip
                    )

                    print(
                        f"  H264: "
                        f"{h_new} new, "
                        f"{h_skip} cached"
                    )

                    compressed_path.unlink(
                        missing_ok=True
                    )

                    compressed_path = None

                # =============================================
                # Manifest
                # =============================================

                h264_by_idx = {}

                if h264_records is not None:

                    h264_by_idx = {
                        rec["window_idx"]:
                            rec
                        for rec
                        in h264_records
                    }

                for rec in original_records:

                    h_rec = (
                        h264_by_idx.get(
                            rec[
                                "window_idx"
                            ]
                        )
                    )

                    manifest_rows.append({
                        "split":
                            split,

                        "class":
                            label,

                        "label":
                            int(
                                row["label"]
                            ),

                        "identity":
                            row["identity"],

                        # =====================================
                        # Grouping key:
                        # one ORIGINAL video
                        # =====================================
                        "video_id":
                            video_id,

                        # =====================================
                        # Individual temporal window
                        # =====================================
                        "window_id":
                            rec[
                                "window_id"
                            ],

                        "window_idx":
                            rec[
                                "window_idx"
                            ],

                        "start_frame":
                            rec[
                                "start_frame"
                            ],

                        "end_frame":
                            min(
                                rec[
                                    "start_frame"
                                ]
                                + CLIP_LENGTH
                                - 1,
                                n_frames - 1,
                            ),

                        "n_video_frames":
                            n_frames,

                        "original_latent_path":
                            str(
                                rec[
                                    "latent_path"
                                ]
                            ),

                        "original_waveform_path":
                            str(
                                rec[
                                    "waveform_path"
                                ]
                            ),

                        "h264_latent_path":
                            (
                                str(
                                    h_rec[
                                        "latent_path"
                                    ]
                                )
                                if h_rec
                                is not None
                                else ""
                            ),

                        "h264_waveform_path":
                            (
                                str(
                                    h_rec[
                                        "waveform_path"
                                    ]
                                )
                                if h_rec
                                is not None
                                else ""
                            ),

                        "original_bitrate":
                            original_bitrate,

                        "h264_target_bitrate":
                            target_bitrate,
                    })

            except Exception as e:

                print(
                    "\nFAILED:"
                )

                print(
                    f"  ID:    "
                    f"{video_id}"
                )

                print(
                    f"  Video: "
                    f"{video_path}"
                )

                print(
                    f"  Error: "
                    f"{e}"
                )

                failures.append({
                    "split":
                        split,

                    "class":
                        label,

                    "video_id":
                        video_id,

                    "video_path":
                        str(
                            video_path
                        ),

                    "error":
                        str(e),
                })

                if (
                    compressed_path
                    is not None
                ):

                    try:

                        compressed_path.unlink(
                            missing_ok=True
                        )

                    except Exception:
                        pass

            # =============================================
            # Save incrementally after every video
            # =============================================

            pd.DataFrame(
                manifest_rows
            ).to_csv(
                manifest_path,
                index=False,
            )

            pd.DataFrame(
                failures
            ).to_csv(
                failure_path,
                index=False,
            )

            print(
                "\n  Running totals:"
            )

            print(
                f"    original new: "
                f"{total_original_new}"
            )

            print(
                f"    original cached: "
                f"{total_original_skip}"
            )

            print(
                f"    H264 new: "
                f"{total_h264_new}"
            )

            print(
                f"    H264 cached: "
                f"{total_h264_skip}"
            )

            print(
                f"    failures: "
                f"{len(failures)}"
            )

    # ========================================================
    # Final summary
    # ========================================================

    print(
        "\n"
        "============================================"
    )

    print(
        "MULTI-WINDOW EXTRACTION COMPLETE"
    )

    print(
        "============================================"
    )

    print(
        "Original new:",
        total_original_new,
    )

    print(
        "Original cached:",
        total_original_skip,
    )

    print(
        "H264 new:",
        total_h264_new,
    )

    print(
        "H264 cached:",
        total_h264_skip,
    )

    print(
        "Failed videos:",
        len(failures),
    )

    print(
        "\nManifest:"
    )

    print(
        manifest_path
    )

    print(
        "\nOriginal latents:"
    )

    print(
        original_latent_root
    )

    print(
        "\nOriginal waveforms:"
    )

    print(
        original_waveform_root
    )

    print(
        "\nH264 training latents:"
    )

    print(
        h264_latent_root
    )

    print(
        "\nH264 training waveforms:"
    )

    print(
        h264_waveform_root
    )

    # ========================================================
    # Manifest statistics
    # ========================================================

    if manifest_path.exists():

        manifest = pd.read_csv(
            manifest_path
        )

        if len(manifest) > 0:

            print(
                "\nWindows by split/class:"
            )

            print(
                manifest.groupby(
                    [
                        "split",
                        "class",
                    ]
                ).size()
            )

            print(
                "\nUnique videos by split/class:"
            )

            print(
                manifest.groupby(
                    [
                        "split",
                        "class",
                    ]
                )[
                    "video_id"
                ].nunique()
            )

            window_counts = (
                manifest.groupby(
                    [
                        "split",
                        "class",
                        "video_id",
                    ]
                )
                .size()
            )

            print(
                "\nMean windows/video:"
            )

            print(
                window_counts.groupby(
                    [
                        "split",
                        "class",
                    ]
                )
                .mean()
                .round(2)
            )

            print(
                "\nMin windows/video:"
            )

            print(
                window_counts.groupby(
                    [
                        "split",
                        "class",
                    ]
                ).min()
            )

            print(
                "\nMax windows/video:"
            )

            print(
                window_counts.groupby(
                    [
                        "split",
                        "class",
                    ]
                ).max()
            )


if __name__ == "__main__":
    main()
