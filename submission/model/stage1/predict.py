"""Stage 1 ConvNeXt V2-Nano recapture inference."""

from __future__ import annotations

import random
import warnings
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .convnextv2_video import convnextv2_nano_video


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".3gp", ".3gpp", ".wmv"}
COLUMNS = ["ID", "answer"]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def _set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _videos(data_dir: Path) -> list[Path]:
    """Accept the official ``<stage>/videos`` layout or a direct video folder."""
    root = data_dir / "videos" if (data_dir / "videos").is_dir() else data_dir
    if not root.is_dir():
        raise FileNotFoundError(f"Stage 1 video directory not found: {root}")
    videos = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise FileNotFoundError(f"No Stage 1 videos found under: {root}")
    return videos


def _decode_views(
    path: Path,
    *,
    frames: int,
    size: int,
    resize_height: int,
    resize_width: int,
    spatial_crops: int,
) -> torch.Tensor:
    """Decode deterministic validation views shaped ``[V,T,3,S,S]``."""
    capture = cv2.VideoCapture(str(path))
    total = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    # Use the center of each normalized temporal bin so duration/FPS may vary.
    indices = [round(((index + 0.5) / frames) * (total - 1)) for index in range(frames)]
    decoded: list[np.ndarray] = []
    try:
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, bgr = capture.read()
            if not ok or bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            decoded.append(
                cv2.resize(rgb, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
            )
    finally:
        capture.release()
    if not decoded:
        raise ValueError(f"cannot decode video: {path.name}")
    while len(decoded) < frames:
        decoded.append(decoded[-1])

    max_top = resize_height - size
    max_left = resize_width - size
    if max_top < 0 or max_left < 0:
        raise ValueError("checkpoint resize dimensions must be at least the crop size")
    if spatial_crops == 1:
        lefts = [max_left // 2]
    elif spatial_crops == 3:
        lefts = [0, max_left // 2, max_left]
    else:
        raise ValueError("spatial_crops must be 1 or 3")
    top = max_top // 2

    output: list[torch.Tensor] = []
    for left in lefts:
        clip: list[torch.Tensor] = []
        for rgb in decoded[:frames]:
            patch = np.ascontiguousarray(rgb[top : top + size, left : left + size])
            tensor = torch.from_numpy(patch).permute(2, 0, 1).float().div_(255.0)
            clip.append((tensor - IMAGENET_MEAN) / IMAGENET_STD)
        output.append(torch.stack(clip))
    return torch.stack(output)


class Stage1Videos(Dataset):
    def __init__(self, videos: list[Path], settings: Mapping[str, int]) -> None:
        self.videos = videos
        self.settings = settings

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, index: int):
        try:
            views = _decode_views(self.videos[index], **self.settings)
            error = ""
        except Exception as exc:  # Keep one damaged file from aborting all three stages.
            views = torch.zeros(
                self.settings["spatial_crops"], self.settings["frames"], 3,
                self.settings["size"], self.settings["size"],
            )
            error = f"{type(exc).__name__}: {exc}"
        return views, index, error


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Stage 1 checkpoint not found: {path}. Download best.pt from Google Drive."
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"unsupported Stage 1 checkpoint object: {type(checkpoint)!r}")
    if checkpoint.get("backend") != "convnextv2_video" or "model" not in checkpoint:
        raise ValueError("best.pt is not the expected ConvNeXt V2 video checkpoint")
    return checkpoint


def predict(data_dir: Path, model_dir: Path) -> pd.DataFrame:
    """Return ``ID, answer`` using the domain-adapted and fine-tuned checkpoint."""
    _set_seed()
    cv2.setNumThreads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    videos = _videos(Path(data_dir))
    checkpoint = _load_checkpoint(Path(model_dir) / "best.pt")

    settings = {
        "frames": int(checkpoint.get("frames", 8)),
        "size": int(checkpoint.get("size", 224)),
        "resize_height": int(checkpoint.get("resize_short_side", 256)),
        "resize_width": int(checkpoint.get("decode_width", 633)),
        "spatial_crops": int(checkpoint.get("spatial_crops", 3)),
    }
    model = convnextv2_nano_video(
        dropout=float(checkpoint.get("dropout", 0.3)),
        drop_path_rate=float(checkpoint.get("drop_path_rate", 0.1)),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    threshold = float(checkpoint.get("threshold", 0.5))
    loader = DataLoader(
        Stage1Videos(videos, settings),
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    probabilities: list[float | None] = [None] * len(videos)
    with torch.inference_mode():
        for views, video_indices, errors in loader:
            batch, crops, clip_frames, channels, height, width = views.shape
            clips = views.reshape(
                batch * crops, clip_frames, channels, height, width
            ).to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                crop_probabilities = torch.sigmoid(model(clips))
            video_probabilities = crop_probabilities.float().reshape(batch, crops).mean(1)
            for sample_index, probability, error in zip(
                video_indices.tolist(), video_probabilities.cpu().tolist(), errors
            ):
                if error:
                    warnings.warn(
                        f"Stage 1 decode failed for {videos[sample_index].name}: {error}; "
                        "using conservative RERECORDED fallback",
                        RuntimeWarning,
                    )
                else:
                    probabilities[sample_index] = float(probability)

    rows = [
        {
            "ID": path.stem,
            "answer": "RERECORDED"
            if (probability if probability is not None else 1.0) >= threshold
            else "ORIGINAL",
        }
        for path, probability in zip(videos, probabilities)
    ]
    del model, checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pd.DataFrame(rows, columns=COLUMNS)
