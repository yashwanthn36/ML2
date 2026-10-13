"""Media decoding primitives shared by paired audio-video adapters."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


def read_video_frames(path: str | Path, *, start: float | None, end: float | None,
                      num_frames: int, image_size: int) -> torch.Tensor:
    """Decode evenly spaced RGB frames as float ``[T,C,H,W]`` in [0,1]."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Could not open video: {path}")
    fps, frame_count = capture.get(cv2.CAP_PROP_FPS), capture.get(cv2.CAP_PROP_FRAME_COUNT)
    if start is None and end is None:
        if fps <= 0 or frame_count <= 0:
            capture.release()
            raise OSError(f"Could not determine video duration for full_clip sampling: {path}")
        start, end = 0.0, frame_count / fps
    else:
        start, end = float(start), float(end)
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        capture.release()
        raise ValueError(f"Invalid video interval [{start}, {end}) for {path}")
    duration = frame_count / fps if fps > 0 else max(end, 0.1)
    start = max(0.0, min(start, max(duration - 0.01, 0.0)))
    end = max(start + 0.01, min(end, duration))
    times = np.linspace(start, end, num_frames, endpoint=False, dtype=np.float64)
    times += max((end - start) / (2 * num_frames), 0.005)
    times = np.minimum(times, max(duration - 0.01, 0.0))
    frames, last_good = [], None
    try:
        for second in times:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(second * 1000.0))
            ok, frame = capture.read()
            if ok:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
                last_good = frame
            elif last_good is not None:
                frame = last_good
            else:
                continue
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise OSError(f"Could not decode any frames from {path}")
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return torch.from_numpy(np.stack(frames[:num_frames]).astype(np.float32) / 255).permute(0, 3, 1, 2).contiguous()
