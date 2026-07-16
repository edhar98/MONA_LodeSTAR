from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from services.tdms_cache import get_images


def normalize_tdms_frame(frame: np.ndarray, normalize: bool) -> np.ndarray:
    if normalize:
        fmin, fmax = float(frame.min()), float(frame.max())
        if fmax > fmin:
            return ((frame.astype(np.float32) - fmin) / (fmax - fmin + 1e-8) * 255).astype(np.uint8)
        return np.zeros_like(frame, dtype=np.uint8)
    if frame.dtype == np.uint16:
        return (frame >> 8).astype(np.uint8)
    return np.clip(frame, 0, 255).astype(np.uint8)


def normalize_tdms_stack(images: np.ndarray, normalize: bool) -> np.ndarray:
    if normalize:
        flat = images.reshape(images.shape[0], -1).astype(np.float32)
        fmin = flat.min(axis=1)
        fmax = flat.max(axis=1)
        scale = fmax - fmin
        out = np.zeros(images.shape, dtype=np.uint8)
        valid = scale > 0
        if np.any(valid):
            scaled = (images[valid].astype(np.float32) - fmin[valid, None, None]) / (scale[valid, None, None] + 1e-8) * 255.0
            out[valid] = np.clip(scaled, 0, 255).astype(np.uint8)
        return out
    if images.dtype == np.uint16:
        return (images >> 8).astype(np.uint8)
    return np.clip(images, 0, 255).astype(np.uint8)


def extract_frame(file_path: Path, file_info: dict, index: int):
    if file_info["type"] == "tdms":
        normalize = file_info.get("tdms_settings", {}).get("normalize", True)
        images = get_images(str(file_path))
        if images is None:
            raise ValueError(f"No image data in {file_path}")
        if index < 0 or index >= len(images):
            raise ValueError(f"Frame {index} out of range")
        frame = normalize_tdms_frame(images[index], normalize)
        return Image.fromarray(frame), len(images)
    img = Image.open(file_path)
    if img.mode != "L":
        img = img.convert("L")
    return img, 1


def load_file_stack(file_info: dict) -> np.ndarray:
    path = Path(file_info["path"])
    if file_info["type"] == "tdms":
        normalize = file_info.get("tdms_settings", {}).get("normalize", True)
        images = get_images(str(path))
        if images is None:
            raise ValueError(f"No image data in {path}")
        stack = normalize_tdms_stack(images, normalize)
        file_info["frame_count"] = int(stack.shape[0])
        file_info["width"] = int(stack.shape[2])
        file_info["height"] = int(stack.shape[1])
        return stack
    img = Image.open(path)
    if img.mode != "L":
        img = img.convert("L")
    arr = np.array(img, dtype=np.uint8)
    file_info["frame_count"] = 1
    file_info["width"] = int(arr.shape[1])
    file_info["height"] = int(arr.shape[0])
    return arr[np.newaxis, ...]


def build_session_frame_getter(file_infos: List[dict], frames_needed: Optional[set] = None):
    from concurrent.futures import ThreadPoolExecutor

    needed = {int(f) for f in frames_needed} if frames_needed else None
    max_needed = max(needed) if needed else None
    stacks: Dict[int, np.ndarray] = {}
    ranges = []
    offset = 0
    to_load = []

    for i, file_info in enumerate(file_infos):
        if max_needed is not None and offset > max_needed:
            break

        known = int(file_info.get("frame_count") or 0)
        if file_info["type"] != "tdms":
            known = 1
            file_info["frame_count"] = 1

        if known > 1:
            start, end = offset, offset + known
            ranges.append((start, end, i))
            if needed is None or any(start <= f < end for f in needed):
                to_load.append(i)
            offset = end
            continue

        stack = load_file_stack(file_info)
        n = int(stack.shape[0])
        start, end = offset, offset + n
        ranges.append((start, end, i))
        if needed is None or any(start <= f < end for f in needed):
            stacks[i] = stack
        offset = end

    missing = [i for i in to_load if i not in stacks]
    if len(missing) == 1:
        stacks[missing[0]] = load_file_stack(file_infos[missing[0]])
    elif missing:
        def _load_one(i: int):
            stacks[i] = load_file_stack(file_infos[i])
        with ThreadPoolExecutor(max_workers=min(4, len(missing))) as pool:
            list(pool.map(_load_one, missing))

    def get_frame(global_idx: int):
        idx = int(global_idx)
        for start, end, i in ranges:
            if start <= idx < end:
                stack = stacks.get(i)
                if stack is None:
                    return None
                return stack[idx - start]
        return None

    get_frame.total_frames = offset
    get_frame.loaded_files = len(stacks)
    return get_frame


def parse_tdms_info(file_path: Path, file_info: dict) -> dict:
    try:
        images = get_images(str(file_path))
        if images is not None:
            file_info["frame_count"] = images.shape[0]
            file_info["width"] = images.shape[2]
            file_info["height"] = images.shape[1]
            file_info["dtype"] = str(images.dtype)
        else:
            file_info["error"] = "Could not auto-detect image dimensions"
    except Exception as e:
        file_info["error"] = str(e)
    return file_info
