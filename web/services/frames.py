from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from services.tdms_cache import get_images


def _file_signature(path: Path):
    """Identify the external file version used by a frame snapshot."""
    try:
        resolved = path.resolve(strict=True)
        st = resolved.stat()
        if not resolved.is_file():
            raise ValueError(f"Input is not a regular file: {path}")
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError(f"Input file is missing or was deleted: {path}") from exc
    return (str(resolved), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _check_file_signature(path: Path, expected):
    if _file_signature(path) != expected:
        raise ValueError(f"Input file changed while loading or using frames: {path}. Reload the input and retry.")


def _update_dimensions(file_info: dict, count: int, width: int, height: int, signature):
    file_info.update(frame_count=int(count), width=int(width), height=int(height))
    file_info["source_version"] = dict(zip(
        ("resolved_path", "device", "inode", "size", "mtime_ns", "ctime_ns"), signature))
    file_info.pop("error", None)


def normalize_tdms_frame(frame: np.ndarray, normalize: bool) -> np.ndarray:
    if normalize:
        fmin, fmax = float(frame.min()), float(frame.max())
        if fmax > fmin:
            return ((frame.astype(np.float32) - fmin) / (fmax - fmin + 1e-8) * 255).astype(np.uint8)
        return np.zeros_like(frame, dtype=np.uint8)
    if np.issubdtype(frame.dtype, np.integer):
        minimum = int(frame.min())
        maximum = int(frame.max())
        if minimum >= 0 and maximum <= 65535 and maximum > 255:
            return np.right_shift(frame, 8).astype(np.uint8)
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
    signature = _file_signature(file_path)
    if file_info["type"] == "tdms":
        normalize = file_info.get("tdms_settings", {}).get("normalize", True)
        images = get_images(str(file_path))
        if images is None:
            raise ValueError(f"No image data in {file_path}")
        if index < 0 or index >= len(images):
            raise ValueError(f"Frame {index} out of range")
        frame = normalize_tdms_frame(images[index], normalize)
        _check_file_signature(file_path, signature)
        _update_dimensions(file_info, len(images), frame.shape[1], frame.shape[0], signature)
        return Image.fromarray(frame), len(images)
    if index != 0:
        raise ValueError(f"Frame {index} out of range")
    with Image.open(file_path) as source:
        img = source.convert("L") if source.mode != "L" else source.copy()
    _check_file_signature(file_path, signature)
    _update_dimensions(file_info, 1, img.width, img.height, signature)
    return img, 1


def load_file_stack(file_info: dict) -> np.ndarray:
    path = Path(file_info["path"])
    signature = _file_signature(path)
    if file_info["type"] == "tdms":
        normalize = file_info.get("tdms_settings", {}).get("normalize", True)
        images = get_images(str(path))
        if images is None:
            raise ValueError(f"No image data in {path}")
        stack = normalize_tdms_stack(images, normalize)
        _check_file_signature(path, signature)
        _update_dimensions(file_info, stack.shape[0], stack.shape[2], stack.shape[1], signature)
        return stack
    with Image.open(path) as img:
        if img.mode != "L":
            img = img.convert("L")
        arr = np.array(img, dtype=np.uint8)
    _check_file_signature(path, signature)
    _update_dimensions(file_info, 1, arr.shape[1], arr.shape[0], signature)
    return arr[np.newaxis, ...]


def build_session_frame_getter(file_infos: List[dict], frames_needed: Optional[set] = None):
    needed = {int(f) for f in frames_needed} if frames_needed else None
    max_needed = max(needed) if needed else None
    stacks: Dict[int, np.ndarray] = {}
    ranges = []
    offset = 0
    signatures = {}

    for i, file_info in enumerate(file_infos):
        if max_needed is not None and offset > max_needed:
            break

        # External inputs may have changed since registration. Their current
        # lengths, including skipped prefix files, determine global offsets.
        path = Path(file_info["path"])
        signatures[i] = _file_signature(path)
        stack = load_file_stack(file_info)
        _check_file_signature(path, signatures[i])
        n = int(stack.shape[0])
        start, end = offset, offset + n
        ranges.append((start, end, i))
        if needed is None or any(start <= f < end for f in needed):
            stacks[i] = stack
        offset = end

    def get_frame(global_idx: int):
        idx = int(global_idx)
        for start, end, i in ranges:
            # Prefix versions define this index too; do not return silently
            # misaligned data if an earlier external stack changes afterward.
            _check_file_signature(Path(file_infos[i]["path"]), signatures[i])
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
        signature = _file_signature(file_path)
        images = get_images(str(file_path))
        _check_file_signature(file_path, signature)
        if images is not None:
            _update_dimensions(file_info, images.shape[0], images.shape[2], images.shape[1], signature)
            file_info["dtype"] = str(images.dtype)
        else:
            raise ValueError("Could not auto-detect image dimensions")
    except Exception as e:
        for key in ("frame_count", "width", "height", "dtype", "source_version"):
            file_info.pop(key, None)
        file_info["error"] = str(e)
    return file_info
