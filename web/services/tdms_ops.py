import base64
from io import BytesIO
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image
from tdms_explorer.image_analysis import ImageAnalyzer

from services.frames import normalize_tdms_frame
from services.tdms_cache import get_explorer, get_images

_cmap_cache: Dict[str, Any] = {}


def list_structure(path: str) -> dict:
    return get_explorer(path).list_contents()


def _frame_array(path: str, index: int) -> Tuple[np.ndarray, int]:
    images = get_images(path)
    if images is None:
        raise ValueError("No image data in TDMS")
    if index < 0 or index >= images.shape[0]:
        raise ValueError(f"Frame {index} out of range")
    return images[index], int(images.shape[0])


def _analyzer(path: str, frame: int) -> ImageAnalyzer:
    arr, _ = _frame_array(path, frame)
    return ImageAnalyzer(arr)


def array_to_png_b64(arr: np.ndarray, cmap: str = "gray") -> str:
    if arr is None:
        raise ValueError("Empty image")
    img = np.asarray(arr)
    if img.dtype != np.uint8:
        if img.dtype == np.uint16:
            img = (img >> 8).astype(np.uint8)
        else:
            f = img.astype(np.float32)
            fmin, fmax = float(np.min(f)), float(np.max(f))
            if fmax > fmin:
                img = ((f - fmin) / (fmax - fmin) * 255).astype(np.uint8)
            else:
                img = np.zeros_like(f, dtype=np.uint8)
    if cmap and cmap != "gray":
        try:
            lut = _cmap_cache.get(cmap)
            if lut is None:
                import matplotlib.cm as cm
                lut = (cm.get_cmap(cmap)(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
                _cmap_cache[cmap] = lut
            pil = Image.fromarray(lut[img], mode="RGB")
        except Exception:
            pil = Image.fromarray(img, mode="L")
    else:
        pil = Image.fromarray(img, mode="L")
    buf = BytesIO()
    pil.save(buf, format="PNG", compress_level=1)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def get_frame_image(path: str, index: int, normalize: bool = True, cmap: str = "gray") -> Tuple[str, int, int, int]:
    raw, n = _frame_array(path, index)
    frame = normalize_tdms_frame(raw, normalize)
    return array_to_png_b64(frame, cmap=cmap), n, int(frame.shape[1]), int(frame.shape[0])


def get_channel_series(path: str, group: str, channel: str, max_points: int = 4000) -> Dict[str, Any]:
    explorer = get_explorer(path)
    data = explorer.get_raw_channel_data(group, channel)
    if data is None:
        raise ValueError(f"Channel not found: {group}/{channel}")
    arr = np.asarray(data).astype(np.float64).ravel()
    n = int(arr.size)
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points).astype(np.int64)
        arr = arr[idx]
        x = idx.tolist()
    else:
        x = list(range(n))
    return {
        "group": group,
        "channel": channel,
        "n_original": n,
        "x": x,
        "y": arr.tolist(),
        "min": float(np.min(arr)) if arr.size else None,
        "max": float(np.max(arr)) if arr.size else None,
        "mean": float(np.mean(arr)) if arr.size else None,
    }


def analyze_histogram(path: str, frame: int, bins: int = 256) -> dict:
    analyzer = _analyzer(path, frame)
    hist = analyzer.create_histogram(bins=bins)
    stats = analyzer.analyze_image()
    if hist is None:
        raise ValueError("Histogram failed")
    return {"histogram": _jsonable(hist), "stats": _jsonable(stats or {})}


def analyze_filter(path: str, frame: int, filter_type: str, sigma: float = 1.0) -> dict:
    analyzer = _analyzer(path, frame)
    kwargs = {"sigma": sigma} if filter_type in ("gaussian",) else {"size": max(1, int(round(sigma * 2 + 1)))}
    if filter_type == "bilateral":
        kwargs = {"sigma_color": sigma * 25, "sigma_spatial": sigma}
    result = analyzer.apply_filter(filter_type=filter_type, **kwargs)
    if result is None:
        raise ValueError("Filter failed")
    return {"image": array_to_png_b64(result), "filter_type": filter_type}


def analyze_edges(path: str, frame: int, method: str = "canny") -> dict:
    analyzer = _analyzer(path, frame)
    result = analyzer.detect_edges(method=method)
    if result is None:
        raise ValueError("Edge detection failed")
    return {"image": array_to_png_b64(result), "method": method}


def analyze_profile(path: str, frame: int, direction: str = "horizontal", position: Optional[int] = None) -> dict:
    analyzer = _analyzer(path, frame)
    result = analyzer.get_image_profile(direction=direction, position=position)
    if result is None:
        raise ValueError("Profile failed")
    return _jsonable(result)


def compare_frames(path: str, frame_a: int, frame_b: int, method: str = "difference") -> dict:
    images = get_images(path)
    if images is None:
        raise ValueError("No image data")
    for idx in (frame_a, frame_b):
        if idx < 0 or idx >= images.shape[0]:
            raise ValueError(f"Frame {idx} out of range")
    a = normalize_tdms_frame(images[frame_a], True)
    b = normalize_tdms_frame(images[frame_b], True)
    analyzer = ImageAnalyzer(images[frame_a])
    diff = analyzer.compare_images(images[frame_b], method=method)
    if diff is None:
        raise ValueError("Compare failed")
    return {
        "frame_a": array_to_png_b64(a),
        "frame_b": array_to_png_b64(b),
        "diff": array_to_png_b64(diff),
        "method": method,
        "frame_count": int(images.shape[0]),
    }


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj
