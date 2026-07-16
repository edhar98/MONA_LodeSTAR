import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from tdms_explorer import TDMSFileExplorer

_lock = threading.Lock()
_MAX_ENTRIES = 4
_explorers: "OrderedDict[Tuple, TDMSFileExplorer]" = OrderedDict()
_images: "OrderedDict[Tuple, Optional[np.ndarray]]" = OrderedDict()


def _cache_key(path: str) -> Tuple:
    p = Path(path).resolve()
    st = p.stat()
    return (str(p), int(st.st_mtime_ns), int(st.st_size))


def _trim(store: OrderedDict):
    while len(store) > _MAX_ENTRIES:
        store.popitem(last=False)


def invalidate(path: Optional[str] = None):
    with _lock:
        if path is None:
            _explorers.clear()
            _images.clear()
            return
        p = str(Path(path).resolve())
        for store in (_explorers, _images):
            dead = [k for k in store if k[0] == p]
            for k in dead:
                store.pop(k, None)


def get_explorer(path: str) -> TDMSFileExplorer:
    key = _cache_key(path)
    with _lock:
        if key in _explorers:
            _explorers.move_to_end(key)
            return _explorers[key]
        explorer = TDMSFileExplorer(str(Path(path).resolve()))
        _explorers[key] = explorer
        _trim(_explorers)
        return explorer


def get_images(path: str) -> Optional[np.ndarray]:
    key = _cache_key(path)
    with _lock:
        if key in _images:
            _images.move_to_end(key)
            return _images[key]
        if key in _explorers:
            explorer = _explorers[key]
            _explorers.move_to_end(key)
        else:
            explorer = TDMSFileExplorer(str(Path(path).resolve()))
            _explorers[key] = explorer
            _trim(_explorers)
        images = explorer.extract_images()
        _images[key] = images
        _trim(_images)
        return images
