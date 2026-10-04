import threading
import stat
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
    """Metadata freshness, not a cryptographic content identity.

    Inode/ctime detect replacement and ordinary edits even if mtime is restored.
    Deliberately identical metadata is outside this lightweight cache contract.
    """
    p = Path(path).resolve()
    try:
        st = p.stat()
    except FileNotFoundError as exc:
        raise ValueError("TDMS source is missing or was deleted; select an existing file") from exc
    except PermissionError as exc:
        raise ValueError("TDMS source is no longer readable") from exc
    if not stat.S_ISREG(st.st_mode):
        raise ValueError("TDMS source is no longer a regular file")
    return (str(p), st.st_dev, st.st_ino, st.st_mtime_ns, st.st_ctime_ns, st.st_size)


def _discard_versions(key: Tuple):
    # External files remain editable. Retain only the current version of a path.
    for store in (_explorers, _images):
        for old in list(store):
            if old[0] == key[0] and old != key:
                del store[old]


def _check_unchanged(path: str, key: Tuple):
    try:
        if _cache_key(path) != key:
            raise ValueError("TDMS source changed while reading; retry after editing finishes")
    except Exception:
        _explorers.pop(key, None)
        _images.pop(key, None)
        raise


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
    with _lock:
        key = _cache_key(path)
        _discard_versions(key)
        if key in _explorers:
            _explorers.move_to_end(key)
            return _explorers[key]
        explorer = TDMSFileExplorer(key[0])
        _check_unchanged(path, key)
        _explorers[key] = explorer
        _trim(_explorers)
        return explorer


def get_images(path: str) -> Optional[np.ndarray]:
    with _lock:
        key = _cache_key(path)
        _discard_versions(key)
        if key in _images:
            _images.move_to_end(key)
            return _images[key]
        if key in _explorers:
            explorer = _explorers[key]
            _explorers.move_to_end(key)
        else:
            explorer = TDMSFileExplorer(key[0])
            _explorers[key] = explorer
            _trim(_explorers)
        try:
            images = explorer.extract_images()
        except Exception:
            _explorers.pop(key, None)
            _images.pop(key, None)
            raise
        _check_unchanged(path, key)
        _images[key] = images
        _trim(_images)
        return images
