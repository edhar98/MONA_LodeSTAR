"""Publish completed files without clobbering externally edited results."""
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def new_output(destination):
    destination = Path(destination)
    fd, name = tempfile.mkstemp(prefix=".pending-", suffix=destination.suffix,
                                dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        # Unlike replace(), link() atomically fails if the destination exists.
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
