#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v pyinstaller &> /dev/null; then
    echo "PyInstaller is not installed. Installing..."
    pip install pyinstaller
fi

echo "Ensuring imageio-ffmpeg is installed for video support..."
pip install imageio-ffmpeg > /dev/null 2>&1 || echo "Warning: imageio-ffmpeg installation failed. MP4 conversion may not work."

echo "Building tdms_to_png executable..."

cat > /tmp/tdms_to_png_build.spec << EOF
# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = [
    'numpy',
    'nptdms',
    'PIL',
    'PIL.Image',
    'imageio',
    'imageio.plugins.ffmpeg',
    'multiprocessing',
]

try:
    tmp_ret = collect_all('imageio')
    datas += tmp_ret[0]
    binaries += tmp_ret[1]
    hiddenimports += tmp_ret[2]
except:
    pass

try:
    tmp_ret = collect_all('nptdms')
    datas += tmp_ret[0]
    binaries += tmp_ret[1]
    hiddenimports += tmp_ret[2]
except:
    pass

try:
    tmp_ret = collect_all('PIL')
    datas += tmp_ret[0]
    binaries += tmp_ret[1]
    hiddenimports += tmp_ret[2]
except:
    pass

excludes = [
    'torch',
    'tensorflow',
    'sklearn',
    'scipy',
    'matplotlib',
    'pandas',
    'jupyter',
    'IPython',
    'notebook',
    'PyQt5',
    'cv2',
    'skimage',
    'lightning',
    'wandb',
    'deeplay',
    'transformers',
    'langchain',
    'pydantic',
    'sqlalchemy',
    'boto3',
    'botocore',
    'sphinx',
    'docutils',
    'numba',
    'llvmlite',
    'pyarrow',
    'dask',
    'distributed',
    'xarray',
    'plotly',
    'kaleido',
    'statsmodels',
    'patsy',
    'pywt',
    'soundfile',
    'pygame',
    'OpenGL',
    'grpc',
    'zmq',
    'jupyterlab',
    'nbformat',
    'ipywidgets',
    'triton',
    'lxml',
    'ruamel',
    'argon2',
    'bcrypt',
    'nacl',
    'cryptography',
    'pycparser',
    'pygments',
    'wcwidth',
    'dateutil',
    'jinja2',
    'certifi',
    'urllib3',
    'cloudpickle',
    'anyio',
    'websockets',
    'pytz',
    'h5py',
    'charset_normalizer',
    'psutil',
    'setuptools',
    'pkg_resources',
    'importlib_metadata',
    'importlib_resources',
    'platformdirs',
    'tomli',
    'wheel',
    'jedi',
    'parso',
    'orjson',
    'narwhals',
    'regex',
    'sentry_sdk',
    'zoneinfo',
    'shelve',
    'sqlite3',
    'jaraco',
    'zipp',
    'more_itertools',
    'nvidia',
]

a = Analysis(
    ['$SCRIPT_DIR/tdms_to_png.py'],
    pathex=['$SCRIPT_DIR'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='tdms_to_png',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
EOF

pyinstaller --workpath "$SCRIPT_DIR/build" --distpath "$SCRIPT_DIR/dist" /tmp/tdms_to_png_build.spec

rm -f /tmp/tdms_to_png_build.spec

echo ""
echo "Build complete! Executable is located at:"
echo "  dist/tdms_to_png (Linux)"
echo "  dist/tdms_to_png.exe (Windows)"
echo ""
echo "You can copy it to your PATH or use it directly."
