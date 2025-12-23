# Tools

Utilities for MONA LodeSTAR data processing and image manipulation.

## `tdms_to_png.py`

Converts TDMS files to PNG images with optional MP4 video output.

- Parallel processing
- Pattern matching for batch processing
- Automatic dtype conversion (uint8, uint16, float32)
- Skip existing files (resume capability)
- TDMS structure inspection

```bash
pip install nptdms pillow numpy
pip install imageio imageio-ffmpeg  # for MP4
```

```bash
python tools/tdms_to_png.py input.tdms -o output_dir
python tools/tdms_to_png.py "file_{:03d}.tdms" -o output --start-index 1 --num-files 10
python tools/tdms_to_png.py input.tdms --list-structure
python tools/tdms_to_png.py input.tdms -o output --to-mp4 --fps 30
```

Jupyter:
```python
!python tools/tdms_to_png.py data/experiment.tdms -o output --to-mp4 --fps 30
```

Standalone executable (build with `./build_tdms_to_png.sh`):
```bash
./tdms_to_png input.tdms -o output --to-mp4 --fps 30
```

See [tdms_to_png_README.md](tdms_to_png_README.md) for full documentation.

---

## `crop.py`

Interactive GUI for cropping images with square selection.

- Click-drag to draw square box
- Drag inside box to move
- Scroll wheel to resize
- Automatic backend detection (Qt5/Qt4/Tk)

```bash
pip install matplotlib pillow numpy PyQt5
```

```bash
python tools/crop.py input.png output_cropped.png
```
