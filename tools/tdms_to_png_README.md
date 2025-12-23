# TDMS to PNG/MP4 Converter

Converts TDMS (Technical Data Management Streaming) files to PNG images with optional MP4 video output. Supports parallel processing, automatic dtype conversion, and flexible file pattern matching.

## Features

- PNG image conversion from TDMS files
- Optional MP4 video output (`--to-mp4`)
- Multi-core parallel processing
- Pattern matching for batch processing
- Automatic dtype normalization (uint8, uint16, float32)
- Skip existing files (resume capability)
- TDMS structure inspection

## Requirements

```bash
pip install nptdms pillow numpy
pip install imageio imageio-ffmpeg  # for MP4 output
```

- Python 3.8+
- FFmpeg (bundled with imageio-ffmpeg)

## Usage

```bash
python tools/tdms_to_png.py input.tdms -o output_dir
python tools/tdms_to_png.py "file_{:03d}.tdms" -o output --start-index 1 --num-files 10
python tools/tdms_to_png.py input.tdms --list-structure
python tools/tdms_to_png.py input.tdms -o output --to-mp4 --fps 30
```

## Command-Line Options

### Required
- `input`: Input TDMS file path or pattern (`{:03d}` for numbered files)

### Output
- `-o, --output DIR`: Output directory (required unless `--list-structure`)
- `-f, --force`: Overwrite existing files

### File Selection
- `--start-index N`: Starting file index (default: 1)
- `--num-files N`: Max files to process

### Image Parameters
- `--width N`: Image width (default: 1024)
- `--height N`: Image height (default: 1024)
- `--channel N`: Channel index (default: 0)
- `--group NAME`: TDMS group name

### Video Parameters
- `--to-mp4`: Create MP4 videos
- `--fps FLOAT`: Frames per second (default: 30.0)

### Data Type
- `--dtype {uint8,uint16,float32}`: Output dtype (default: preserve input)

### Processing
- `--workers N`: Parallel workers (default: CPU count)
- `--base-name NAME`: Output file base name

### Utilities
- `--list-structure`: Show TDMS structure and exit

## Examples

### Convert Single File
```bash
python tools/tdms_to_png.py data/experiment.tdms -o output/images
```

### Pattern Matching
```bash
python tools/tdms_to_png.py "data/experiment_{:03d}_video.tdms" \
    -o output/images --start-index 1 --num-files 5
```

### Convert to PNG and MP4
```bash
python tools/tdms_to_png.py input.tdms -o output --to-mp4 --fps 30
```

### Custom Dimensions
```bash
python tools/tdms_to_png.py input.tdms -o output --width 2048 --height 2048 --dtype uint8
```

### Inspect Structure
```bash
python tools/tdms_to_png.py data/experiment.tdms --list-structure
```

Output:
```
TDMS file structure for data/experiment.tdms:
  Group: '/'Image'' (1 channels)
    Channel 0: Image, shape=(104857600,), dtype=uint16
```

### Resume Conversion
```bash
python tools/tdms_to_png.py "data/*.tdms" -o output      # skips existing
python tools/tdms_to_png.py "data/*.tdms" -o output -f   # force overwrite
```

## Output Naming

- PNG: `{base_name}_{index:03d}.png`
- MP4: `{base_name}.mp4`

## Dtype Conversion

Automatic normalization when converting:
- uint16 → uint8: 0-65535 to 0-255
- uint8 → uint16: 0-255 to 0-65535
- float → uint8/uint16: normalized to full range

## Troubleshooting

### "Group 'Image' not found"
```bash
python tools/tdms_to_png.py input.tdms --list-structure
python tools/tdms_to_png.py input.tdms -o output --group "/'Image'"
```

### "imageio is required for video conversion"
```bash
pip install imageio imageio-ffmpeg
```

### "Data size is not divisible by image size"
Check dimensions match: `total_pixels = num_images * width * height`

## Jupyter Notebook Usage

```python
!python tools/tdms_to_png.py data/experiment.tdms --list-structure
```

```python
!python tools/tdms_to_png.py data/experiment.tdms -o output --to-mp4 --fps 30
```

```python
!python tools/tdms_to_png.py "data/experiment_{:03d}.tdms" -o output --start-index 1 --num-files 5 --workers 4
```

## Standalone Executable

Build with PyInstaller:
```bash
cd tools
./build_tdms_to_png.sh
```

Output: `dist/tdms_to_png` (Linux) or `dist/tdms_to_png.exe` (Windows)

Usage is identical to Python script:
```bash
./tdms_to_png input.tdms -o output --to-mp4 --fps 30
```
