# Janus Crescent Ratio

Standalone frame-0 analysis for fixed Janus-particle TDMS measurements.

The measured proxy is:

```text
crescent_area_ratio = segmented_crescent_area_px / rim_excluded_disk_area_px
```

For an ideal half-coated spherical particle, the projected-hemisphere mapping is:

```text
theta_deg = degrees(arccos(clip(2 * crescent_area_ratio - 1, -1, 1)))
out_of_plane_angle_deg = degrees(arcsin(clip(1 - 2 * crescent_area_ratio, 0, 1)))
```

Theta is the polar cosine angle: ratios 1, 0.5, and 0 correspond to 0, 90, and 180 degrees. The out-of-plane value uses the one-sided sine convention: a ratio of 0.5 means 0 degrees, a ratio of 0.25 means 30 degrees, and a ratio of 0 means 90 degrees. Ratios above 0.5 map to 0 out-of-plane degrees.

## Interactive GUI

Pass one TDMS or image path to the guided GUI:

```bash
/opt/mona_jupyterhub_env/bin/python tools/janus_crescent_ratio/src/crescent_ratio_gui.py \
  "/mnt/75/Data/Akshay/5CB paper/Measurements/Janus/30.07.26/Janus/1/P1_001_video.tdms" \
  --output-dir tools/janus_crescent_ratio/outputs/gui
```

The window proceeds through three stages:

1. Draw, move, or resize the frame-0 crop.
2. Draw the particle circle manually. An automatic inner-circle suggestion is also available.
3. Review the full disk, allowed interior, excluded bright rim, and bright crescent. Adjust the rim width and bright-threshold percentile before saving.

Each saved selection produces:

- `<name>_frame0_crescent_measurement.csv`: the ratio and all mask areas.
- `<name>_frame0_selection.json`: reusable full-frame crop, circle, and segmentation settings.
- `<name>_frame0_crop.png`: the selected frame-0 crop.
- `<name>_frame0_overlay.png`: the complete QC visualization.

The GUI requires a desktop display. When connecting over SSH, enable X11 forwarding.

## Run

From the repository root:

```bash
/opt/mona_jupyterhub_env/bin/python tools/janus_crescent_ratio/src/crescent_ratio.py \
  --input-root "/mnt/75/Data/Akshay/5CB paper/Measurements/Janus/30.07.26/Janus" \
  --output-dir tools/janus_crescent_ratio/outputs \
  --crop-size 180 --min-radius 18 --max-radius 35 \
  --rim-exclusion-px 5 --polarity bright
```

The pipeline first extracts a 180 x 180 px crop around the frame center, then
runs circle detection only inside that crop. The 18–35 px radius constraint is
for the inner Janus particle and deliberately excludes the much larger circular
ring surrounding it. Detected centers are converted back to full-frame pixel
coordinates before they are written to CSV.

The detected radius defines the particle disk. Crescent thresholding and area
normalization are performed within an inner disk whose radius is 5 px smaller.
This removes the orientation-independent bright annulus at the particle
boundary from both the numerator and denominator.

If the relevant region is not exactly at the frame center, set it explicitly:

```bash
--crop-center-x 480 --crop-center-y 500 --crop-size 180
```

For a fast smoke test:

```bash
/opt/mona_jupyterhub_env/bin/python tools/janus_crescent_ratio/src/crescent_ratio.py \
  --input-root "/mnt/75/Data/Akshay/5CB paper/Measurements/Janus/30.07.26/Janus" \
  --output-dir /tmp/janus_crescent_ratio_smoke \
  --limit 3 --overlay-limit 3 --rim-exclusion-px 5 --polarity bright
```

The crescent is measured as a bright feature by default. `--polarity dark`
remains available for datasets with an inverted intensity convention.

```bash
--polarity dark
```

## Outputs

The default output directory is `tools/janus_crescent_ratio/outputs/`.

- `janus_crescent_ratio_frame0_measurements.csv`: one row per TDMS file.
- `janus_crescent_ratio_file_summary.csv`: per-file ratio summary.
- `janus_crescent_ratio_folder_summary.csv`: per-folder ratio summary.
- `janus_crescent_ratio_histogram.png`: distribution of ratios.
- `overlays/*_overlay.png`: crop, full disk, allowed interior, excluded bright rim, background, and crescent masks.
- `run_config.json`: run configuration.

## Manual Seeds

For difficult files, pass `--seed-csv path/to/seeds.csv`. The CSV can key rows by full `path` or by `file` name:

```csv
file,center_x,center_y,radius_px
P1_001_video.tdms,475,505,30
```

Manual seed centers use full-frame coordinates. For automatic detection, keep
`--max-radius` below the enclosing-ring radius; increase `--crop-size` if the
particle can be farther from the frame center.
