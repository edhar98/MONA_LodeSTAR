#!/usr/bin/env python3
"""Interactive frame-0 crop, circle, and crescent-ratio measurement."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

import matplotlib


def _is_notebook() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return shell is not None and "IPKernelApp" in shell.config
    except Exception:
        return False


if not _is_notebook() and "MPLBACKEND" not in os.environ:
    for backend in ("Qt5Agg", "TkAgg"):
        try:
            matplotlib.use(backend, force=True)
            break
        except Exception:
            continue

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import Button, Slider
import numpy as np
import pandas as pd
from PIL import Image

from crescent_ratio import (
    CropRegion,
    ParticleDetection,
    load_frame0,
    measure_frame,
    normalize_uint8,
    save_overlay,
    select_analysis_crop,
    to_grayscale,
)


class CrescentRatioGUI:
    """Three-stage GUI for crop, manual circle, and segmentation review."""

    def __init__(
        self,
        input_path: Path,
        output_dir: Path,
        crop_size: int = 180,
        rim_exclusion_px: float = 5.0,
        threshold_percentile: float | None = None,
    ) -> None:
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.frame = load_frame0(self.input_path)
        self.gray = to_grayscale(self.frame)
        self.display = normalize_uint8(self.gray)
        self.height, self.width = self.gray.shape
        _, self.crop_region = select_analysis_crop(self.gray, crop_size=crop_size)
        self.rim_exclusion_px = float(rim_exclusion_px)
        self.threshold_percentile = threshold_percentile

        self.circle_x: float | None = None
        self.circle_y: float | None = None
        self.circle_radius: float | None = None
        self.measurement = None
        self.debug = None

        self.figure = plt.figure(figsize=(13, 9))
        self._event_ids: list[int] = []
        self._widgets: list[Button | Slider] = []
        self._crop_patch: Rectangle | None = None
        self._circle_patch: Circle | None = None
        self._drawing = False
        self._dragging = False
        self._slider_adjusting = False
        self._rendered_preview_settings: tuple[float, float] | None = None
        self._start_x: float | None = None
        self._start_y: float | None = None
        self._drag_offset = (0.0, 0.0)
        self.show_crop_stage()

    def _disconnect_events(self) -> None:
        for event_id in self._event_ids:
            self.figure.canvas.mpl_disconnect(event_id)
        self._event_ids.clear()

    def _connect(self, event_name: str, callback) -> None:
        self._event_ids.append(self.figure.canvas.mpl_connect(event_name, callback))

    def _register_widget(self, widget: Button | Slider) -> Button | Slider:
        self._widgets.append(widget)
        return widget

    def _reset_figure(self) -> None:
        self._disconnect_events()
        for widget in self._widgets:
            widget.disconnect_events()
        self._widgets.clear()

        mouse_grabber = self.figure.canvas.mouse_grabber
        if mouse_grabber is not None:
            self.figure.canvas.release_mouse(mouse_grabber)

        self.figure.clear()
        self._crop_patch = None
        self._circle_patch = None

    def show_crop_stage(self, _event=None) -> None:
        self._reset_figure()
        self.ax = self.figure.add_axes([0.05, 0.14, 0.90, 0.80])
        self.ax.imshow(self.display, cmap="gray", origin="upper")
        self.ax.set_title("1. Draw or move the square crop; scroll over it to resize")
        self.ax.set_axis_off()
        self._draw_crop_patch()

        button_ax = self.figure.add_axes([0.76, 0.035, 0.18, 0.055])
        self.crop_button = self._register_widget(Button(button_ax, "Use Crop"))
        self.crop_button.on_clicked(self.show_circle_stage)

        self._connect("button_press_event", self._crop_press)
        self._connect("button_release_event", self._crop_release)
        self._connect("motion_notify_event", self._crop_motion)
        self._connect("scroll_event", self._crop_scroll)
        self.figure.canvas.draw_idle()

    def _draw_crop_patch(self) -> None:
        if self._crop_patch is not None:
            self._crop_patch.remove()
        region = self.crop_region
        self._crop_patch = Rectangle(
            (region.x0, region.y0),
            region.width,
            region.height,
            fill=False,
            edgecolor="magenta",
            linewidth=2,
        )
        self.ax.add_patch(self._crop_patch)

    def _set_square_crop(self, center_x: float, center_y: float, size: float) -> None:
        size_i = max(20, min(int(round(size)), self.width, self.height))
        x0 = int(round(center_x - size_i / 2))
        y0 = int(round(center_y - size_i / 2))
        x0 = min(max(x0, 0), self.width - size_i)
        y0 = min(max(y0, 0), self.height - size_i)
        self.crop_region = CropRegion(x0, y0, x0 + size_i, y0 + size_i)
        self._draw_crop_patch()
        self.figure.canvas.draw_idle()

    def _inside_crop(self, x: float, y: float) -> bool:
        region = self.crop_region
        return region.x0 <= x <= region.x1 and region.y0 <= y <= region.y1

    def _crop_press(self, event) -> None:
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None or event.button != 1:
            return
        if self._inside_crop(event.xdata, event.ydata):
            self._dragging = True
            self._drag_offset = (event.xdata - self.crop_region.x0, event.ydata - self.crop_region.y0)
        else:
            self._drawing = True
            self._start_x, self._start_y = event.xdata, event.ydata

    def _crop_motion(self, event) -> None:
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return
        if self._drawing and self._start_x is not None and self._start_y is not None:
            size = max(abs(event.xdata - self._start_x), abs(event.ydata - self._start_y))
            center_x = (event.xdata + self._start_x) / 2
            center_y = (event.ydata + self._start_y) / 2
            self._set_square_crop(center_x, center_y, size)
        elif self._dragging:
            size = self.crop_region.width
            center_x = event.xdata - self._drag_offset[0] + size / 2
            center_y = event.ydata - self._drag_offset[1] + size / 2
            self._set_square_crop(center_x, center_y, size)

    def _crop_release(self, _event) -> None:
        self._drawing = False
        self._dragging = False

    def _crop_scroll(self, event) -> None:
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return
        if not self._inside_crop(event.xdata, event.ydata):
            return
        scale = 1.1 if event.button == "up" else 0.9
        region = self.crop_region
        self._set_square_crop(
            (region.x0 + region.x1) / 2,
            (region.y0 + region.y1) / 2,
            region.width * scale,
        )

    def show_circle_stage(self, _event=None) -> None:
        self._reset_figure()
        self.crop_gray = self.crop_region.extract(self.gray)
        self.crop_display = normalize_uint8(self.crop_gray)
        self.ax = self.figure.add_axes([0.05, 0.14, 0.90, 0.80])
        self.ax.imshow(self.crop_display, cmap="gray", origin="upper")
        self.ax.set_title("2. Left-drag from particle center to edge; right-click moves center; scroll changes radius")
        self.ax.set_axis_off()

        if self.circle_x is not None:
            self._draw_circle_patch()

        back_ax = self.figure.add_axes([0.05, 0.035, 0.16, 0.055])
        auto_ax = self.figure.add_axes([0.42, 0.035, 0.16, 0.055])
        preview_ax = self.figure.add_axes([0.76, 0.035, 0.18, 0.055])
        self.back_button = self._register_widget(Button(back_ax, "Back to Crop"))
        self.auto_button = self._register_widget(Button(auto_ax, "Auto Circle"))
        self.preview_button = self._register_widget(Button(preview_ax, "Preview Masks"))
        self.back_button.on_clicked(self.show_crop_stage)
        self.auto_button.on_clicked(self._auto_circle)
        self.preview_button.on_clicked(self.show_preview_stage)

        self._connect("button_press_event", self._circle_press)
        self._connect("button_release_event", self._circle_release)
        self._connect("motion_notify_event", self._circle_motion)
        self._connect("scroll_event", self._circle_scroll)
        self.figure.canvas.draw_idle()

    def _max_circle_radius(self, center_x: float, center_y: float) -> float:
        crop_h, crop_w = self.crop_gray.shape
        return max(1.0, min(center_x, center_y, crop_w - 1 - center_x, crop_h - 1 - center_y))

    def _set_circle(self, center_x: float, center_y: float, radius: float) -> None:
        crop_h, crop_w = self.crop_gray.shape
        center_x = float(np.clip(center_x, 0, crop_w - 1))
        center_y = float(np.clip(center_y, 0, crop_h - 1))
        max_radius = self._max_circle_radius(center_x, center_y)
        self.circle_x = center_x
        self.circle_y = center_y
        self.circle_radius = float(np.clip(radius, 2.0, max_radius))
        self._draw_circle_patch()
        self.figure.canvas.draw_idle()

    def _draw_circle_patch(self) -> None:
        if self._circle_patch is not None:
            self._circle_patch.remove()
        if self.circle_x is None or self.circle_y is None or self.circle_radius is None:
            return
        self._circle_patch = Circle(
            (self.circle_x, self.circle_y),
            self.circle_radius,
            fill=False,
            edgecolor="lime",
            linewidth=2,
        )
        self.ax.add_patch(self._circle_patch)

    def _circle_press(self, event) -> None:
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return
        if event.button == 3 and self.circle_radius is not None:
            self._set_circle(event.xdata, event.ydata, self.circle_radius)
            return
        if event.button == 1:
            self._drawing = True
            self._start_x, self._start_y = event.xdata, event.ydata
            self._set_circle(event.xdata, event.ydata, 2.0)

    def _circle_motion(self, event) -> None:
        if not self._drawing or event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return
        if self._start_x is None or self._start_y is None:
            return
        radius = float(np.hypot(event.xdata - self._start_x, event.ydata - self._start_y))
        self._set_circle(self._start_x, self._start_y, radius)

    def _circle_release(self, _event) -> None:
        self._drawing = False

    def _circle_scroll(self, event) -> None:
        if event.inaxes is not self.ax or self.circle_radius is None:
            return
        scale = 1.08 if event.button == "up" else 0.92
        self._set_circle(self.circle_x, self.circle_y, self.circle_radius * scale)

    def _auto_circle(self, _event=None) -> None:
        from crescent_ratio import detect_particle

        detection = detect_particle(
            self.crop_gray,
            center_window=max(self.crop_gray.shape),
            min_radius=18,
            max_radius=min(35, max(18, min(self.crop_gray.shape) // 3)),
        )
        self._set_circle(detection.center_x, detection.center_y, detection.radius_px)

    def _manual_detection(self) -> ParticleDetection:
        if self.circle_x is None or self.circle_y is None or self.circle_radius is None:
            raise ValueError("Draw the particle circle before previewing")
        return ParticleDetection(
            center_x=self.crop_region.x0 + self.circle_x,
            center_y=self.crop_region.y0 + self.circle_y,
            radius_px=self.circle_radius,
            method="gui_manual_circle",
            score=1.0,
        )

    def _measure(self, threshold_percentile: float | None) -> None:
        detection = self._manual_detection()
        self.measurement, self.debug = measure_frame(
            self.frame,
            self.input_path,
            polarity="bright",
            seed=detection,
            rim_exclusion_px=self.rim_exclusion_px,
            threshold_percentile=threshold_percentile,
            selected_crop=self.crop_region,
        )

    def _auto_threshold_percentile(self) -> float:
        self._measure(None)
        interior = self.debug["interior"]
        values = self.gray[interior] - self.measurement.background_value
        if values.size == 0:
            return 90.0
        percentile = 100.0 * float(np.mean(values <= self.measurement.threshold_value))
        return float(np.clip(percentile, 1.0, 99.0))

    def show_preview_stage(self, _event=None) -> None:
        try:
            initial_percentile = (
                self._auto_threshold_percentile()
                if self.threshold_percentile is None
                else float(self.threshold_percentile)
            )
        except ValueError as exc:
            print(exc)
            return

        self.threshold_percentile = initial_percentile
        self._reset_figure()
        self.preview_axes = [self.figure.add_subplot(1, 4, index + 1) for index in range(4)]
        self.figure.subplots_adjust(left=0.03, right=0.98, top=0.88, bottom=0.23, wspace=0.08)

        rim_max = max(1.0, min(15.0, self.circle_radius - 1.0))
        rim_ax = self.figure.add_axes([0.12, 0.11, 0.32, 0.035])
        threshold_ax = self.figure.add_axes([0.57, 0.11, 0.32, 0.035])
        self.rim_slider = self._register_widget(Slider(
            rim_ax,
            "Rim exclusion [px]",
            0.0,
            rim_max,
            valinit=min(self.rim_exclusion_px, rim_max),
            valstep=0.5,
        ))
        self.threshold_slider = self._register_widget(Slider(
            threshold_ax,
            "Bright threshold [%]",
            1.0,
            99.0,
            valinit=initial_percentile,
            valstep=0.5,
        ))
        self.rim_slider.on_changed(self._preview_changed)
        self.threshold_slider.on_changed(self._preview_changed)
        self._connect("button_press_event", self._preview_press)
        self._connect("button_release_event", self._preview_release)

        back_ax = self.figure.add_axes([0.05, 0.025, 0.18, 0.05])
        save_ax = self.figure.add_axes([0.77, 0.025, 0.18, 0.05])
        self.circle_back_button = self._register_widget(Button(back_ax, "Back to Circle"))
        self.save_button = self._register_widget(Button(save_ax, "Save Measurement"))
        self.circle_back_button.on_clicked(self.show_circle_stage)
        self.save_button.on_clicked(self.save_result)
        self._update_preview()

    def _preview_changed(self, _value) -> None:
        self.rim_exclusion_px = float(self.rim_slider.val)
        self.threshold_percentile = float(self.threshold_slider.val)

    def _preview_press(self, event) -> None:
        self._slider_adjusting = event.inaxes in {
            self.rim_slider.ax,
            self.threshold_slider.ax,
        }

    def _preview_release(self, _event) -> None:
        if not self._slider_adjusting:
            return
        self._slider_adjusting = False
        self._preview_changed(None)
        settings = (self.rim_exclusion_px, self.threshold_percentile)
        if settings != self._rendered_preview_settings:
            self._update_preview()

    def _update_preview(self) -> None:
        self._measure(self.threshold_percentile)
        self._rendered_preview_settings = (
            self.rim_exclusion_px,
            self.threshold_percentile,
        )
        crop_display = self.crop_region.extract(self.display)
        panels = [
            ("manual particle circle", None, None),
            ("measurement interior", self.debug["interior"], (0.2, 0.9, 0.4, 0.35)),
            ("excluded bright rim", self.debug["excluded_annulus"], (1.0, 0.35, 0.1, 0.55)),
            ("bright crescent", self.debug["crescent"], (1.0, 0.1, 0.1, 0.55)),
        ]
        for ax, (title, mask, color) in zip(self.preview_axes, panels):
            ax.clear()
            ax.imshow(crop_display, cmap="gray", origin="upper")
            if mask is not None:
                crop_mask = self.crop_region.extract(mask)
                rgba = np.zeros((*crop_mask.shape, 4), dtype=float)
                rgba[crop_mask] = color
                ax.imshow(rgba, origin="upper")
            ax.add_patch(Circle((self.circle_x, self.circle_y), self.circle_radius, fill=False, color="lime", lw=1.5))
            inner_radius = max(0.0, self.circle_radius - self.rim_exclusion_px)
            ax.add_patch(Circle((self.circle_x, self.circle_y), inner_radius, fill=False, color="yellow", lw=1.0, ls="--"))
            ax.set_title(title)
            ax.set_axis_off()
        self.figure.suptitle(
            f"3. Review masks | ratio = {self.measurement.crescent_area_ratio:.4f} | theta = {self.measurement.theta_deg:.1f} deg | out of plane = {self.measurement.out_of_plane_angle_deg:.1f} deg"
        )
        self.figure.canvas.draw_idle()

    def save_result(self, _event=None) -> None:
        self._update_preview()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = self.input_path.stem
        csv_path = self.output_dir / f"{stem}_frame0_crescent_measurement.csv"
        json_path = self.output_dir / f"{stem}_frame0_selection.json"
        crop_path = self.output_dir / f"{stem}_frame0_crop.png"
        overlay_path = self.output_dir / f"{stem}_frame0_overlay.png"

        pd.DataFrame([asdict(self.measurement)]).to_csv(csv_path, index=False)
        selection = {
            "input_path": str(self.input_path),
            "frame": 0,
            "crop": asdict(self.crop_region),
            "circle": asdict(self._manual_detection()),
            "polarity": "bright",
            "rim_exclusion_px": self.rim_exclusion_px,
            "threshold_percentile": self.threshold_percentile,
            "crescent_area_ratio": self.measurement.crescent_area_ratio,
            "theta_deg": self.measurement.theta_deg,
            "out_of_plane_angle_deg": self.measurement.out_of_plane_angle_deg,
        }
        json_path.write_text(json.dumps(selection, indent=2))
        Image.fromarray(normalize_uint8(self.crop_region.extract(self.gray))).save(crop_path)
        save_overlay(
            overlay_path,
            self.debug["gray"],
            self.debug["disk"],
            self.debug["interior"],
            self.debug["background"],
            self.debug["crescent"],
            self.debug["detection"],
            self.debug["crop_region"],
            title=f"{self.input_path.name} ratio={self.measurement.crescent_area_ratio:.4f}",
        )

        print(f"Saved ratio {self.measurement.crescent_area_ratio:.6f}")
        print(f"Measurement: {csv_path}")
        print(f"Selection:   {json_path}")
        print(f"Overlay:     {overlay_path}")
        plt.close(self.figure)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="TDMS or image path; frame 0 is used for TDMS files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tools/janus_crescent_ratio/outputs/gui"),
        help="Directory for measurement CSV, selection JSON, crop, and overlay.",
    )
    parser.add_argument("--crop-size", type=int, default=180, help="Initial square crop size in pixels.")
    parser.add_argument("--rim-exclusion-px", type=float, default=5.0, help="Initial excluded rim width.")
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=None,
        help="Initial bright threshold percentile; default derives it from automatic Otsu thresholding.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input.is_file():
        raise SystemExit(f"Input file does not exist: {args.input}")
    if args.crop_size < 20:
        raise SystemExit("--crop-size must be at least 20 pixels")
    if args.rim_exclusion_px < 0:
        raise SystemExit("--rim-exclusion-px must be non-negative")
    if args.threshold_percentile is not None and not 0 <= args.threshold_percentile <= 100:
        raise SystemExit("--threshold-percentile must be between 0 and 100")

    try:
        CrescentRatioGUI(
            args.input,
            args.output_dir,
            crop_size=args.crop_size,
            rim_exclusion_px=args.rim_exclusion_px,
            threshold_percentile=args.threshold_percentile,
        )
        plt.show()
    except Exception as exc:
        print(f"Failed to open crescent-ratio GUI: {exc}", file=sys.stderr)
        print("Check DISPLAY/X11 forwarding and the Qt or Tk Matplotlib backend.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
