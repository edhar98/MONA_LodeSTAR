from io import BytesIO
from pathlib import Path
import sys
import unittest

import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "janus_crescent_ratio" / "src"))

from crescent_ratio import (
    CropRegion,
    ParticleDetection,
    circular_mask,
    crescent_ratio_to_out_of_plane_angle_deg,
    crescent_ratio_to_theta_deg,
    detect_particle,
    display_uint8,
    measure_frame,
    normalize_frame_uint8,
    save_overlay,
    segment_crescent,
    select_analysis_crop,
)


class CrescentRatioTests(unittest.TestCase):
    def test_crescent_ratio_to_out_of_plane_angle_mapping(self):
        self.assertEqual(crescent_ratio_to_out_of_plane_angle_deg(0.5), 0.0)
        self.assertEqual(crescent_ratio_to_out_of_plane_angle_deg(0.0), 90.0)
        self.assertAlmostEqual(crescent_ratio_to_out_of_plane_angle_deg(0.25), 30.0)
        self.assertEqual(crescent_ratio_to_out_of_plane_angle_deg(0.75), 0.0)

    def test_crescent_ratio_to_theta_mapping(self):
        self.assertEqual(crescent_ratio_to_theta_deg(1.0), 0.0)
        self.assertEqual(crescent_ratio_to_theta_deg(0.5), 90.0)
        self.assertAlmostEqual(crescent_ratio_to_theta_deg(0.25), 120.0)
        self.assertEqual(crescent_ratio_to_theta_deg(0.0), 180.0)

    def test_circular_mask_area_is_close_to_pi_r_squared(self):
        radius = 20
        mask = circular_mask((100, 100), 50, 50, radius)
        expected = np.pi * radius**2
        self.assertLess(abs(mask.sum() - expected) / expected, 0.05)

    def test_bright_crescent_ratio_on_synthetic_disk(self):
        image = np.full((100, 100), 10.0)
        detection = ParticleDetection(50, 50, 25, "test", 1.0)
        disk = circular_mask(image.shape, detection.center_x, detection.center_y, detection.radius_px)
        yy, xx = np.mgrid[:100, :100]
        crescent_region = disk & (xx > 55) & (yy < 60)
        image[disk] = 20.0
        image[crescent_region] = 120.0

        disk_mask, interior, _, crescent, _, _, qc_status, _ = segment_crescent(
            image,
            detection,
            polarity="bright",
            rim_exclusion_px=5,
        )

        self.assertEqual(qc_status, "ok")
        self.assertEqual(int(disk_mask.sum()), int(disk.sum()))
        self.assertLess(interior.sum(), disk_mask.sum())
        self.assertGreater(crescent.sum(), 0)
        self.assertLess(crescent.sum() / disk_mask.sum(), 0.5)

    def test_bright_particle_annulus_is_excluded_from_crescent(self):
        image = np.full((120, 120), 10.0)
        detection = ParticleDetection(60, 60, 30, "test", 1.0)
        yy, xx = np.mgrid[:120, :120]
        distance = np.hypot(xx - detection.center_x, yy - detection.center_y)
        disk = distance <= detection.radius_px
        bright_annulus = disk & (distance > detection.radius_px - 5)
        bright_crescent = (distance < 20) & (xx > 64)
        image[disk] = 20.0
        image[bright_annulus] = 140.0
        image[bright_crescent] = 120.0

        _, interior, _, crescent, _, _, _, _ = segment_crescent(
            image,
            detection,
            polarity="bright",
            rim_exclusion_px=5,
        )

        self.assertFalse(np.any(crescent & bright_annulus))
        self.assertTrue(np.all(crescent <= interior))
        self.assertGreater(crescent.sum(), 0)

    def test_measure_frame_accepts_manual_seed(self):
        image = np.full((80, 80), 5.0)
        seed = ParticleDetection(40, 40, 15, "manual_seed", 1.0)
        disk = circular_mask(image.shape, seed.center_x, seed.center_y, seed.radius_px)
        image[disk] = 15.0
        image[disk & (np.indices(image.shape)[1] > 42)] = 90.0

        measurement, _ = measure_frame(image, Path("sample.tdms"), seed=seed)

        self.assertEqual(measurement.detection_method, "manual_seed")
        self.assertGreater(measurement.crescent_area_ratio, 0)
        self.assertEqual(
            measurement.crescent_area_ratio,
            measurement.crescent_area_px / measurement.interior_area_px,
        )

    def test_measure_frame_uses_gui_crop_and_threshold_percentile(self):
        image = np.full((120, 140), 10.0)
        seed = ParticleDetection(70, 55, 20, "gui_manual_circle", 1.0)
        disk = circular_mask(image.shape, seed.center_x, seed.center_y, seed.radius_px)
        image[disk] = 20.0
        image[disk & (np.indices(image.shape)[1] > 74)] = 100.0
        selected_crop = CropRegion(30, 20, 110, 100)

        measurement, debug = measure_frame(
            image,
            Path("sample.tdms"),
            seed=seed,
            selected_crop=selected_crop,
            threshold_percentile=75,
        )

        self.assertEqual(debug["crop_region"], selected_crop)
        self.assertEqual(measurement.threshold_percentile, 75)
        self.assertEqual(measurement.detection_method, "gui_manual_circle")

    def test_measure_frame_normalizes_intensity_when_requested(self):
        image = np.linspace(1000, 5000, 6400, dtype=np.float64).reshape(80, 80)
        seed = ParticleDetection(40, 40, 15, "manual_seed", 1.0)

        measurement, debug = measure_frame(image, Path("sample.tdms"), seed=seed, normalize=True)

        self.assertTrue(measurement.normalized)
        self.assertGreaterEqual(float(np.min(debug["gray"])), 0.0)
        self.assertLessEqual(float(np.max(debug["gray"])), 255.0)

    def test_display_normalization_toggle_changes_high_bit_depth_contrast(self):
        image = np.linspace(1000, 5000, 6400, dtype=np.uint16).reshape(80, 80)

        raw_display = display_uint8(image, normalize=False)
        normalized_display = display_uint8(image, normalize=True)

        self.assertLess(int(np.ptp(raw_display)), int(np.ptp(normalized_display)))
        self.assertGreaterEqual(int(normalized_display.max()), 250)

    def test_frame_normalization_uses_min_max_scaling(self):
        image = np.array([[1000, 2000], [3000, 5000]], dtype=np.int32)
        expected = ((image.astype(np.float32) - 1000) / (5000 - 1000 + 1e-8) * 255).astype(np.uint8)

        normalized = normalize_frame_uint8(image)

        np.testing.assert_array_equal(normalized, expected)

    def test_raw_display_handles_16_bit_camera_values_in_int32_container(self):
        image = np.linspace(2000, 27000, 6400, dtype=np.int32).reshape(80, 80)

        raw_display = display_uint8(image, normalize=False)

        self.assertGreater(int(np.ptp(raw_display)), 90)
        self.assertLessEqual(int(raw_display.max()), 106)

    def test_overlay_uses_requested_normalization_mode(self):
        image = np.linspace(1000, 5000, 6400, dtype=np.uint16).reshape(80, 80)
        detection = ParticleDetection(40, 40, 15, "manual_seed", 1.0)
        disk = circular_mask(image.shape, 40, 40, 15)
        interior = circular_mask(image.shape, 40, 40, 10)
        crescent = interior & (np.indices(image.shape)[1] > 40)
        crop = CropRegion(20, 20, 60, 60)
        raw_buffer = BytesIO()
        normalized_buffer = BytesIO()

        save_overlay(raw_buffer, image, disk, interior, crescent, detection, crop, normalize=False)
        save_overlay(normalized_buffer, image, disk, interior, crescent, detection, crop, normalize=True)

        self.assertNotEqual(raw_buffer.getvalue(), normalized_buffer.getvalue())

    def test_crop_rejects_large_enclosing_ring(self):
        size = 256
        yy, xx = np.mgrid[:size, :size]
        image = np.full((size, size), 100.0, dtype=np.float32)

        enclosing_distance = np.hypot(xx - 128, yy - 128)
        image[(enclosing_distance > 67) & (enclosing_distance < 73)] = 25.0

        particle_distance = np.hypot(xx - 116, yy - 125)
        image[particle_distance < 27] = 160.0
        crescent = (particle_distance < 25) & (yy < 125) & (xx > 105)
        image[crescent] = 15.0
        image = cv2.GaussianBlur(image, (5, 5), 1)

        crop, region = select_analysis_crop(image, crop_size=160)
        detection = detect_particle(crop, center_window=160, min_radius=18, max_radius=35, hough_param2=15)
        center_x = detection.center_x + region.x0
        center_y = detection.center_y + region.y0

        self.assertLess(np.hypot(center_x - 116, center_y - 125), 5)
        self.assertLessEqual(detection.radius_px, 35)
        self.assertGreaterEqual(detection.radius_px, 18)


if __name__ == "__main__":
    unittest.main()
