#!/usr/bin/env python3
"""
LodeSTAR position detection + template-matching orientation.
Pipeline: 2ch LodeSTAR (self-supervised) -> HoughCircles refinement -> NCC template matching.
Run from repo root.
"""
from __future__ import annotations

import os
import re
import sys
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import deeptrack as dt

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SRC_DIR))
sys.path.insert(0, os.path.dirname(_SRC_DIR))  # src/ for utils
sys.path.insert(0, _REPO_ROOT)  # repo root for lodestar_link

import deeptrack.deeplay as dl
from lodestar_link.lodestar import LodeSTAR as BaseLodeSTAR
from lodestar_link.transforms import RandomTranslation2d, Transform, Transforms
import kornia
import cv2
from scipy import ndimage
from skimage import morphology


def normalize_cos_sin(cos: float, sin: float) -> tuple[float, float]:
    mag = np.sqrt(cos * cos + sin * sin) + 1e-9
    return cos / mag, sin / mag


def _suppress_nearby(
    positions: np.ndarray,
    scores: np.ndarray,
    min_distance: float,
) -> np.ndarray:
    if len(positions) <= 1 or min_distance <= 0:
        return np.arange(len(positions))
    order = np.argsort(-scores)
    keep: list[int] = []
    for idx in order:
        xy = positions[idx]
        too_close = False
        for k in keep:
            d2 = (positions[k, 0] - xy[0]) ** 2 + (positions[k, 1] - xy[1]) ** 2
            if d2 < min_distance ** 2:
                too_close = True
                break
        if not too_close:
            keep.append(idx)
    return np.array(keep, dtype=int)


def detect_single_with_smooth_orientation(
    model: BaseLodeSTAR,
    y_pred_one: torch.Tensor,
    weights_one: torch.Tensor,
    alpha: float,
    beta: float,
    cutoff: float,
    mode: str,
    window: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    score = model.get_detection_score(
        y_pred_one, weights_one.unsqueeze(0), alpha, beta
    )
    score_trimmed = score[3:-3, 3:-3]
    if score_trimmed.size == 0:
        dets = model.detect_single(y_pred_one, weights_one, alpha, beta, cutoff, mode)
        if len(dets) == 0 or dets.shape[1] < 4:
            empty = np.empty((0, 4), dtype=np.float64) if len(dets) == 0 else dets[:, :4].astype(np.float64)
            return empty, np.zeros(len(empty), dtype=np.float64)
        out = []
        for i in range(len(dets)):
            cos_n, sin_n = normalize_cos_sin(float(dets[i, 2]), float(dets[i, 3]))
            out.append([float(dets[i, 0]), float(dets[i, 1]), cos_n, sin_n])
        arr = np.array(out, dtype=np.float64)
        return arr, np.ones(len(arr), dtype=np.float64)
    th = cutoff
    if mode == "quantile":
        th = np.quantile(score_trimmed, cutoff)
    elif mode == "ratio":
        th = np.max(score_trimmed.flatten()) * cutoff
    hmax = morphology.h_maxima(np.squeeze(score_trimmed), th) == 1
    hmax = np.pad(hmax, ((3, 3), (3, 3)))
    rows, cols = np.where(hmax)
    pred = y_pred_one.detach().cpu()
    n_ch, H, W = pred.shape
    dets_list: list[list[float]] = []
    det_scores: list[float] = []
    for r, c in zip(rows, cols):
        x = pred[0, r, c].item()
        y = pred[1, r, c].item()
        if n_ch >= 4:
            r0, r1 = max(0, r - window), min(H, r + window + 1)
            c0, c1 = max(0, c - window), min(W, c + window + 1)
            cos_avg = pred[2, r0:r1, c0:c1].mean().item()
            sin_avg = pred[3, r0:r1, c0:c1].mean().item()
            cos_n, sin_n = normalize_cos_sin(cos_avg, sin_avg)
            dets_list.append([x, y, cos_n, sin_n])
        else:
            dets_list.append([x, y])
        det_scores.append(float(score[r, c]))
    if not dets_list:
        n_cols = 4 if n_ch >= 4 else 2
        return np.empty((0, n_cols)), np.zeros(0, dtype=np.float64)
    return np.array(dets_list, dtype=np.float64), np.array(det_scores, dtype=np.float64)


def refine_position_to_center(
    image: np.ndarray,
    x: float,
    y: float,
    search_radius: int = 20,
    min_radius: int = 3,
    max_radius: int | None = None,
) -> tuple[float, float]:
    img = image if image.ndim == 2 else np.dot(image[..., :3], [0.299, 0.587, 0.114])
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape[:2]
    x_int, y_int = int(round(x)), int(round(y))
    r = max(search_radius, (max_radius or search_radius) + 2)
    x0 = max(0, x_int - r)
    x1 = min(w, x_int + r + 1)
    y0 = max(0, y_int - r)
    y1 = min(h, y_int + r + 1)
    patch = img[y0:y1, x0:x1]
    if patch.size == 0:
        return x, y
    ph, pw = patch.shape[:2]
    pmin, pmax = float(patch.min()), float(patch.max())
    if pmax <= pmin:
        return x, y
    cx_in = x_int - x0
    cy_in = y_int - y0
    patch_uint8 = np.clip((patch - pmin) / (pmax - pmin) * 255, 0, 255).astype(np.uint8)
    max_r = max_radius if max_radius is not None else min(ph, pw) // 2 - 1
    max_r = max(min_radius + 1, max_r)
    circles = cv2.HoughCircles(
        patch_uint8,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=1,
        param1=50,
        param2=18,
        minRadius=min_radius,
        maxRadius=max_r,
    )
    if circles is not None and circles.size > 0:
        circles = np.squeeze(circles, axis=0)
        if circles.ndim == 1:
            circles = circles[np.newaxis, :]
        best, best_d2 = None, float("inf")
        for row in circles:
            cx_p, cy_p = float(row[0]), float(row[1])
            d2 = (cx_p - cx_in) ** 2 + (cy_p - cy_in) ** 2
            if d2 < best_d2:
                best_d2, best = d2, (cx_p, cy_p)
        if best is not None:
            return float(x0 + best[0]), float(y0 + best[1])
    r_disk = min(search_radius, min(ph, pw) // 2)
    yy, xx = np.mgrid[0:ph, 0:pw]
    dist = np.sqrt((xx - cx_in) ** 2 + (yy - cy_in) ** 2)
    mask = dist <= r_disk
    if np.any(mask):
        intensity = np.maximum(patch - patch[mask].min(), 0.0)
        intensity = np.where(mask, intensity, 0.0)
        total = intensity.sum() + 1e-9
        com_x = np.sum(xx * intensity) / total
        com_y = np.sum(yy * intensity) / total
        return float(x0 + com_x), float(y0 + com_y)
    return x, y


def compute_orientation_from_gradient(image: np.ndarray, x: float, y: float, radius: int = 15, refine: bool = False) -> tuple[float, float]:
    if refine:
        x, y = refine_position_to_center(image, x, y, search_radius=radius)
    h, w = image.shape[:2]
    x_int, y_int = int(round(x)), int(round(y))
    x0 = max(0, x_int - radius)
    x1 = min(w, x_int + radius + 1)
    y0 = max(0, y_int - radius)
    y1 = min(h, y_int + radius + 1)
    patch = image[y0:y1, x0:x1].astype(np.float64)
    if patch.size == 0:
        return 1.0, 0.0
    ph, pw = patch.shape[:2]
    cy, cx = (y_int - y0), (x_int - x0)
    yy, xx = np.mgrid[0:ph, 0:pw]
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    mask = dist <= radius
    if not np.any(mask):
        return 1.0, 0.0
    intensity = patch[mask]
    dx = (xx - cx)[mask]
    dy = (yy - cy)[mask]
    intensity_centered = intensity - intensity.mean()
    total_weight = np.sum(np.abs(intensity_centered)) + 1e-9
    vec_x = np.sum(dx * intensity_centered) / total_weight
    vec_y = np.sum(dy * intensity_centered) / total_weight
    mag = np.sqrt(vec_x**2 + vec_y**2)
    if mag < 1e-6:
        return 1.0, 0.0
    return vec_x / mag, vec_y / mag


class LodeSTAR(BaseLodeSTAR):
    def forward(self, x):
        _, _, Hx, Wx = x.shape
        y = self.model(x)
        _, _, Hy, Wy = y.shape
        x_range = torch.arange(Hy, device=x.device) * Hx / Hy
        y_range = torch.arange(Wy, device=x.device) * Wx / Wy
        if self.training:
            x_range = x_range - Hx / 2 + 0.5
            y_range = y_range - Wx / 2 + 0.5
        Y, X = torch.meshgrid(y_range, x_range, indexing="xy")
        delta_x = y[:, 0]
        delta_y = y[:, 1]
        weights = y[:, -1].sigmoid()
        X = X + delta_x
        Y = Y + delta_y
        return torch.cat([X[:, None], Y[:, None], y[:, 2:-1], weights[:, None]], dim=1)

    def compute_loss(self, y_hat, inverse_fn):
        loss_dict = super().compute_loss(y_hat, inverse_fn)
        if self.num_outputs >= 4:
            cos_sin = y_hat[:, 2:4]
            mag = (cos_sin[:, 0:1]**2 + cos_sin[:, 1:2]**2).sqrt()
            mag_loss = ((mag - 1.0)**2).mean()
            loss_dict["magnitude_regularization"] = mag_loss * 10.0
        return loss_dict


def _rotation_4ch_inverse(x: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    sh = x.shape
    if sh[1] < 4:
        return x
    x_flat = x.view(sh[0], sh[1], -1)
    a = -angle.to(x.device).view(-1, 1, 1)
    c, s = torch.cos(a), torch.sin(a)
    out = x_flat.clone()
    vi_01 = x_flat[:, 0:1, :]
    vj_01 = x_flat[:, 1:2, :]
    out[:, 0:1, :] = c * vi_01 - s * vj_01
    out[:, 1:2, :] = s * vi_01 + c * vj_01
    vi_23 = x_flat[:, 2:3, :]
    vj_23 = x_flat[:, 3:4, :]
    out[:, 2:3, :] = c * vi_23 - s * vj_23
    out[:, 3:4, :] = s * vi_23 + c * vj_23
    return out.view(sh)


class Rotation4Ch(Transform):
    def __init__(self, angle=lambda: np.random.uniform(-np.pi, np.pi)):
        super().__init__(self._forward, self._backward, angle=angle)

    @staticmethod
    def _forward(x: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        angle = angle.type_as(x).to(x.device)
        return kornia.geometry.transform.rotate(
            x, angle * 180 / np.pi, align_corners=True, padding_mode="reflection"
        )

    @staticmethod
    def _backward(x: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        return _rotation_4ch_inverse(x, angle)


def make_orientation_transforms() -> Transforms:
    return Transforms([
        RandomTranslation2d(),
        Rotation4Ch(),
    ])


def load_sample_as_numpy(path: str) -> np.ndarray:
    training_image = np.array(dt.LoadImage(path).resolve()).astype(np.float32)
    if len(training_image.shape) == 3 and training_image.shape[-1] == 3:
        training_image = np.dot(training_image[..., :3], [0.299, 0.587, 0.114])
    if len(training_image.shape) == 2:
        training_image = training_image[..., np.newaxis]
    return training_image


def create_training_pipeline(
    sample_path: str,
    scale_min: float = 1.0,
    scale_max: float = 1.0,
    rotation_range: tuple[float, float] = (0.0, 0.0),
    translation_range: tuple[float, float] = (0.0, 0.0),
    mul_min: float = 0.1,
    mul_max: float = 0.9,
    add_min: float = -0.1,
    add_max: float = 0.1,
):
    training_image = load_sample_as_numpy(sample_path)
    return (
        dt.Value(training_image)
        >> dt.Affine(
            scale=lambda: np.random.uniform(scale_min, scale_max),
            rotate=lambda: 2 * np.pi * np.random.uniform(rotation_range[0], rotation_range[1]),
            translate=lambda: np.random.uniform(translation_range[0], translation_range[1], 2),
            mode="constant",
        )
        >> dt.Multiply(lambda: np.random.uniform(mul_min, mul_max))
        >> dt.Add(lambda: np.random.uniform(add_min, add_max))
        >> dt.MoveAxis(-1, 0)
        >> dt.pytorch.ToTensor(dtype=torch.float32)
    )


def load_csv_gt(csv_path: str) -> dict[int, dict]:
    df = pd.read_csv(csv_path, index_col=0)
    cols = list(df.columns)
    frame_col = "frame" if "frame" in cols else cols[0]
    x_col = "x" if "x" in cols else "x"
    y_col = "y" if "y" in cols else "y"
    phi_col = "phi" if "phi" in cols else "phi"
    df[phi_col] = pd.to_numeric(df[phi_col], errors="coerce")
    sorted_frames = sorted(df[frame_col].unique())
    out: dict[int, dict] = {}
    for image_idx, frame_val in enumerate(sorted_frames):
        subset = df[df[frame_col] == frame_val].copy()
        subset = subset.dropna(subset=[phi_col])
        if subset.empty:
            out[image_idx] = {"positions": np.empty((0, 2)), "phi": np.array([]), "frame": int(frame_val)}
            continue
        out[image_idx] = {
            "positions": subset[[x_col, y_col]].values.astype(np.float64),
            "phi": subset[phi_col].values.astype(np.float64),
            "frame": int(frame_val),
        }
    return out


def _parse_phi_from_filename(path: str) -> float:
    m = re.search(r'phi(\d+\.?\d*)', os.path.basename(path))
    if m is None:
        raise ValueError(f"No phi in filename: {path}")
    return float(m.group(1))


def _load_training_samples(
    sample_path: str,
    max_samples: Optional[int] = None,
) -> list[tuple[torch.Tensor, float]]:
    samples: list[tuple[torch.Tensor, float]] = []
    seen_paths: set[str] = set()

    def _append_sample(fp: str) -> None:
        if fp in seen_paths:
            return
        pd_deg = _parse_phi_from_filename(fp)
        r = load_sample_as_numpy(fp)
        st = torch.from_numpy(r).permute(2, 0, 1).unsqueeze(0).float()
        samples.append((st, np.radians(pd_deg)))
        seen_paths.add(fp)

    if max_samples == 1:
        _append_sample(sample_path)
        return samples

    sub_dir = os.path.join(os.path.dirname(sample_path), "Samples")
    if os.path.isdir(sub_dir):
        for f in sorted(os.listdir(sub_dir)):
            if max_samples is not None and len(samples) >= max_samples:
                break
            if not f.endswith(".png") or "phi" not in f:
                continue
            fp = os.path.join(sub_dir, f)
            try:
                _append_sample(fp)
            except ValueError:
                continue

    if not samples:
        _append_sample(sample_path)
        return samples

    if max_samples is not None and len(samples) < max_samples:
        try:
            if sample_path not in seen_paths:
                _append_sample(sample_path)
        except ValueError:
            pass

    return samples


def _pad_sample(sample: torch.Tensor) -> torch.Tensor:
    _, _, H, W = sample.shape
    pad_size = max(H, W) * 3
    if pad_size % 2:
        pad_size += 1
    pad_h = (pad_size - H) // 2
    pad_w = (pad_size - W) // 2
    mean_val = sample.mean().item()
    padded = torch.full((1, 1, pad_size, pad_size), mean_val, dtype=sample.dtype)
    padded[0, 0, pad_h:pad_h + H, pad_w:pad_w + W] = sample[0, 0]
    return padded


def _make_augmented_batch(
    samples: list[tuple[torch.Tensor, float]],
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    padded_cache = [_pad_sample(s) for s, _ in samples]
    val_ranges = [float(s.max() - s.min()) + 1e-6 for s, _ in samples]
    angles = np.random.uniform(-np.pi, np.pi, batch_size)
    ref_phis = np.zeros(batch_size)
    imgs = []
    for i in range(batch_size):
        idx = np.random.randint(len(samples))
        ref_phis[i] = samples[idx][1]
        t = padded_cache[idx].clone()
        t = kornia.geometry.transform.rotate(
            t, torch.tensor([float(np.degrees(angles[i]))]),
            align_corners=True, padding_mode="reflection",
        )
        mul = np.random.uniform(0.1, 0.9)
        add = np.random.uniform(-0.1, 0.1)
        t = t * mul + add
        noise_std = np.random.uniform(0.0, 0.05) * val_ranges[idx]
        t = t + torch.randn_like(t) * noise_std
        imgs.append(t)
    return torch.cat(imgs, dim=0).to(device), angles, ref_phis


def train(
    sample_path: str,
    num_epochs: int = 200,
    batch_size: int = 8,
    n_transforms: int = 4,
    steps_per_epoch: int = 64,
    lr: float = 1e-4,
    anchor_weight: float = 5.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    out_dir: str = "lodestar_orientation_test_out",
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    samples = _load_training_samples(sample_path)
    print(f"Loaded {len(samples)} training sample(s):")
    for s, phi_r in samples:
        print(f"  {s.shape[-2]}×{s.shape[-1]}  φ={np.degrees(phi_r):.1f}°")
    transforms = make_orientation_transforms()
    model = LodeSTAR(
        num_outputs=4,
        transforms=transforms,
        n_transforms=n_transforms,
    )
    if hasattr(model, "build"):
        model = model.build()
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(num_epochs):
        model.train()
        ep_ssl, ep_anchor = 0.0, 0.0
        for _ in range(steps_per_epoch):
            batch, angles, ref_phis = _make_augmented_batch(samples, batch_size, device)
            transformed, inverse_fn = model.transform_data(batch)
            y_hat = model(transformed)
            loss_dict = model.compute_loss(y_hat, inverse_fn)
            y_direct = model(batch)
            cos_ch = y_direct[:, 2]
            sin_ch = y_direct[:, 3]
            Hy, Wy = cos_ch.shape[1], cos_ch.shape[2]
            cy_out, cx_out = Hy // 2, Wy // 2
            ar = 2
            r0, r1 = max(0, cy_out - ar), min(Hy, cy_out + ar + 1)
            c0, c1 = max(0, cx_out - ar), min(Wy, cx_out + ar + 1)
            anchor_loss = torch.tensor(0.0, device=device)
            for b in range(batch_size):
                exp_phi = ref_phis[b] + angles[b]
                exp_cos = float(np.cos(exp_phi))
                exp_sin = float(np.sin(exp_phi))
                avg_cos = cos_ch[b, r0:r1, c0:c1].mean()
                avg_sin = sin_ch[b, r0:r1, c0:c1].mean()
                anchor_loss = anchor_loss + (avg_cos - exp_cos) ** 2 + (avg_sin - exp_sin) ** 2
            anchor_loss = anchor_loss / batch_size
            loss_dict["orientation_anchor"] = anchor_loss * anchor_weight
            loss = sum(loss_dict.values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_ssl += loss_dict["between_image_disagreement"].item() + loss_dict["within_image_disagreement"].item()
            ep_anchor += anchor_loss.item()
        n = steps_per_epoch
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"epoch {ep+1}/{num_epochs} ssl={ep_ssl/n:.4f} anchor={ep_anchor/n:.4f}")
    ckpt = os.path.join(out_dir, "orientation_ckpt.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"Saved {ckpt}")
    return ckpt


def train_pos(
    sample_path: str,
    num_epochs: int = 200,
    batch_size: int = 8,
    n_transforms: int = 4,
    steps_per_epoch: int = 64,
    lr: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    out_dir: str = "lodestar_orientation_test_out",
    max_samples: Optional[int] = None,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    samples = _load_training_samples(sample_path, max_samples=max_samples)
    print(f"Loaded {len(samples)} training sample(s) for position-only model")
    transforms = make_orientation_transforms()
    model = LodeSTAR(num_outputs=2, transforms=transforms, n_transforms=n_transforms)
    if hasattr(model, "build"):
        model = model.build()
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(num_epochs):
        model.train()
        ep_ssl = 0.0
        for _ in range(steps_per_epoch):
            batch, _, _ = _make_augmented_batch(samples, batch_size, device)
            transformed, inverse_fn = model.transform_data(batch)
            y_hat = model(transformed)
            loss_dict = model.compute_loss(y_hat, inverse_fn)
            loss = sum(loss_dict.values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_ssl += loss.item()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"epoch {ep+1}/{num_epochs} ssl={ep_ssl/steps_per_epoch:.4f}")
    ckpt = os.path.join(out_dir, "pos_ckpt.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"Saved {ckpt}")
    return ckpt


def _draw_orientation_arrow(ax: Any, x: float, y: float, phi: float, scale: float, color: str) -> None:
    dx = scale * np.cos(phi)
    dy = scale * np.sin(phi)
    ax.arrow(x, y, dx, dy, head_width=scale * 0.3, head_length=scale * 0.2, fc=color, ec=color, linewidth=1.5)
    ax.plot(x, y, "o", color=color, markersize=4)


def _link_frames(
    prev_pos: np.ndarray,
    curr_pos: np.ndarray,
    max_link_dist: float,
) -> dict[int, int]:
    links: dict[int, int] = {}
    if len(prev_pos) == 0 or len(curr_pos) == 0:
        return links
    used = np.zeros(len(prev_pos), dtype=bool)
    for j in range(len(curr_pos)):
        best_i, best_d2 = None, float("inf")
        for i in range(len(prev_pos)):
            if used[i]:
                continue
            d2 = (curr_pos[j, 0] - prev_pos[i, 0]) ** 2 + (curr_pos[j, 1] - prev_pos[i, 1]) ** 2
            if d2 < best_d2:
                best_d2, best_i = d2, i
        if best_i is not None and best_d2 < max_link_dist ** 2:
            used[best_i] = True
            links[j] = best_i
    return links


def _run_detection(
    ckpt_path: str,
    csv_path: str,
    images_dir: str | None,
    alpha: float,
    beta: float,
    cutoff: float,
    mode: str,
    device: str,
    min_distance: float,
) -> tuple[dict, dict, list[int], "LodeSTAR"]:
    import matplotlib.pyplot as plt
    from PIL import Image
    if images_dir is None:
        images_dir = os.path.join(os.path.dirname(os.path.dirname(csv_path)), "images")
    video_id = os.path.basename(csv_path).replace("_video.csv", "")
    gt = load_csv_gt(csv_path)
    transforms = make_orientation_transforms()
    ckpt_state = torch.load(ckpt_path, map_location=device)
    final_key = [k for k in ckpt_state.keys() if "weight" in k][-1]
    ckpt_out = ckpt_state[final_key].shape[0]
    num_out = ckpt_out - 1
    model = LodeSTAR(num_outputs=num_out, transforms=transforms, n_transforms=4)
    if hasattr(model, "build"):
        model = model.build()
    print(f"Checkpoint outputs: {ckpt_out} -> num_outputs={num_out}")
    model.load_state_dict(ckpt_state)
    model = model.to(device).eval()
    frame_indices = sorted(gt.keys())
    frame_data: dict[int, dict] = {}
    for image_idx in frame_indices:
        frame_info = gt[image_idx]
        fname = f"{video_id}_{image_idx + 1:03d}.png"
        img_path = os.path.join(images_dir, fname)
        if not os.path.isfile(img_path):
            for ext in (".jpg", ".tif", ".tiff"):
                p = os.path.join(images_dir, fname.replace(".png", ext))
                if os.path.isfile(p):
                    img_path = p
                    break
        if not os.path.isfile(img_path):
            continue
        img = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
        x = torch.from_numpy(img[np.newaxis, np.newaxis, :, :]).float().to(device)
        with torch.no_grad():
            y = model(x)
        y_pred, weights = y[:, :-1], y[:, -1:]
        w_np = weights[0, 0].detach().cpu().numpy()
        if w_np.shape[0] != img.shape[0] or w_np.shape[1] != img.shape[1]:
            w_resized = torch.nn.functional.interpolate(
                weights, size=(img.shape[0], img.shape[1]), mode="bilinear", align_corners=False
            )
            w_np = w_resized[0, 0].detach().cpu().numpy()
        border = 30
        H_out, W_out = weights.shape[-2], weights.shape[-1]
        weights_masked = weights.clone()
        scale_h = img.shape[0] / H_out
        border_out = max(1, int(border / scale_h))
        weights_masked[..., :border_out, :] = 0
        weights_masked[..., -border_out:, :] = 0
        weights_masked[..., :, :border_out] = 0
        weights_masked[..., :, -border_out:] = 0
        dets_raw, det_scores = detect_single_with_smooth_orientation(
            model, y_pred[0], weights_masked[0], alpha, beta, cutoff, mode, window=1,
        )
        n_before = len(dets_raw)
        refined_all = np.empty((n_before, 2), dtype=np.float64)
        cos_sin_raw = np.zeros((n_before, 2), dtype=np.float64)
        for i in range(n_before):
            raw_x, raw_y = float(dets_raw[i, 1]), float(dets_raw[i, 0])
            rx, ry = refine_position_to_center(img, raw_x, raw_y, search_radius=25)
            refined_all[i] = [rx, ry]
            if dets_raw.shape[1] >= 4:
                cos_sin_raw[i] = [float(dets_raw[i, 2]), float(dets_raw[i, 3])]
        if n_before > 0:
            keep_idx = _suppress_nearby(refined_all, det_scores, min_distance)
        else:
            keep_idx = np.arange(0, dtype=int)
        refined = refined_all[keep_idx]
        cos_sin_kept = cos_sin_raw[keep_idx]
        if image_idx == frame_indices[0]:
            print(f"Frame 1: {n_before} raw -> {len(refined)} after suppression (min_dist={min_distance})")
        frame_data[image_idx] = {
            "refined": refined, "cos_sin": cos_sin_kept,
            "img": img, "w_np": w_np, "gt": frame_info,
        }
    return frame_data, gt, sorted(frame_data.keys()), model


def test_4ch(
    ckpt_path: str,
    csv_path: str,
    images_dir: str | None = None,
    alpha: float = 0.6,
    beta: float = 0.4,
    cutoff: float = 0.2,
    mode: str = "constant",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    vis_dir: str | None = "lodestar_orientation_test_out/vis_4ch",
    vis_max_frames: int = 20,
    out_csv: str | None = "lodestar_orientation_test_out/detections_4ch.csv",
    min_distance: float = 10.0,
) -> None:
    import matplotlib.pyplot as plt
    frame_data, gt, sorted_keys, model = _run_detection(
        ckpt_path, csv_path, images_dir, alpha, beta, cutoff, mode, device, min_distance,
    )
    if not frame_data:
        return
    video_id = os.path.basename(csv_path).replace("_video.csv", "")
    arrow_scale = 15.0
    all_detections: list[dict] = []
    orient_errors: list[float] = []
    for image_idx in sorted_keys:
        fd = frame_data[image_idx]
        refined, cos_sin, gt_info = fd["refined"], fd["cos_sin"], fd["gt"]
        for i in range(len(refined)):
            phi_4ch = float(np.arctan2(cos_sin[i, 1], cos_sin[i, 0]))
            all_detections.append({
                "frame": image_idx + 1, "x": float(refined[i, 0]), "y": float(refined[i, 1]),
                "cos": float(cos_sin[i, 0]), "sin": float(cos_sin[i, 1]), "phi": phi_4ch,
            })
        used = np.zeros(len(refined), dtype=bool)
        for i in range(len(gt_info["positions"])):
            gx, gy = float(gt_info["positions"][i, 0]), float(gt_info["positions"][i, 1])
            gt_phi = gt_info["phi"][i]
            best_j, best_d2 = None, float("inf")
            for j in range(len(refined)):
                if used[j]:
                    continue
                d2 = (refined[j, 0] - gx) ** 2 + (refined[j, 1] - gy) ** 2
                if d2 < best_d2:
                    best_d2, best_j = d2, j
            if best_j is not None and best_d2 < 10000:
                used[best_j] = True
                pred_phi = np.arctan2(cos_sin[best_j, 1], cos_sin[best_j, 0])
                err = np.abs(np.arctan2(np.sin(pred_phi - gt_phi), np.cos(pred_phi - gt_phi)))
                orient_errors.append(err)
    if orient_errors:
        print(f"4ch Model Orientation MAE: {np.degrees(np.mean(orient_errors)):.2f}°")
    else:
        print("No matched detections for 4ch orientation")
    if all_detections:
        phis = np.array([d["phi"] for d in all_detections])
        mags = np.array([np.sqrt(d["cos"] ** 2 + d["sin"] ** 2) for d in all_detections])
        print(f"Predicted phi: mean={np.degrees(phis.mean()):.1f}°, std={np.degrees(phis.std()):.1f}°, "
              f"raw cos/sin mag: mean={mags.mean():.3f}, min={mags.min():.3f}")
    if out_csv and all_detections:
        det_df = pd.DataFrame(all_detections)
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        det_df.to_csv(out_csv, index=False)
        print(f"Detections saved to: {out_csv}")
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)
        for vi, image_idx in enumerate(sorted_keys):
            if vi >= vis_max_frames:
                break
            fd = frame_data[image_idx]
            img, refined, cos_sin = fd["img"], fd["refined"], fd["cos_sin"]
            gt_info, w_np = fd["gt"], fd["w_np"]
            fig, (ax_img, ax_w) = plt.subplots(1, 2, figsize=(12, 6))
            ax_img.imshow(img / 255.0, cmap="gray")
            for i, (gx, gy) in enumerate(gt_info["positions"]):
                _draw_orientation_arrow(ax_img, float(gx), float(gy), gt_info["phi"][i], arrow_scale, "lime")
            for i in range(len(refined)):
                rx, ry = float(refined[i, 0]), float(refined[i, 1])
                phi_4ch = float(np.arctan2(cos_sin[i, 1], cos_sin[i, 0]))
                _draw_orientation_arrow(ax_img, rx, ry, phi_4ch, arrow_scale, "red")
            ax_img.set_title(f"frame {image_idx + 1}: green=GT, red=4ch model ({len(refined)} dets)")
            ax_img.axis("off")
            w_disp = w_np - w_np.min()
            w_disp /= (w_disp.max() + 1e-8)
            ax_w.imshow(w_disp, cmap="hot", vmin=0, vmax=1)
            ax_w.set_title("weight map")
            ax_w.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"{video_id}_{image_idx + 1:03d}_orientation.png"), dpi=120, bbox_inches="tight")
            plt.close()
        print(f"Visualizations saved to: {vis_dir}")


def test(
    ckpt_path: str,
    csv_path: str,
    images_dir: str | None = None,
    alpha: float = 0.6,
    beta: float = 0.4,
    cutoff: float = 0.2,
    mode: str = "constant",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    vis_dir: str | None = "lodestar_orientation_test_out/vis_vel",
    vis_max_frames: int = 20,
    out_csv: str | None = "lodestar_orientation_test_out/detections_vel.csv",
    min_distance: float = 10.0,
    max_link_dist: float = 50.0,
) -> None:
    import matplotlib.pyplot as plt
    frame_data, gt, sorted_keys, model = _run_detection(
        ckpt_path, csv_path, images_dir, alpha, beta, cutoff, mode, device, min_distance,
    )
    if not frame_data:
        return
    video_id = os.path.basename(csv_path).replace("_video.csv", "")
    arrow_scale = 15.0
    print(f"Detected centers in {len(frame_data)} frames. Computing velocity orientation...")

    for fd in frame_data.values():
        fd["phi"] = np.full(len(fd["refined"]), np.nan)

    for ki in range(len(sorted_keys) - 1):
        k_curr, k_next = sorted_keys[ki], sorted_keys[ki + 1]
        pos_curr = frame_data[k_curr]["refined"]
        pos_next = frame_data[k_next]["refined"]
        links = _link_frames(pos_curr, pos_next, max_link_dist)
        for j_next, i_curr in links.items():
            dx = pos_next[j_next, 0] - pos_curr[i_curr, 0]
            dy = pos_next[j_next, 1] - pos_curr[i_curr, 1]
            if dx * dx + dy * dy > 0.25:
                frame_data[k_curr]["phi"][i_curr] = np.arctan2(dy, dx)

    n_with = sum(int(np.isfinite(fd["phi"]).sum()) for fd in frame_data.values())
    n_total = sum(len(fd["refined"]) for fd in frame_data.values())
    print(f"Velocity orientation: {n_with}/{n_total} detections")

    all_detections: list[dict] = []
    orient_errors: list[tuple[str, float]] = []
    for image_idx in sorted_keys:
        fd = frame_data[image_idx]
        refined, phi_arr, gt_info = fd["refined"], fd["phi"], fd["gt"]
        for i in range(len(refined)):
            row: dict = {"frame": image_idx + 1, "x": float(refined[i, 0]), "y": float(refined[i, 1])}
            if np.isfinite(phi_arr[i]):
                row["phi"] = float(phi_arr[i])
            all_detections.append(row)
        used = np.zeros(len(refined), dtype=bool)
        for i in range(len(gt_info["positions"])):
            gx, gy = float(gt_info["positions"][i, 0]), float(gt_info["positions"][i, 1])
            gt_phi = gt_info["phi"][i]
            best_j, best_d2 = None, float("inf")
            for j in range(len(refined)):
                if used[j]:
                    continue
                d2 = (refined[j, 0] - gx) ** 2 + (refined[j, 1] - gy) ** 2
                if d2 < best_d2:
                    best_d2, best_j = d2, j
            if best_j is not None and best_d2 < 10000:
                used[best_j] = True
                if np.isfinite(phi_arr[best_j]):
                    err = np.abs(np.arctan2(np.sin(phi_arr[best_j] - gt_phi), np.cos(phi_arr[best_j] - gt_phi)))
                    orient_errors.append(("velocity", err))

    if orient_errors:
        vel_errs = [e for _, e in orient_errors]
        print(f"Velocity Orientation MAE: {np.degrees(np.mean(vel_errs)):.2f}°")
    else:
        print("No matched detections with velocity orientation")

    if out_csv and all_detections:
        det_df = pd.DataFrame(all_detections)
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        det_df.to_csv(out_csv, index=False)
        print(f"Detections saved to: {out_csv}")

    if vis_dir:
        for vi, image_idx in enumerate(sorted_keys):
            if vi >= vis_max_frames:
                break
            fd = frame_data[image_idx]
            img, refined, phi_arr = fd["img"], fd["refined"], fd["phi"]
            gt_info, w_np = fd["gt"], fd["w_np"]
            fig, (ax_img, ax_w) = plt.subplots(1, 2, figsize=(12, 6))
            ax_img.imshow(img / 255.0, cmap="gray")
            for i, (gx, gy) in enumerate(gt_info["positions"]):
                _draw_orientation_arrow(ax_img, float(gx), float(gy), gt_info["phi"][i], arrow_scale, "lime")
            for i in range(len(refined)):
                rx, ry = float(refined[i, 0]), float(refined[i, 1])
                if np.isfinite(phi_arr[i]):
                    _draw_orientation_arrow(ax_img, rx, ry, phi_arr[i], arrow_scale, "red")
                else:
                    ax_img.plot(rx, ry, "o", color="red", markersize=5, markeredgecolor="white", markeredgewidth=0.5)
            ax_img.set_title(f"frame {image_idx + 1}: green=GT, red=velocity ({len(refined)} dets)")
            ax_img.axis("off")
            w_disp = w_np - w_np.min()
            w_disp /= (w_disp.max() + 1e-8)
            ax_w.imshow(w_disp, cmap="hot", vmin=0, vmax=1)
            ax_w.set_title("weight map")
            ax_w.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"{video_id}_{image_idx + 1:03d}_orientation.png"), dpi=120, bbox_inches="tight")
            plt.close()
        print(f"Visualizations saved to: {vis_dir}")


def test_on_sample(ckpt_path: str, sample_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    import matplotlib.pyplot as plt
    transforms = make_orientation_transforms()
    model = LodeSTAR(num_outputs=4, transforms=transforms, n_transforms=4)
    if hasattr(model, "build"):
        model = model.build()
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device).eval()
    ref_phi_deg = _parse_phi_from_filename(sample_path)
    raw = load_sample_as_numpy(sample_path)
    sample_t = torch.from_numpy(raw).permute(2, 0, 1).unsqueeze(0).float()
    padded = _pad_sample(sample_t)
    x = padded.to(device)
    print(f"Sample {raw.shape[:2]} padded to {padded.shape[-2:]}, ref φ={ref_phi_deg:.1f}° (from filename)")
    print("Rotation diagnostic:")
    errors = []
    for ang_deg in range(0, 360, 30):
        rotated = kornia.geometry.transform.rotate(
            x, torch.tensor([float(ang_deg)], device=x.device),
            align_corners=True, padding_mode="reflection",
        )
        with torch.no_grad():
            yr = model(rotated)
        Hy, Wy = yr.shape[2], yr.shape[3]
        c_val = yr[0, 2, Hy // 2, Wy // 2].item()
        s_val = yr[0, 3, Hy // 2, Wy // 2].item()
        mag = np.sqrt(c_val ** 2 + s_val ** 2)
        pred_deg = np.degrees(np.arctan2(s_val, c_val))
        exp_deg = ref_phi_deg + ang_deg
        err = abs(np.degrees(np.arctan2(np.sin(np.radians(pred_deg - exp_deg)), np.cos(np.radians(pred_deg - exp_deg)))))
        errors.append(err)
        print(f"  rot={ang_deg:>3d}°  pred={pred_deg:>7.1f}°  exp={exp_deg:>7.1f}°  err={err:>5.1f}°  mag={mag:.3f}")
    print(f"  Mean rotation error: {np.mean(errors):.1f}°")


def _estimate_annular_mask(template_2d: np.ndarray) -> tuple[int, int]:
    h, w = template_2d.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_r = min(h, w) // 2 - 1
    bg = template_2d[dist > max_r * 0.9].mean() if np.any(dist > max_r * 0.9) else 0.0
    peak = template_2d.max()
    threshold = bg + (peak - bg) * 0.15
    profile = np.zeros(max_r + 1)
    for r_i in range(max_r + 1):
        ring = (dist >= r_i - 0.5) & (dist <= r_i + 0.5)
        if ring.any():
            profile[r_i] = template_2d[ring].mean()
    r_inner = 0
    for r_i in range(max_r):
        if profile[r_i] > threshold:
            r_inner = max(0, r_i - 1)
            break
    r_outer = max_r
    for r_i in range(max_r, 0, -1):
        if profile[r_i] > threshold:
            r_outer = min(max_r, r_i + 1)
            break
    return int(r_inner), int(r_outer)


def test_template(
    ckpt_path: str,
    csv_path: str,
    sample_path: str,
    images_dir: str | None = None,
    alpha: float = 0.6,
    beta: float = 0.4,
    cutoff: float = 0.2,
    mode: str = "constant",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    vis_dir: str | None = "lodestar_orientation_test_out/vis_template",
    vis_max_frames: int = 20,
    out_csv: str | None = "lodestar_orientation_test_out/detections_template.csv",
    min_distance: float = 10.0,
    angle_step: int = 2,
) -> None:
    import matplotlib.pyplot as plt
    template_phi_deg = _parse_phi_from_filename(sample_path)
    template_raw = load_sample_as_numpy(sample_path)
    template_2d = (template_raw[:, :, 0] if template_raw.ndim == 3 else template_raw).astype(np.float64)
    th, tw = template_2d.shape
    cy, cx = th // 2, tw // 2
    r_inner, r_outer = _estimate_annular_mask(template_2d)
    yy, xx = np.mgrid[:th, :tw]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = (dist2 >= r_inner ** 2) & (dist2 <= r_outer ** 2)
    print(f"Annular mask: r_inner={r_inner}, r_outer={r_outer} ({mask.sum()} pixels)")
    angles_arr = np.arange(0, 360, angle_step)
    normed_templates = []
    for a in angles_arr:
        rot = ndimage.rotate(template_2d, float(a), reshape=False, mode='reflect')
        rm = rot[mask] - rot[mask].mean()
        nrm = np.sqrt(np.sum(rm ** 2))
        if nrm < 1e-6:
            normed_templates.append(np.zeros_like(rm))
            continue
        normed_templates.append(rm / nrm)
    normed_templates = np.array(normed_templates)
    self_scores = normed_templates @ normed_templates[0]
    print(f"Self-test NCC: max={self_scores.max():.4f} at 0°, "
          f"min={self_scores.min():.4f} at {angles_arr[np.argmin(self_scores)]}°, "
          f"contrast={self_scores.max() - self_scores.min():.4f}")
    print(f"Template {th}×{tw}, φ={template_phi_deg:.1f}°, {len(angles_arr)} rotations (step={angle_step}°)")
    frame_data, gt, sorted_keys, model = _run_detection(
        ckpt_path, csv_path, images_dir, alpha, beta, cutoff, mode, device, min_distance,
    )
    if not frame_data:
        return
    video_id = os.path.basename(csv_path).replace("_video.csv", "")
    arrow_scale = 15.0
    all_detections: list[dict] = []
    orient_errors: list[float] = []
    orient_errors_filtered: list[float] = []
    ncc_scores_all: list[float] = []
    for image_idx in sorted_keys:
        fd = frame_data[image_idx]
        refined, gt_info, img = fd["refined"], fd["gt"], fd["img"]
        img_f64 = img.astype(np.float64)
        phis = np.full(len(refined), np.nan)
        ncc_per_det = np.full(len(refined), np.nan)
        ih, iw = img_f64.shape[:2]
        search_r = 5
        padded_img = np.pad(img_f64, ((th, th), (tw, tw)), mode='reflect')
        for i in range(len(refined)):
            rx, ry = float(refined[i, 0]), float(refined[i, 1])
            best_ncc_global, best_angle_global = -2.0, 0
            for dx in range(-search_r, search_r + 1):
                for dy in range(-search_r, search_r + 1):
                    cx_shifted = rx + dx
                    cy_shifted = ry + dy
                    x0 = int(round(cx_shifted - tw / 2)) + tw
                    y0 = int(round(cy_shifted - th / 2)) + th
                    patch = padded_img[y0:y0 + th, x0:x0 + tw]
                    if patch.shape[0] != th or patch.shape[1] != tw:
                        continue
                    pm = patch[mask] - patch[mask].mean()
                    pnorm = np.sqrt(np.sum(pm ** 2))
                    if pnorm < 1e-6:
                        continue
                    scores = normed_templates @ (pm / pnorm)
                    local_best = int(np.argmax(scores))
                    if scores[local_best] > best_ncc_global:
                        best_ncc_global = float(scores[local_best])
                        best_angle_global = local_best
            if best_ncc_global > -1.0:
                ncc_scores_all.append(best_ncc_global)
                ncc_per_det[i] = best_ncc_global
                phis[i] = np.radians(template_phi_deg - float(angles_arr[best_angle_global]))
            row: dict = {"frame": image_idx + 1, "x": rx, "y": ry}
            if np.isfinite(phis[i]):
                row["phi"] = float(phis[i])
                row["ncc"] = float(ncc_per_det[i])
            all_detections.append(row)
        fd["template_phi"] = phis
        fd["ncc"] = ncc_per_det
        match_dist_thresh = min_distance * 1.5
        used = np.zeros(len(refined), dtype=bool)
        for i in range(len(gt_info["positions"])):
            gx, gy = float(gt_info["positions"][i, 0]), float(gt_info["positions"][i, 1])
            gt_phi = gt_info["phi"][i]
            best_j, best_d2 = None, float("inf")
            for j in range(len(refined)):
                if used[j]:
                    continue
                d2 = (refined[j, 0] - gx) ** 2 + (refined[j, 1] - gy) ** 2
                if d2 < best_d2:
                    best_d2, best_j = d2, j
            if best_j is not None and best_d2 < match_dist_thresh ** 2 and np.isfinite(phis[best_j]):
                used[best_j] = True
                err = np.abs(np.arctan2(np.sin(phis[best_j] - gt_phi), np.cos(phis[best_j] - gt_phi)))
                orient_errors.append(err)
                if ncc_per_det[best_j] >= 0.5:
                    orient_errors_filtered.append(err)
    if ncc_scores_all:
        ncc_arr = np.array(ncc_scores_all)
        print(f"NCC scores: mean={ncc_arr.mean():.4f}, std={ncc_arr.std():.4f}, "
              f"min={ncc_arr.min():.4f}, max={ncc_arr.max():.4f}")
    if orient_errors:
        errs_deg = np.degrees(orient_errors)
        print(f"Template Orientation MAE: {errs_deg.mean():.2f}° (median={np.median(errs_deg):.2f}°, "
              f"<20°: {(errs_deg<20).sum()}/{len(errs_deg)}, matched={len(errs_deg)})")
        if orient_errors_filtered:
            ef_deg = np.degrees(orient_errors_filtered)
            print(f"  NCC≥0.5 filtered MAE: {ef_deg.mean():.2f}° (n={len(ef_deg)})")
    else:
        print("No matched detections for template orientation")
    if all_detections:
        valid = [d["phi"] for d in all_detections if "phi" in d]
        if valid:
            print(f"Predicted phi: mean={np.degrees(np.mean(valid)):.1f}°, std={np.degrees(np.std(valid)):.1f}°")
    if out_csv and all_detections:
        det_df = pd.DataFrame(all_detections)
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        det_df.to_csv(out_csv, index=False)
        print(f"Detections saved to: {out_csv}")
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)
        for vi, image_idx in enumerate(sorted_keys):
            if vi >= vis_max_frames:
                break
            fd = frame_data[image_idx]
            img, refined, gt_info, w_np = fd["img"], fd["refined"], fd["gt"], fd["w_np"]
            phis = fd["template_phi"]
            fig, (ax_img, ax_w) = plt.subplots(1, 2, figsize=(12, 6))
            ax_img.imshow(img / 255.0, cmap="gray")
            for i, (gx, gy) in enumerate(gt_info["positions"]):
                _draw_orientation_arrow(ax_img, float(gx), float(gy), gt_info["phi"][i], arrow_scale, "lime")
            for i in range(len(refined)):
                rx, ry = float(refined[i, 0]), float(refined[i, 1])
                if np.isfinite(phis[i]):
                    _draw_orientation_arrow(ax_img, rx, ry, phis[i], arrow_scale, "red")
                else:
                    ax_img.plot(rx, ry, "o", color="red", markersize=5)
            ax_img.set_title(f"frame {image_idx + 1}: green=GT, red=template ({len(refined)} dets)")
            ax_img.axis("off")
            w_bg = np.median(w_np)
            w_disp = np.clip(w_np - w_bg, 0, None)
            w_disp /= (w_disp.max() + 1e-8)
            ax_w.imshow(w_disp, cmap="hot", vmin=0, vmax=1)
            ax_w.set_title("weight map")
            ax_w.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"{video_id}_{image_idx + 1:03d}_orientation.png"), dpi=120, bbox_inches="tight")
            plt.close()
        print(f"Visualizations saved to: {vis_dir}")


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="LodeSTAR position + template orientation")
    p.add_argument("--train", action="store_true", help="Run 4ch training (position+orientation)")
    p.add_argument("--train-pos", action="store_true", help="Train 2ch position-only model -> pos_ckpt.pt")
    p.add_argument("--test", action="store_true", help="Run test (velocity orientation) -> vis_vel")
    p.add_argument("--test-4ch", action="store_true", help="Run test (4ch model cos/sin) -> vis_4ch")
    p.add_argument("--test-template", action="store_true", help="Run test (template matching orientation) -> vis_template")
    p.add_argument("--test-batch", action="store_true", help="Run template matching on all CSVs in csv directory")
    p.add_argument("--test-sample", action="store_true", help="Test on training sample")
    p.add_argument("--sample", type=str, default="data/Samples/JP_Fe_wf_2_40/JP_Fe_wf_2_40_phi0112.6.png")
    p.add_argument("--csv", type=str, default="data/JP_FE/wf_2_40/04/csv/JP_Fe_wf_2_40_slm075_574_video.csv")
    p.add_argument("--ckpt", type=str, default="lodestar_orientation_test_out/orientation_ckpt.pt")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--steps-per-epoch", type=int, default=64)
    p.add_argument("--anchor-weight", type=float, default=5.0, help="Weight for orientation anchor loss")
    p.add_argument("--out-dir", type=str, default="lodestar_orientation_test_out")
    p.add_argument("--vis-dir", type=str, default="lodestar_orientation_test_out/vis_vel", help="Save orientation plots here; set '' to disable")
    p.add_argument("--vis-max-frames", type=int, default=20, help="Max frames to plot when --vis-dir is set")
    p.add_argument("--out-csv", type=str, default="lodestar_orientation_test_out/detections.csv", help="Save detections to CSV")
    p.add_argument("--cutoff", type=float, default=0.2, help="Weight map cutoff for detection")
    p.add_argument("--min-distance", type=float, default=10.0, help="Min pixel distance between detections (suppress duplicates)")
    p.add_argument("--max-link-dist", type=float, default=50.0, help="Max pixel distance for linking particles between frames")
    p.add_argument("--single-sample", action="store_true", help="Train with only --sample (same as --num-samples 1)")
    p.add_argument("--num-samples", type=int, default=None, dest="num_samples",
                   help="Use at most N PNGs from Samples/ sorted by name; omit for all. Implies single file if 1.")
    args = p.parse_args()
    sample_path = os.path.join(_REPO_ROOT, args.sample) if not os.path.isabs(args.sample) else args.sample
    csv_path = os.path.join(_REPO_ROOT, args.csv) if not os.path.isabs(args.csv) else args.csv
    if args.train:
        if not os.path.isfile(sample_path):
            raise FileNotFoundError(sample_path)
        train(sample_path, num_epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
              anchor_weight=args.anchor_weight, out_dir=args.out_dir)
    if args.train_pos:
        if not os.path.isfile(sample_path):
            raise FileNotFoundError(sample_path)
        lim: Optional[int] = 1 if args.single_sample else args.num_samples
        train_pos(sample_path, num_epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
                  out_dir=args.out_dir, max_samples=lim)
    if args.test_sample:
        ckpt = os.path.join(_REPO_ROOT, args.ckpt) if not os.path.isabs(args.ckpt) else args.ckpt
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(ckpt)
        test_on_sample(ckpt, sample_path)
    if args.test_4ch:
        ckpt = os.path.join(_REPO_ROOT, args.ckpt) if not os.path.isabs(args.ckpt) else args.ckpt
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(ckpt)
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(csv_path)
        vis_4ch = os.path.join(_REPO_ROOT, "lodestar_orientation_test_out/vis_4ch")
        out_4ch = os.path.join(_REPO_ROOT, "lodestar_orientation_test_out/detections_4ch.csv")
        test_4ch(ckpt, csv_path, cutoff=args.cutoff, vis_dir=vis_4ch, vis_max_frames=args.vis_max_frames, out_csv=out_4ch, min_distance=args.min_distance)
    if args.test_template:
        ckpt = os.path.join(_REPO_ROOT, args.ckpt) if not os.path.isabs(args.ckpt) else args.ckpt
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(ckpt)
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(csv_path)
        vis_tmpl = os.path.join(_REPO_ROOT, "lodestar_orientation_test_out/vis_template")
        out_tmpl = os.path.join(_REPO_ROOT, "lodestar_orientation_test_out/detections_template.csv")
        test_template(ckpt, csv_path, sample_path, cutoff=args.cutoff, vis_dir=vis_tmpl, vis_max_frames=args.vis_max_frames,
                      out_csv=out_tmpl, min_distance=args.min_distance)
    if args.test_batch:
        ckpt = os.path.join(_REPO_ROOT, args.ckpt) if not os.path.isabs(args.ckpt) else args.ckpt
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(ckpt)
        csv_dir = os.path.dirname(csv_path)
        csv_files = sorted([f for f in os.listdir(csv_dir) if f.endswith("_video.csv")])
        if not csv_files:
            raise FileNotFoundError(f"No *_video.csv files in {csv_dir}")
        print(f"Batch: {len(csv_files)} videos in {csv_dir}")
        batch_results: list[dict] = []
        for ci, csv_file in enumerate(csv_files):
            csv_fp = os.path.join(csv_dir, csv_file)
            vid_id = csv_file.replace("_video.csv", "")
            vis_d = os.path.join(_REPO_ROOT, f"lodestar_orientation_test_out/vis_batch/{vid_id}")
            out_c = os.path.join(_REPO_ROOT, f"lodestar_orientation_test_out/batch/{vid_id}_detections.csv")
            print(f"\n[{ci+1}/{len(csv_files)}] {vid_id}")
            test_template(ckpt, csv_fp, sample_path, cutoff=args.cutoff,
                          vis_dir=vis_d, vis_max_frames=3, out_csv=out_c,
                          min_distance=args.min_distance)
        print(f"\nBatch complete: {len(csv_files)} videos processed")
    if args.test:
        ckpt = os.path.join(_REPO_ROOT, args.ckpt) if not os.path.isabs(args.ckpt) else args.ckpt
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(ckpt)
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(csv_path)
        vis_dir: str | None = args.vis_dir or None
        if vis_dir and not os.path.isabs(vis_dir):
            vis_dir = os.path.join(_REPO_ROOT, vis_dir)
        out_csv: str | None = args.out_csv or None
        if out_csv and not os.path.isabs(out_csv):
            out_csv = os.path.join(_REPO_ROOT, out_csv)
        test(ckpt, csv_path, vis_dir=vis_dir, vis_max_frames=args.vis_max_frames, out_csv=out_csv, min_distance=args.min_distance, max_link_dist=args.max_link_dist)


if __name__ == "__main__":
    main()
