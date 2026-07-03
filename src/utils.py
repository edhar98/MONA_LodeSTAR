import os
import re
import yaml
import logging
from jinja2 import Environment, BaseLoader
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from scipy import ndimage
import cv2
import pandas as pd

def calculate_detection_metrics(gt_bboxes, detections, gt_labels=None, detection_labels=None, distance_threshold=20):
    if len(gt_bboxes) == 0 and len(detections) == 0:
        return {'precision': 1.0, 'recall': 1.0, 'f1_score': 1.0, 'tp': 0, 'fp': 0, 'fn': 0}
    
    if len(gt_bboxes) == 0:
        return {'precision': 0.0, 'recall': 1.0, 'f1_score': 0.0, 'tp': 0, 'fp': len(detections), 'fn': 0}
    
    if len(detections) == 0:
        return {'precision': 1.0, 'recall': 0.0, 'f1_score': 0.0, 'tp': 0, 'fp': 0, 'fn': len(gt_bboxes)}
    
    gt_positions = gt_bboxes[:, :2]
    det_positions = detections[:, :2]
    
    distances = cdist(gt_positions, det_positions)
    
    matched_gt = set()
    matched_det = set()
    true_positives = 0
    
    for gt_idx in range(len(gt_positions)):
        for det_idx in range(len(det_positions)):
            if distances[gt_idx, det_idx] <= distance_threshold:
                type_match = True
                if gt_labels is not None and detection_labels is not None:
                    if gt_idx < len(gt_labels) and det_idx < len(detection_labels):
                        type_match = gt_labels[gt_idx] == detection_labels[det_idx]
                
                if type_match and gt_idx not in matched_gt and det_idx not in matched_det:
                    matched_gt.add(gt_idx)
                    matched_det.add(det_idx)
                    true_positives += 1
    
    false_positives = len(detections) - true_positives
    false_negatives = len(gt_bboxes) - true_positives
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'tp': true_positives,
        'fp': false_positives,
        'fn': false_negatives
    }

def _load_config(file_name='samples.yaml'):
    config_path = os.path.join(os.path.dirname(__file__), file_name)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_yaml(path_file):
    """
    Load a YAML file.
    """
    with open(path_file, encoding="utf-8") as f:
        dictionary = yaml.safe_load(f)
    return dictionary


def save_yaml(dictionary, path_file):
    """
    Save a dictionary as a YAML file.
    """
    # Only create directory if path_file contains a directory
    dir_path = os.path.dirname(path_file)
    if dir_path:  # Only create directory if it's not empty
        os.makedirs(dir_path, exist_ok=True)
    with open(path_file, 'w', encoding="utf-8") as f:
        yaml.dump(dictionary, f, default_flow_style=False)


def _parse_stack_detection_path(path, suffix="_detections.csv"):
    name = os.path.basename(str(path))
    pattern = rf"^(?P<prefix>.+)_(?P<stack>\d+){re.escape(suffix)}$"
    match = re.match(pattern, name)
    if not match:
        return None
    return match.group("prefix"), int(match.group("stack"))


def merge_detection_csvs(
    input_dir,
    output_path=None,
    pattern="*_detections.csv",
    frames_per_stack=100,
    add_stack_columns=True,
):
    """Merge per-stack detection CSVs into one global-frame detection CSV.

    Per-stack filenames must end as ``_<stack>_detections.csv``. Local frame
    values are offset by numeric stack order, so the first stack remains
    0..frames_per_stack-1, the second becomes frames_per_stack..2*frames_per_stack-1,
    matching the existing run-level detection CSV convention.
    """
    from pathlib import Path

    input_dir = Path(input_dir)
    parsed = []
    for path in sorted(input_dir.glob(pattern)):
        info = _parse_stack_detection_path(path)
        if info is None:
            continue
        prefix, stack_id = info
        parsed.append((stack_id, prefix, path))

    if not parsed:
        raise FileNotFoundError(
            f"No per-stack detection CSVs matching '*_<stack>_detections.csv' in {input_dir}"
        )

    parsed.sort(key=lambda item: item[0])
    prefixes = {prefix for _, prefix, _ in parsed}
    if len(prefixes) != 1:
        raise ValueError(f"Multiple detection filename prefixes found: {sorted(prefixes)}")
    prefix = parsed[0][1]

    rows = []
    for stack_index, (stack_id, _, path) in enumerate(parsed):
        df = pd.read_csv(path, index_col=0)
        if "frame" not in df.columns:
            raise ValueError(f"Missing required 'frame' column: {path}")
        local_frame = pd.to_numeric(df["frame"], errors="raise").astype(int)
        if add_stack_columns:
            df["stack"] = int(stack_id)
            df["frame_local"] = local_frame
        df["frame"] = local_frame + stack_index * int(frames_per_stack)
        rows.append(df)

    merged = (
        pd.concat(rows, ignore_index=True)
        .sort_values(["frame", "x", "y"] if {"x", "y"}.issubset(rows[0].columns) else ["frame"])
        .reset_index(drop=True)
    )

    if output_path is None:
        output_path = input_dir / f"{prefix}_detections.csv"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path)

    summary = {
        "output_path": str(output_path),
        "input_dir": str(input_dir),
        "n_files": len(parsed),
        "n_rows": int(len(merged)),
        "frame_min": int(merged["frame"].min()) if len(merged) else None,
        "frame_max": int(merged["frame"].max()) if len(merged) else None,
        "stacks": [int(stack_id) for stack_id, _, _ in parsed],
        "frames_per_stack": int(frames_per_stack),
    }
    return merged, summary


def setup_logger(name, log_file=None, level=logging.INFO):
    """Set up logger with console and optional file handler"""
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def parse_xml_annotations(xml_path):
    """Parse XML annotations to get ground truth bounding boxes and SNR"""
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    bboxes = []
    labels = []
    snr = None
    
    # Extract SNR if available
    snr_elem = root.find('snr')
    if snr_elem is not None:
        snr = float(snr_elem.text)
    
    for obj in root.findall('object'):
        label = obj.find('name').text
        bbox = obj.find('bndbox')
        
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        # Convert to center coordinates
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        
        bboxes.append([center_x, center_y])
        labels.append(label)
    
    return np.array(bboxes), labels, snr

class XMLWriter:
    """XML annotation writer for Pascal VOC format"""
    
    def __init__(self, path, width, height, depth=3, database='Unknown', segmented=0):
        # XML template for Pascal VOC format
        self.xml_template = """<annotation>
    <folder>{{ folder }}</folder>
    <filename>{{ filename }}</filename>
    <path>{{ path }}</path>
    <source>
        <database>{{ database }}</database>
    </source>
    <size>
        <width>{{ width }}</width>
        <height>{{ height }}</height>
        <depth>{{ depth }}</depth>
    </size>
    <segmented>{{ segmented }}</segmented>
{% if snr is not none %}    <snr>{{ snr }}</snr>{% endif %}
{% for object in objects %}    <object>
        <name>{{ object.name }}</name>
        <orientation>{{ object.orientation }}</orientation>
        <bndbox>
            <xmin>{{ object.xmin }}</xmin>
            <ymin>{{ object.ymin }}</ymin>
            <xmax>{{ object.xmax }}</xmax>
            <ymax>{{ object.ymax }}</ymax>
        </bndbox>
    </object>{% endfor %}
</annotation>"""
        
        # Create Jinja2 environment
        self.environment = Environment(loader=BaseLoader())
        self.template = self.environment.from_string(self.xml_template)
        
        abspath = os.path.abspath(path)
        
        self.template_parameters = {
            'path': abspath,
            'filename': os.path.basename(abspath),
            'folder': os.path.basename(os.path.dirname(abspath)),
            'width': width,
            'height': height,
            'depth': depth,
            'database': database,
            'segmented': segmented,
            'objects': [],
            'snr': None
        }
    
    def addObject(self, name, xmin, ymin, xmax, ymax, orientation=0):
        """Add an object annotation to the XML"""
        self.template_parameters['objects'].append({
            'name': name,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'orientation': orientation,
        })
    
    def setSNR(self, snr):
        """Set the SNR value for the annotation"""
        self.template_parameters['snr'] = snr
    
    def save(self, annotation_path):
        """Save the XML annotation to file"""
        with open(annotation_path, 'w') as file:
            content = self.template.render(**self.template_parameters)
            file.write(content)

# Alias for backward compatibility
Writer = XMLWriter


def preprocess_image(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    if len(image.shape) == 3:
        if image.shape[-1] == 3:
            image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
        elif image.shape[0] == 3:
            image = np.dot(image[:3].transpose(1, 2, 0), [0.299, 0.587, 0.114])
        elif image.shape[-1] == 1:
            image = image[..., 0]
        elif image.shape[0] == 1:
            image = image[0]
        else:
            image = image[0] if image.shape[0] < image.shape[-1] else image[..., 0]
    elif len(image.shape) > 3:
        if image.shape[1] == 3:
            image = np.dot(image[0].transpose(1, 2, 0), [0.299, 0.587, 0.114])
        else:
            image = image[0, 0] if len(image.shape) == 4 else image[0]
    
    if len(image.shape) != 2:
        raise ValueError(f"Image must be 2D after processing, got shape {image.shape}")
    
    return image


def cluster_nearby_detections(detections: np.ndarray, distance_threshold: float = 20) -> np.ndarray:
    if len(detections) <= 1:
        return detections
    
    tree = cKDTree(detections)
    pairs = tree.query_pairs(r=distance_threshold)
    
    n = len(detections)
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    for i, j in pairs:
        union(i, j)
    
    clusters = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(i)
    
    return np.array([np.mean(detections[indices], axis=0) for indices in clusters.values()])


def nms_detections(detections: np.ndarray, scores: np.ndarray = None,
                   min_distance: float = 15) -> np.ndarray:
    """Non-maximum suppression for particle detections.

    Sorts by confidence (descending), greedily keeps each detection if no
    already-kept detection is within min_distance pixels. Unlike
    cluster_nearby_detections, this preserves both peaks when two particles
    are close but distinct.

    Args:
        detections: (N, 2) array of [x, y] positions
        scores:     (N,) confidence scores; if None all detections are equal
                    and order is preserved
        min_distance: suppress detections within this radius of a kept one

    Returns:
        (M, 2) array of kept positions, M <= N
    """
    if len(detections) <= 1:
        return detections

    if scores is None:
        scores = np.ones(len(detections))

    order = np.argsort(scores)[::-1]
    kept = []
    suppressed = np.zeros(len(detections), dtype=bool)

    for idx in order:
        if suppressed[idx]:
            continue
        kept.append(idx)
        if len(kept) == 1:
            tree = cKDTree(detections)
        neighbours = tree.query_ball_point(detections[idx], r=min_distance)
        for nb in neighbours:
            if nb != idx:
                suppressed[nb] = True

    return detections[np.array(kept)]


def load_csv_ground_truth(csv_path: str) -> dict:
    df = pd.read_csv(csv_path, index_col=0)

    sorted_frames = sorted(df['frame'].unique())

    frames = {}
    for image_idx, frame_val in enumerate(sorted_frames):
        frame_data = df[df['frame'] == frame_val]
        frames[image_idx] = {
            'positions': frame_data[['x', 'y']].values,
            'phi': frame_data['phi'].values,
            'max_intensity': frame_data['max_inensity'].values,
            'summed_intensity': frame_data['summed_inensity'].values,
            'frame': int(frame_val)
        }
    return frames


def detect_by_area(weights: np.ndarray, cutoff: float = 0.9,
                   min_area: int = 100, max_area: int = 2500) -> np.ndarray:
    if weights is None:
        return np.empty((0, 2))
    
    binary_mask = (weights > cutoff).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        detections.append([cx, cy])
    
    return np.array(detections) if detections else np.empty((0, 2))


def detect_by_watershed(weights: np.ndarray, cutoff: float = 0.3,
                        min_distance: int = 10, min_area: int = 20) -> np.ndarray:
    """Detect touching/overlapping particles via distance-transform watershed.

    Steps:
      1. Threshold weight map → binary mask
      2. Distance transform → ridges peak at each particle centre
      3. Local maxima of distance map → per-particle seeds/markers
      4. Watershed on negated distance map, seeded by markers
      5. Centroid of each labelled region
    """
    if weights is None:
        return np.empty((0, 2))

    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max

    binary = (weights > cutoff).astype(np.uint8)
    if binary.sum() == 0:
        return np.empty((0, 2))

    dist = ndimage.distance_transform_edt(binary)

    # peak_local_max returns (row, col) coords of local maxima
    peak_coords = peak_local_max(dist, min_distance=min_distance, labels=binary)
    if len(peak_coords) == 0:
        return np.empty((0, 2))

    markers = np.zeros_like(binary, dtype=np.int32)
    for idx, (r, c) in enumerate(peak_coords, start=1):
        markers[r, c] = idx

    labels = watershed(-dist, markers, mask=binary)

    detections = []
    for label_id in range(1, labels.max() + 1):
        region = labels == label_id
        if region.sum() < min_area:
            continue
        cy, cx = ndimage.center_of_mass(region)
        detections.append([cx, cy])

    return np.array(detections) if detections else np.empty((0, 2))


def save_image_with_detections(image: np.ndarray, detections: np.ndarray, save_path: str,
                               gt_bboxes: np.ndarray = None,
                               det_color: tuple = (255, 0, 0), gt_color: tuple = (0, 255, 0),
                               marker_radius: int = 3, marker_thickness: int = 1,
                               orientations: np.ndarray = None, arrow_length: float = 15.0):
    image = preprocess_image(image)

    if image.max() <= 1.0:
        image_uint8 = (image * 255).astype(np.uint8)
    else:
        image_uint8 = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)

    image_rgb = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)

    if gt_bboxes is not None and len(gt_bboxes) > 0:
        for x, y in gt_bboxes[:, :2]:
            cv2.circle(image_rgb, (int(x), int(y)), marker_radius, gt_color, marker_thickness)

    if len(detections) > 0:
        for i, det in enumerate(detections):
            x, y = det[0], det[1]
            cv2.circle(image_rgb, (int(x), int(y)), marker_radius, det_color, marker_thickness)
            if orientations is not None and i < len(orientations) and np.isfinite(orientations[i]):
                phi = float(orientations[i])
                dx = arrow_length * np.cos(phi)
                dy = arrow_length * np.sin(phi)
                cv2.arrowedLine(
                    image_rgb, (int(x), int(y)),
                    (int(x + dx), int(y + dy)),
                    det_color, max(1, marker_thickness), tipLength=0.3,
                )

    cv2.imwrite(save_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def draw_orientation_arrow(ax, x: float, y: float, phi: float, scale: float = 15.0,
                           color: str = 'red') -> None:
    dx = scale * np.cos(phi)
    dy = scale * np.sin(phi)
    ax.arrow(x, y, dx, dy, head_width=scale * 0.3, head_length=scale * 0.2,
             fc=color, ec=color, linewidth=1.5)
    ax.plot(x, y, 'o', color=color, markersize=4)


def refine_position_to_center(image: np.ndarray, x: float, y: float,
                              search_radius: int = 25, min_radius: int = 3,
                              max_radius: int = None) -> tuple:
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
        patch_uint8, cv2.HOUGH_GRADIENT, dp=1, minDist=1,
        param1=50, param2=18, minRadius=min_radius, maxRadius=max_r,
    )
    if circles is not None and circles.size > 0:
        circles = np.squeeze(circles, axis=0)
        if circles.ndim == 1:
            circles = circles[np.newaxis, :]
        best, best_d2 = None, float('inf')
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


def _estimate_annular_mask(template_2d: np.ndarray) -> tuple:
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


def build_template_bank(sample_path: str, angle_step: int = 2,
                        template_phi_deg: float = None) -> dict:
    if angle_step <= 0:
        raise ValueError("angle_step must be > 0")
    if template_phi_deg is None:
        m = re.search(r'phi(\d+\.?\d*)', os.path.basename(sample_path))
        if m is None:
            raise ValueError(f"Template filename must contain phi<angle> or template_phi_deg must be set: {sample_path}")
        template_phi_deg = float(m.group(1))
    img = cv2.imread(sample_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read template: {sample_path}")
    if img.ndim == 3:
        img = np.dot(img[..., :3], [0.114, 0.587, 0.299])
    template_2d = img.astype(np.float64)
    th, tw = template_2d.shape
    cy, cx = th // 2, tw // 2
    r_inner, r_outer = _estimate_annular_mask(template_2d)
    yy, xx = np.mgrid[:th, :tw]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = (dist2 >= r_inner ** 2) & (dist2 <= r_outer ** 2)
    angles_arr = np.arange(0, 360, angle_step)
    normed = []
    for a in angles_arr:
        rot = ndimage.rotate(template_2d, float(a), reshape=False, mode='reflect')
        rm = rot[mask] - rot[mask].mean()
        nrm = np.sqrt(np.sum(rm ** 2))
        if nrm < 1e-6:
            normed.append(np.zeros_like(rm))
            continue
        normed.append(rm / nrm)
    return {
        'normed_templates': np.array(normed),
        'angles_arr': angles_arr,
        'mask': mask,
        'template_phi_deg': template_phi_deg,
        'th': th,
        'tw': tw,
        'r_inner': r_inner,
        'r_outer': r_outer,
    }


def match_orientations(image: np.ndarray, positions: np.ndarray,
                       template_bank: dict, search_r: int = 5) -> np.ndarray:
    if len(positions) == 0:
        return np.empty((0, 2), dtype=np.float64)
    img = preprocess_image(image).astype(np.float64)
    th = template_bank['th']
    tw = template_bank['tw']
    mask = template_bank['mask']
    normed_templates = template_bank['normed_templates']
    angles_arr = template_bank['angles_arr']
    template_phi_deg = template_bank['template_phi_deg']
    padded_img = np.pad(img, ((th, th), (tw, tw)), mode='reflect')
    out = np.full((len(positions), 2), np.nan, dtype=np.float64)
    for i in range(len(positions)):
        x = float(positions[i, 0])
        y = float(positions[i, 1])
        best_ncc, best_angle = -2.0, 0
        found = False
        for dx in range(-search_r, search_r + 1):
            for dy in range(-search_r, search_r + 1):
                x0 = int(round(x + dx - tw / 2)) + tw
                y0 = int(round(y + dy - th / 2)) + th
                patch = padded_img[y0:y0 + th, x0:x0 + tw]
                if patch.shape[0] != th or patch.shape[1] != tw:
                    continue
                pm = patch[mask] - patch[mask].mean()
                pnorm = np.sqrt(np.sum(pm ** 2))
                if pnorm < 1e-6:
                    continue
                scores = normed_templates @ (pm / pnorm)
                local_best = int(np.argmax(scores))
                if scores[local_best] > best_ncc:
                    best_ncc = float(scores[local_best])
                    best_angle = local_best
                    found = True
        if found:
            out[i, 0] = float(np.radians(template_phi_deg - float(angles_arr[best_angle])))
            out[i, 1] = best_ncc
    return out


def orientation_postprocess(image: np.ndarray, detections: np.ndarray,
                            template_bank: dict, refine_radius: int = 25,
                            search_r: int = 5) -> np.ndarray:
    if refine_radius <= 0:
        raise ValueError("refine_radius must be > 0")
    if search_r < 0:
        raise ValueError("search_r must be >= 0")
    if len(detections) == 0:
        return np.empty((0, 4), dtype=np.float64)
    refined = np.array([
        refine_position_to_center(image, float(d[0]), float(d[1]), search_radius=refine_radius)
        for d in detections
    ], dtype=np.float64)
    phi_ncc = match_orientations(image, refined, template_bank, search_r=search_r)
    return np.column_stack([refined, phi_ncc])


def create_video_from_detections(images_dir: str, output_path: str, fps: int = 10,
                                  extensions: tuple = ('.jpg', '.png', '.tif', '.tiff')) -> str:
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(extensions)])
    
    if not image_files:
        raise ValueError(f"No images found in {images_dir}")
    
    first_image = cv2.imread(os.path.join(images_dir, image_files[0]))
    if first_image is None:
        raise ValueError(f"Could not read first image: {image_files[0]}")
    
    height, width = first_image.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        frame = cv2.imread(image_path)
        if frame is not None:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            video_writer.write(frame)
    
    video_writer.release()
    return output_path
