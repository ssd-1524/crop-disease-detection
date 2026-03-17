import os
import cv2
import numpy as np
import base64
import io
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from PIL import Image, ImageDraw, ImageFont
import uvicorn
from scipy import ndimage as scipy_ndimage 
import sam2

# ── 1. App Setup ───────────────────────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://crop-disease-detection-git-master-ssd-1524s-projects.vercel.app/",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 2. Model Definitions ───────────────────────────────────────────────────────
CLASS_NAMES = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]


class CustomMobileNetV2_3(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.2):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT
        self.model = mobilenet_v2(weights=weights)

        for param in self.model.features.parameters():
            param.requires_grad = False

        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )
        nn.init.xavier_uniform_(self.model.classifier[1].weight)
        nn.init.zeros_(self.model.classifier[1].bias)

    def forward(self, x):
        return self.model(x)


# ── 3. Load Models ─────────────────────────────────────────────────────────────
CLASSIFIER_FILENAME = "CustomMobileNetV2_2_best.pth"
SAM2_CHECKPOINT     = "sam2.1_hiera_large.pt"
SAM2_CONFIG = os.path.join(os.path.dirname(sam2.__file__), "configs/sam2.1/sam2.1_hiera_l.yaml")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classifier = CustomMobileNetV2_3(num_classes=len(CLASS_NAMES))
classifier.load_state_dict(torch.load(CLASSIFIER_FILENAME, map_location=device))
classifier.to(device)
classifier.eval()
print("Classifier loaded successfully.")

sam2_model    = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
sam_predictor = SAM2ImagePredictor(sam2_model)

print("SAM2 loaded successfully.")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def classify_image_with_tta(img_rgb: np.ndarray, n_augments: int = 6) -> tuple[str, str]:
    tta_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    base_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    img_pil   = Image.fromarray(img_rgb)
    all_probs = []

    with torch.inference_mode():
        t = base_transform(img_pil).unsqueeze(0).to(device)
        all_probs.append(torch.softmax(classifier(t), dim=1))

    for _ in range(n_augments):
        with torch.inference_mode():
            t = tta_transforms(img_pil).unsqueeze(0).to(device)
            all_probs.append(torch.softmax(classifier(t), dim=1))

    avg_probs = torch.stack(all_probs).mean(dim=0)[0]
    pred_idx  = avg_probs.argmax().item()
    return CLASS_NAMES[pred_idx], f"{avg_probs[pred_idx].item() * 100:.2f}%"


# ── 4. Disease colour profiles ─────────────────────────────────────────────────
DISEASE_COLOR_PROFILES = {
    "Blight": {
        "hsv_ranges": [
            (np.array([10, 50, 80]),  np.array([25, 200, 220])),
            (np.array([5, 40, 40]),   np.array([15, 180, 160])),
        ],
        "lab_ranges": [
            (np.array([60, 133, 145]), np.array([170, 155, 180])),
        ],
        "exclude_green_s_min":  55,
        "use_green_departure":  True,
        "departure_sensitivity": 0.30,
    },
    "Common_Rust": {
        "hsv_ranges": [
            (np.array([8, 100, 80]),  np.array([22, 255, 255])),
            (np.array([0, 80, 60]),   np.array([8, 255, 220])),
            (np.array([170, 80, 60]), np.array([180, 255, 255])),
        ],
        "lab_ranges": [
            (np.array([50, 140, 150]), np.array([190, 175, 195])),
        ],
        "exclude_green_s_min": 55,
        "use_green_departure": False,
    },
    "Gray_Leaf_Spot": {
        "hsv_ranges": [
            (np.array([15, 20, 100]),  np.array([28, 80, 210])),
            (np.array([5, 30, 50]),    np.array([18, 140, 160])),
            (np.array([0, 0, 150]),    np.array([25, 18, 255])),
        ],
        "lab_ranges": [
            (np.array([85, 128, 140]), np.array([160, 145, 165])),
        ],
        "exclude_green_s_min": 35,
        "use_green_departure": False,
    },
}


def classify_image(img_rgb: np.ndarray) -> tuple[str, str]:
    img_pil = Image.fromarray(img_rgb)
    tensor  = preprocess(img_pil).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = classifier(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
    pred_idx   = probs.argmax().item()
    confidence = f"{probs[pred_idx].item() * 100:.2f}%"
    return CLASS_NAMES[pred_idx], confidence


def segment_full_leaf(img_rgb: np.ndarray) -> np.ndarray:
    h, w  = img_rgb.shape[:2]
    cx, cy = w // 2, h // 2

    pos_points = np.array([
        [cx, cy],
        [cx - w // 6, cy],
        [cx + w // 6, cy],
        [cx, cy - h // 6],
        [cx, cy + h // 6],
    ], dtype=np.float32)

    neg_points = np.array([
        [10, 10], [w - 10, 10], [10, h - 10], [w - 10, h - 10],
    ], dtype=np.float32)

    points = np.concatenate([pos_points, neg_points], axis=0)
    labels = np.array([1] * len(pos_points) + [0] * len(neg_points), dtype=np.int32)

    sam_predictor.set_image(img_rgb)
    with torch.inference_mode():
        masks, scores, _ = sam_predictor.predict(
            point_coords=points, point_labels=labels, multimask_output=True,
        )
    return masks[np.argmax(scores)].astype(np.uint8)


def create_color_mask_within_leaf(
    img_rgb: np.ndarray,
    leaf_mask: np.ndarray,
    disease_class: str = "Common_Rust",
) -> np.ndarray:
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    profile  = DISEASE_COLOR_PROFILES.get(disease_class, DISEASE_COLOR_PROFILES["Common_Rust"])
    combined = np.zeros(img_rgb.shape[:2], dtype=np.uint8)

    for lo, hi in profile["hsv_ranges"]:
        combined |= cv2.inRange(hsv, lo, hi)
    for lo, hi in profile["lab_ranges"]:
        combined |= cv2.inRange(lab, lo, hi)

    if profile.get("use_green_departure", False):
        green_zone = cv2.inRange(hsv, np.array([35, 45, 40]), np.array([85, 255, 255]))
        green_zone = cv2.bitwise_and(green_zone, green_zone, mask=leaf_mask)
        green_saturations = s_ch[green_zone > 0]
        if len(green_saturations) > 100:
            healthy_s_median = float(np.median(green_saturations))
            sensitivity      = profile.get("departure_sensitivity", 0.30)
            s_threshold      = healthy_s_median * sensitivity

            low_sat    = (s_ch.astype(np.float32) < s_threshold).astype(np.uint8) * 255
            not_green  = cv2.bitwise_not(cv2.inRange(h_ch, np.array([35]), np.array([85])))
            bright     = (v_ch > 60).astype(np.uint8) * 255

            dep = cv2.bitwise_and(low_sat, bright)
            dep = cv2.bitwise_and(dep, not_green)
            dep = cv2.bitwise_and(dep, dep, mask=leaf_mask)

            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dep  = cv2.morphologyEx(dep, cv2.MORPH_CLOSE, kern)
            dep  = cv2.morphologyEx(dep, cv2.MORPH_OPEN, kern)

            leaf_area = np.count_nonzero(leaf_mask)
            if leaf_area > 0 and np.count_nonzero(dep) / leaf_area < 0.50:
                combined = cv2.bitwise_or(combined, dep)

    s_min      = profile.get("exclude_green_s_min", 40)
    green_mask = cv2.inRange(hsv, np.array([28, s_min, 30]), np.array([90, 255, 255]))
    combined   = cv2.bitwise_and(combined, cv2.bitwise_not(green_mask))
    combined   = cv2.bitwise_and(combined, combined, mask=leaf_mask)

    kern_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kern_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined   = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kern_close)
    combined   = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kern_open)
    return combined


# ── 5. Improved SAM2 helpers ───────────────────────────────────────────────────

def _mask_to_sam2_logit(binary_mask: np.ndarray) -> np.ndarray:
    """
    Convert a binary (H, W) uint8 mask into a SAM2-compatible low-resolution
    logit tensor of shape (1, 256, 256).

    SAM2's mask_input expects *logit* values (pre-sigmoid) at its internal
    256×256 grid:
      • Disease pixels  → +10.0  (sigmoid ≈ 1.00  — "definitely foreground")
      • Background      →  −5.0  (sigmoid ≈ 0.007 — "likely background",
                                  not -10 so SAM can grow past the colour mask)

    Using +10/-5 rather than ±10 gives SAM freedom to extend the mask into
    pixels the colour threshold missed at edges (under-detection), while still
    anchoring the prediction strongly to the rough mask interior.
    """
    resized = cv2.resize(binary_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    logit   = np.where(resized > 0, 10.0, -5.0).astype(np.float32)
    return logit[np.newaxis, :, :]   # (1, 256, 256)


def _sample_points_in_mask(binary: np.ndarray, n: int) -> np.ndarray:
    """
    Sample up to *n* maximally spread points inside a binary mask using
    the distance transform (peak suppression).
    Returns float32 array of shape (k, 2) as (x, y).
    """
    dist  = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    used  = np.zeros_like(dist)
    pts   = []
    for _ in range(n):
        masked = dist * (1 - used)
        if masked.max() < 1:
            break
        yx = np.unravel_index(masked.argmax(), dist.shape)
        pts.append([yx[1], yx[0]])
        cv2.circle(used, (int(yx[1]), int(yx[0])), max(int(dist[yx]), 8), 1, -1)
    return np.array(pts, dtype=np.float32) if pts else np.empty((0, 2), dtype=np.float32)


def _cluster_components(
    labels: np.ndarray,
    stats: np.ndarray,
    num_labels: int,
    gap_px: int = 40,
    min_area: int = 8,
) -> list[list[int]]:
    """
    Merge connected-component indices whose bounding boxes are within
    *gap_px* pixels of each other into clusters.  Tiny noise components
    (< min_area px) are discarded.

    Returns a list of clusters, each cluster being a list of component
    label indices.  Clustering reduces redundant SAM2 calls and gives
    SAM broader spatial context for densely packed lesions.
    """
    valid = [i for i in range(1, num_labels)
             if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not valid:
        return []

    # Build cluster list with simple greedy merging
    def bbox(i):
        x, y, bw, bh = (stats[i, cv2.CC_STAT_LEFT],
                         stats[i, cv2.CC_STAT_TOP],
                         stats[i, cv2.CC_STAT_WIDTH],
                         stats[i, cv2.CC_STAT_HEIGHT])
        return x, y, x + bw, y + bh   # x0, y0, x1, y1

    clusters = [[valid[0]]]
    for idx in valid[1:]:
        x0i, y0i, x1i, y1i = bbox(idx)
        merged = False
        for cluster in clusters:
            for j in cluster:
                x0j, y0j, x1j, y1j = bbox(j)
                # Check if bounding-box edges are within gap_px
                horiz_gap = max(0, max(x0i, x0j) - min(x1i, x1j))
                vert_gap  = max(0, max(y0i, y0j) - min(y1i, y1j))
                if horiz_gap <= gap_px and vert_gap <= gap_px:
                    cluster.append(idx)
                    merged = True
                    break
            if merged:
                break
        if not merged:
            clusters.append([idx])
    return clusters


def _cluster_bbox(cluster: list[int], stats: np.ndarray,
                  img_w: int, img_h: int, pad: int = 24) -> np.ndarray:
    """Return the padded bounding box enclosing all components in a cluster."""
    x0 = min(stats[i, cv2.CC_STAT_LEFT] for i in cluster)
    y0 = min(stats[i, cv2.CC_STAT_TOP]  for i in cluster)
    x1 = max(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]  for i in cluster)
    y1 = max(stats[i, cv2.CC_STAT_TOP]  + stats[i, cv2.CC_STAT_HEIGHT] for i in cluster)
    return np.array([
        max(x0 - pad, 0), max(y0 - pad, 0),
        min(x1 + pad, img_w - 1), min(y1 + pad, img_h - 1),
    ], dtype=np.float32)


def refine_disease_mask_with_sam2(
    img_rgb: np.ndarray,
    rough_mask: np.ndarray,
    leaf_mask: np.ndarray,
    disease_class: str = "Common_Rust",
    min_area: int = 8,
    spot_threshold: int = 150,
    region_threshold: int = 150,
    cluster_gap_px: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """
    3-pass mask-guided SAM2 refinement.

    For each spatial cluster of rough-mask components:

      Pass 1 — box + distance-transform points only.
               Establishes a baseline prediction independent of colour.

      Pass 2 — box + points + rough colour mask as mask_input (logit form).
               The colour mask acts as a spatial prior, guiding SAM2 to the
               exact disease texture.  SAM2 can grow *beyond* the colour
               mask where texture evidence is present, and can trim false
               colour pixels where it isn't.

      Pass 3 — box + points + pass-2 logits as mask_input.
               Iterative boundary sharpening: feeding SAM2's own logits back
               stabilises the prediction and sharpens uncertain edges.

    Selection:
      • If pass-3 IoU w.r.t. rough mask ≥ 0.25 AND expansion ≤ 2.0 → use pass 3.
      • Elif pass-2 meets same criteria → use pass 2.
      • Else → fall back to pass-1 mask (least SAM drift).
      • If ALL passes fail → keep colour pixels verbatim.

    Small isolated spots (area < spot_threshold) are kept verbatim from the
    colour mask — SAM2 over-expands on sub-20px pustules.

    Both output masks are AND-ed with the leaf boundary.
    """
    h, w = img_rgb.shape[:2]

    # ── Analyse rough mask components ─────────────────────────────────────
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        rough_mask, connectivity=8
    )

    spot_mask   = np.zeros((h, w), dtype=np.uint8)
    region_mask = np.zeros((h, w), dtype=np.uint8)

    if num_labels <= 1:
        # Colour mask found nothing — return empties
        return spot_mask, region_mask

    # Classify components as tiny-noise / small-spots / large-regions
    noise_ids  = []
    spot_ids   = []
    region_ids = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            noise_ids.append(i)
        elif area < spot_threshold:
            spot_ids.append(i)
        else:
            region_ids.append(i)

    # Small spots → keep directly (SAM2 over-expands at this scale)
    for i in spot_ids:
        spot_mask |= (labels == i).astype(np.uint8)

    if not region_ids:
        spot_mask = cv2.bitwise_and(spot_mask, leaf_mask)
        return spot_mask, region_mask

    # ── Eroded leaf mask so SAM can't bleed onto the leaf edge ────────────
    leaf_inset = cv2.erode(
        leaf_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    )
    if np.count_nonzero(leaf_inset) < np.count_nonzero(leaf_mask) * 0.5:
        leaf_inset = leaf_mask   # erosion killed too much — revert

    # ── Cluster nearby large components ───────────────────────────────────
    clusters = _cluster_components(labels, stats, num_labels,
                                   gap_px=cluster_gap_px, min_area=region_threshold)
    # Any region_id not captured by clustering (should not happen, but safety)
    clustered_ids = {idx for c in clusters for idx in c}
    for rid in region_ids:
        if rid not in clustered_ids:
            clusters.append([rid])

    # ── SAM2 3-pass refinement per cluster ────────────────────────────────
    sam_predictor.set_image(img_rgb)

    for cluster in clusters:
        # Build a combined binary mask for this cluster
        cluster_bin = np.zeros((h, w), dtype=np.uint8)
        for i in cluster:
            cluster_bin |= (labels == i).astype(np.uint8)

        cluster_area = int(np.count_nonzero(cluster_bin))
        if cluster_area == 0:
            continue

        box = _cluster_bbox(cluster, stats, w, h, pad=24)

        # Sample interior points spread across the cluster
        pos_points = _sample_points_in_mask(cluster_bin, n=6)
        if len(pos_points) == 0:
            M = cv2.moments(cluster_bin)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                pos_points = np.array([[cx, cy]], dtype=np.float32)
            else:
                region_mask |= cluster_bin
                continue
        pos_labels = np.ones(len(pos_points), dtype=np.int32)

        # Logit encoding of the rough colour mask (for mask_input)
        rough_logit = _mask_to_sam2_logit(cluster_bin)  # (1, 256, 256)

        best_mask   = None
        best_logit  = None

        # ── Pass 1: box + points (baseline) ─────────────────────────────
        try:
            with torch.inference_mode():
                masks1, scores1, logits1 = sam_predictor.predict(
                    point_coords=pos_points,
                    point_labels=pos_labels,
                    box=box,
                    multimask_output=True,
                )
            idx1       = np.argmax(scores1)
            best_mask  = masks1[idx1].astype(np.uint8)
            best_logit = logits1[idx1]   # (1, H_low, W_low) — SAM's own logit
        except Exception:
            region_mask |= cluster_bin
            continue

        # ── Pass 2: box + points + rough colour mask as mask_input ───────
        try:
            with torch.inference_mode():
                masks2, scores2, logits2 = sam_predictor.predict(
                    point_coords=pos_points,
                    point_labels=pos_labels,
                    box=box,
                    mask_input=rough_logit,   # <── colour mask as spatial prior
                    multimask_output=False,
                )
            p2_mask  = masks2[0].astype(np.uint8)
            p2_logit = logits2[0]             # (1, H_low, W_low)
        except Exception:
            p2_mask  = best_mask
            p2_logit = best_logit[None] if best_logit.ndim == 2 else best_logit

        # ── Pass 3: box + points + pass-2 logits (iterative sharpening) ──
        try:
            with torch.inference_mode():
                masks3, scores3, logits3 = sam_predictor.predict(
                    point_coords=pos_points,
                    point_labels=pos_labels,
                    box=box,
                    mask_input=p2_logit,      # <── SAM2's own refined logit
                    multimask_output=False,
                )
            p3_mask = masks3[0].astype(np.uint8)
        except Exception:
            p3_mask = p2_mask

        # ── Select best pass ──────────────────────────────────────────────
        def _score_mask(m: np.ndarray) -> tuple[float, float]:
            """Returns (iou_with_rough, expansion_ratio)."""
            m = cv2.bitwise_and(m, leaf_inset)
            area = int(np.count_nonzero(m))
            inter = int(np.count_nonzero(m & cluster_bin))
            union = int(np.count_nonzero(m | cluster_bin))
            iou   = inter / union if union > 0 else 0.0
            ratio = area / cluster_area if cluster_area > 0 else 999.0
            return iou, ratio

        def _acceptable(iou: float, ratio: float) -> bool:
            return iou >= 0.25 and ratio <= 2.0

        p3_iou, p3_ratio = _score_mask(p3_mask)
        p2_iou, p2_ratio = _score_mask(p2_mask)
        p1_iou, p1_ratio = _score_mask(best_mask)

        if _acceptable(p3_iou, p3_ratio):
            chosen = cv2.bitwise_and(p3_mask, leaf_inset)
        elif _acceptable(p2_iou, p2_ratio):
            chosen = cv2.bitwise_and(p2_mask, leaf_inset)
        elif _acceptable(p1_iou, p1_ratio):
            chosen = cv2.bitwise_and(best_mask, leaf_inset)
        else:
            # All three passes drifted — fall back to colour pixels verbatim
            chosen = cv2.bitwise_and(cluster_bin, leaf_inset)

        region_mask |= chosen

    # ── Constrain to leaf ─────────────────────────────────────────────────
    spot_mask   = cv2.bitwise_and(spot_mask,   leaf_mask)
    region_mask = cv2.bitwise_and(region_mask, leaf_mask)

    return spot_mask, region_mask


# ── 6. Improved Severity Calculation ─────────────────────────────────────────
# Horsfall-Barratt scale anchor points (% area → midpoint of visual category):
#   0    → 0 %     (immune)
#   1    → 3 %     (trace)
#   2    → 6 %     (slight)
#   3    → 12 %    (moderate)
#   4    → 25 %    (fairly severe)
#   5    → 50 %    (severe)
#   6    → 75 %    (very severe)
#   7    → 87 %    (extreme)
#   8    → 94 %    (near-complete)
#   9    → 97 %    (complete)
#   10   → 100 %   (total necrosis)
_HB_ANCHORS = np.array(
    [0, 3, 6, 12, 25, 50, 75, 87, 94, 97, 100], dtype=np.float32
)

def _horsfall_barratt_grade(severity_pct: float) -> int:
    """Map a continuous severity percentage to an HB grade (0–10)."""
    diffs = np.abs(_HB_ANCHORS - severity_pct)
    return int(np.argmin(diffs))


def _severity_label_from_pct(pct: float) -> str:
    if pct < 3:
        return "Trace"
    if pct < 12:
        return "Mild"
    if pct < 25:
        return "Moderate"
    if pct < 50:
        return "Severe"
    return "Very Severe"


def _distribution_label(
    lesion_centroids: np.ndarray,
    leaf_mask: np.ndarray,
) -> str:
    """
    Characterise how uniformly lesions are spread across the leaf.

    Strategy: divide the leaf bounding box into a 3×3 grid, count how many
    of the 9 zones contain at least one lesion centroid.
      ≥ 7 zones → 'Dispersed'   (spreading / advanced)
      4–6 zones → 'Scattered'   (progressing)
      1–3 zones → 'Localised'   (early stage)
      0 centroids → 'None'
    """
    if len(lesion_centroids) == 0:
        return "None"

    ys, xs = np.where(leaf_mask > 0)
    if len(xs) == 0:
        return "Unknown"

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    bw = max(x_max - x_min, 1)
    bh = max(y_max - y_min, 1)

    occupied_zones = set()
    for cx, cy in lesion_centroids:
        col = min(int((cx - x_min) / bw * 3), 2)
        row = min(int((cy - y_min) / bh * 3), 2)
        occupied_zones.add((row, col))

    n = len(occupied_zones)
    if n >= 7:
        return "Dispersed"
    if n >= 4:
        return "Scattered"
    return "Localised"


def calculate_severity(
    leaf_mask: np.ndarray,
    spot_mask: np.ndarray,
    region_mask: np.ndarray,
) -> dict:
    """
    Compute a rich set of severity metrics from the final SAM2-refined masks.

    Metrics returned
    ────────────────
    leaf_area_px          – total leaf pixel area
    disease_area_px       – union of spot + region disease pixels
    severity_pct          – disease_area / leaf_area × 100
    severity_label        – 'Trace' / 'Mild' / 'Moderate' / 'Severe' / 'Very Severe'
    hb_grade              – Horsfall-Barratt grade (0–10)
    severity_score        – normalised 0–100 score (log-weighted so early
                            stages are not collapsed to near-zero)
    spot_count            – number of discrete small lesions
    region_count          – number of large contiguous disease regions
    spot_severity_pct     – spot area / leaf area × 100
    region_severity_pct   – region area / leaf area × 100
    mean_lesion_size_px   – average lesion area (spots + regions combined)
    lesion_density        – lesions per 10,000 leaf pixels (scale-invariant)
    distribution          – 'Localised' / 'Scattered' / 'Dispersed' / 'None'
    spot_mask             – refined spot binary mask (uint8)
    region_mask           – refined region binary mask (uint8)
    """
    leaf_u8   = (leaf_mask > 0).astype(np.uint8)
    spot_u8   = cv2.bitwise_and((spot_mask > 0).astype(np.uint8),   leaf_u8)
    region_u8 = cv2.bitwise_and((region_mask > 0).astype(np.uint8), leaf_u8)

    disease_union = cv2.bitwise_or(spot_u8, region_u8)

    leaf_area     = int(np.count_nonzero(leaf_u8))
    spot_area     = int(np.count_nonzero(spot_u8))
    region_area   = int(np.count_nonzero(region_u8))
    total_disease = int(np.count_nonzero(disease_union))

    severity_pct = round(total_disease / leaf_area * 100, 2) if leaf_area > 0 else 0.0
    spot_pct     = round(spot_area    / leaf_area * 100, 2) if leaf_area > 0 else 0.0
    region_pct   = round(region_area  / leaf_area * 100, 2) if leaf_area > 0 else 0.0

    # ── Discrete lesion counts + centroids ───────────────────────────────
    n_spot_labels,   spot_label_map,   spot_stats,   spot_ctrds   = \
        cv2.connectedComponentsWithStats(spot_u8,   connectivity=8)
    n_region_labels, region_label_map, region_stats, region_ctrds = \
        cv2.connectedComponentsWithStats(region_u8, connectivity=8)

    spot_count   = max(n_spot_labels   - 1, 0)
    region_count = max(n_region_labels - 1, 0)
    total_lesions = spot_count + region_count

    # ── Centroids for distribution analysis ──────────────────────────────
    all_centroids = []
    for i in range(1, n_spot_labels):
        all_centroids.append((spot_ctrds[i][0], spot_ctrds[i][1]))
    for i in range(1, n_region_labels):
        all_centroids.append((region_ctrds[i][0], region_ctrds[i][1]))
    all_centroids = np.array(all_centroids, dtype=np.float32) if all_centroids else np.empty((0, 2))

    # ── Mean lesion size ──────────────────────────────────────────────────
    all_areas = []
    for i in range(1, n_spot_labels):
        all_areas.append(int(spot_stats[i, cv2.CC_STAT_AREA]))
    for i in range(1, n_region_labels):
        all_areas.append(int(region_stats[i, cv2.CC_STAT_AREA]))
    mean_lesion_size = round(float(np.mean(all_areas)), 1) if all_areas else 0.0

    # ── Lesion density ────────────────────────────────────────────────────
    # Normalised to per-10 000 leaf pixels for scale invariance
    lesion_density = round(total_lesions / leaf_area * 10_000, 3) if leaf_area > 0 else 0.0

    # ── Distribution ─────────────────────────────────────────────────────
    distribution = _distribution_label(all_centroids, leaf_u8)

    # ── Horsfall-Barratt grade ────────────────────────────────────────────
    hb_grade = _horsfall_barratt_grade(severity_pct)

    # ── Normalised severity score (0-100, log-weighted) ───────────────────
    # Linear severity_pct collapses early stages; log weighting spreads them.
    # Formula: score = 100 × log(1 + severity_pct) / log(101)
    # Result: 1 % → 1.99, 5 % → 8.4, 25 % → 33, 50 % → 57, 100 % → 100
    severity_score = round(100 * np.log1p(severity_pct) / np.log(101), 1)

    return {
        "leaf_area_px":        leaf_area,
        "disease_area_px":     total_disease,
        "severity_pct":        severity_pct,
        "severity_label":      _severity_label_from_pct(severity_pct),
        "hb_grade":            hb_grade,
        "severity_score":      severity_score,
        "spot_count":          spot_count,
        "region_count":        region_count,
        "spot_severity_pct":   spot_pct,
        "region_severity_pct": region_pct,
        "mean_lesion_size_px": mean_lesion_size,
        "lesion_density":      lesion_density,
        "distribution":        distribution,
        "spot_mask":           spot_u8,
        "region_mask":         region_u8,
    }


# ── 7. Overlay Encoder ────────────────────────────────────────────────────────
def encode_overlay(
    img_rgb: np.ndarray,
    spot_mask: np.ndarray,
    region_mask: np.ndarray,
    leaf_mask: np.ndarray,
) -> str:
    """
    Dual-colour disease overlay with legend.
      • Orange (255, 165, 0)  → spots
      • Red    (220, 30, 30)  → regions
      • Green contour          → leaf boundary
    """
    overlay = img_rgb.copy()
    alpha   = 0.55

    spot_px = spot_mask > 0
    if np.any(spot_px):
        overlay[spot_px] = (
            overlay[spot_px] * (1 - alpha) + np.array([255, 165, 0]) * alpha
        ).astype(np.uint8)
        conts, _ = cv2.findContours(spot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, conts, -1, (255, 140, 0), 1)

    region_px = region_mask > 0
    if np.any(region_px):
        overlay[region_px] = (
            overlay[region_px] * (1 - alpha) + np.array([220, 30, 30]) * alpha
        ).astype(np.uint8)
        conts, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, conts, -1, (180, 0, 0), 2)

    leaf_conts, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, leaf_conts, -1, (0, 200, 0), 1)

    h, w = overlay.shape[:2]
    legend_h, legend_w = 50, 180
    lx, ly = 10, h - legend_h - 10
    roi = overlay[ly:ly + legend_h, lx:lx + legend_w].copy()
    cv2.rectangle(overlay, (lx, ly), (lx + legend_w, ly + legend_h), (0, 0, 0), -1)
    overlay[ly:ly + legend_h, lx:lx + legend_w] = (
        roi * 0.3 + overlay[ly:ly + legend_h, lx:lx + legend_w] * 0.7
    ).astype(np.uint8)
    cv2.rectangle(overlay, (lx + 8, ly + 8),  (lx + 22, ly + 22), (255, 165, 0), -1)
    cv2.putText(overlay, "Spots",   (lx + 28, ly + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(overlay, (lx + 8, ly + 28), (lx + 22, ly + 42), (220, 30, 30), -1)
    cv2.putText(overlay, "Regions", (lx + 28, ly + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    pil_img  = Image.fromarray(overlay)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG", quality=92)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# ── 8. Prediction Endpoint ─────────────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr    = np.frombuffer(contents, np.uint8)
        img_bgr  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Step 1 — Classification
        prediction, confidence = classify_image_with_tta(img_rgb, n_augments=6)

        # Defaults for healthy leaf
        severity_pct        = 0.0
        severity_label      = "N/A"
        hb_grade            = 0
        severity_score      = 0.0
        overlay_b64         = None
        spot_count          = 0
        region_count        = 0
        spot_severity_pct   = 0.0
        region_severity_pct = 0.0
        mean_lesion_size_px = 0.0
        lesion_density      = 0.0
        distribution        = "None"

        if prediction != "Healthy":
            # Step 2 — SAM2 leaf segmentation
            leaf_mask = segment_full_leaf(img_rgb)

            if np.count_nonzero(leaf_mask) > 0:
                # Step 3 — Disease-aware colour threshold
                rough_disease_mask = create_color_mask_within_leaf(
                    img_rgb, leaf_mask, disease_class=prediction
                )

                # Step 4 — 3-pass mask-guided SAM2 refinement
                spot_mask, region_mask = refine_disease_mask_with_sam2(
                    img_rgb, rough_disease_mask, leaf_mask,
                    disease_class=prediction,
                )

                # Step 5 — Rich severity metrics
                metrics = calculate_severity(leaf_mask, spot_mask, region_mask)
                severity_pct        = metrics["severity_pct"]
                severity_label      = metrics["severity_label"]
                hb_grade            = metrics["hb_grade"]
                severity_score      = metrics["severity_score"]
                spot_count          = metrics["spot_count"]
                region_count        = metrics["region_count"]
                spot_severity_pct   = metrics["spot_severity_pct"]
                region_severity_pct = metrics["region_severity_pct"]
                mean_lesion_size_px = metrics["mean_lesion_size_px"]
                lesion_density      = metrics["lesion_density"]
                distribution        = metrics["distribution"]

                # Step 6 — Overlay
                overlay_b64 = encode_overlay(
                    img_rgb,
                    metrics["spot_mask"],
                    metrics["region_mask"],
                    leaf_mask,
                )

        return {
            "prediction":           prediction,
            "confidence":           confidence,
            "severity_percentage":  severity_pct,
            "severity_label":       severity_label,
            "hb_grade":             hb_grade,
            "severity_score":       severity_score,
            "sam_mask_image":       overlay_b64,
            "spot_count":           spot_count,
            "region_count":         region_count,
            "spot_severity_pct":    spot_severity_pct,
            "region_severity_pct":  region_severity_pct,
            "mean_lesion_size_px":  mean_lesion_size_px,
            "lesion_density":       lesion_density,
            "distribution":         distribution,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 9. Entry Point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)