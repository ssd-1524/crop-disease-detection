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
from PIL import Image
import uvicorn

# ── 1. App Setup ───────────────────────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_origin_regex=r"https://.*\.vercel\.app",
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
SAM2_CONFIG         = "configs/sam2.1/sam2.1_hiera_l.yaml"

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
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def classify_image_with_tta(img_rgb: np.ndarray, n_augments: int = 6) -> tuple[str, str]:
    tta_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    base_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_pil   = Image.fromarray(img_rgb)
    all_probs = []
    with torch.inference_mode():
        all_probs.append(torch.softmax(classifier(base_tf(img_pil).unsqueeze(0).to(device)), dim=1))
    for _ in range(n_augments):
        with torch.inference_mode():
            all_probs.append(torch.softmax(classifier(tta_tf(img_pil).unsqueeze(0).to(device)), dim=1))
    avg   = torch.stack(all_probs).mean(0)[0]
    idx   = avg.argmax().item()
    return CLASS_NAMES[idx], f"{avg[idx].item() * 100:.2f}%"


# ── 4. Disease colour profiles ─────────────────────────────────────────────────
#
# Fixed HSV/LAB ranges catch clear-cut cases.
# Adaptive local-contrast detectors (see Section 5) handle the rest:
#   Common Rust  → _detect_rust_local_contrast()
#   Gray Leaf Spot → _detect_gls_local_contrast()
#
# morph_open_k = 0  → opening disabled (would erase tiny rust pustules)

DISEASE_COLOR_PROFILES = {
    "Blight": {
        "hsv_ranges": [
            (np.array([10, 50, 80]),  np.array([25, 200, 220])),
            (np.array([5,  40, 40]),  np.array([15, 180, 160])),
        ],
        "lab_ranges": [
            (np.array([60, 133, 145]), np.array([170, 155, 180])),
        ],
        "exclude_green_s_min":   55,
        "morph_close_k":         5,
        "morph_open_k":          3,
        "use_green_departure":   True,
        "departure_sensitivity": 0.30,
    },
    "Common_Rust": {
        # Fixed ranges kept as a fast first-pass; local-contrast detector
        # runs in addition and catches everything lighting shifts hide.
        "hsv_ranges": [
            (np.array([5,  25, 30]),   np.array([25, 255, 255])),  # full orange-brown sweep
            (np.array([0,  20, 25]),   np.array([8,  200, 180])),  # dark reddish
            (np.array([170, 20, 25]),  np.array([180, 200, 180])), # red wrap-around
        ],
        "lab_ranges": [
            (np.array([30, 135, 140]), np.array([220, 185, 210])),  # orange (wide)
            (np.array([15, 128, 130]), np.array([130, 162, 172])),  # dark brown
        ],
        "exclude_green_s_min":  40,
        "morph_close_k":        3,
        "morph_open_k":         0,   # MUST stay 0 — opening kills tiny pustules
        "use_green_departure":  False,
        "use_rust_local":       True,
    },
    "Gray_Leaf_Spot": {
        "hsv_ranges": [
            (np.array([10, 15, 90]),   np.array([30, 90, 220])),   # tan/straw
            (np.array([5,  25, 45]),   np.array([20, 150, 165])),  # brown necrotic
            (np.array([0,  0,  140]),  np.array([30, 22,  255])),  # pale centre
        ],
        "lab_ranges": [
            (np.array([80, 126, 138]), np.array([165, 148, 168])),
        ],
        "exclude_green_s_min":  30,
        "morph_close_k":        5,
        "morph_open_k":         3,
        "use_green_departure":  False,
        "use_gls_local":        True,
    },
}


# ── 5. Adaptive local-contrast detectors ──────────────────────────────────────
#
# Why local contrast instead of (or in addition to) fixed thresholds?
# ────────────────────────────────────────────────────────────────────
# Fixed HSV/LAB ranges encode the *absolute* colour of a lesion, which drifts
# with camera white-balance, exposure, and disease stage.  Local-contrast
# detectors instead ask "is this pixel anomalous relative to its immediate
# neighbourhood?" — a question that is largely invariant to global lighting.
#
#   Rust  → each pustule is a locally WARMER (orange/red) spot vs green tissue
#            → detect via positive deviation of LAB a* and b* from local mean
#   GLS   → each lesion is a locally LESS SATURATED (grayer) zone vs green tissue
#            → detect via negative deviation of HSV S from local mean


def _detect_rust_local_contrast(
    img_rgb: np.ndarray,
    leaf_mask: np.ndarray,
) -> np.ndarray:
    """
    Dual-scale local LAB contrast detector for Common Rust pustules.

    Single-scale detectors fail on very small pustules (2–8 px) because the
    neighbourhood average smooths out their signal before thresholding.
    Running two scales and OR-ing the results catches both micro-pustules
    (tight 7 px kernel) and larger mature pustules (wider 15 px kernel).

    Warmth = (a* − local_a*) + 0.6 × (b* − local_b*)
      a* axis: red vs green  (primary rust signal)
      b* axis: yellow vs blue (secondary confirmation)

    For each scale we threshold the RAW warmth map directly — no secondary
    neighbourhood average — because the Gaussian blur already provides spatial
    smoothing.  The neighbourhood check was the main reason tiny pustules were
    getting suppressed.

    Guards applied after OR:
      • Not green-hued (H 28–90)
      • Brightness V > 25  (exclude pure shadow/background)
      • AND with leaf_mask
      • 3×3 closing to fill sub-pixel gaps
      • Safety cap > 55 % leaf coverage → discard
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    lab     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_ch    = hsv[:, :, 0]
    v_ch    = hsv[:, :, 2]

    a_ch = lab[:, :, 1]
    b_ch = lab[:, :, 2]

    combined = np.zeros(img_rgb.shape[:2], dtype=np.uint8)

    for ksize, threshold in [(7, 5.0), (15, 6.5), (25, 8.0)]:
        ksize   = ksize | 1
        a_local = cv2.GaussianBlur(a_ch, (ksize, ksize), ksize / 3.0)
        b_local = cv2.GaussianBlur(b_ch, (ksize, ksize), ksize / 3.0)
        warmth  = (a_ch - a_local) + 0.6 * (b_ch - b_local)
        hot     = (warmth > threshold).astype(np.uint8) * 255
        combined = cv2.bitwise_or(combined, hot)

    # ── Guards ────────────────────────────────────────────────────────────
    not_green = cv2.bitwise_not(cv2.inRange(h_ch, np.array([28]), np.array([90])))
    bright    = (v_ch > 25).astype(np.uint8) * 255
    result    = cv2.bitwise_and(combined,  not_green)
    result    = cv2.bitwise_and(result,    bright)
    result    = cv2.bitwise_and(result,    leaf_mask)

    kern   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kern)

    leaf_area = int(np.count_nonzero(leaf_mask))
    if leaf_area > 0 and int(np.count_nonzero(result)) / leaf_area > 0.55:
        return np.zeros(img_rgb.shape[:2], dtype=np.uint8)

    return result


def _detect_gls_local_contrast(
    img_rgb: np.ndarray,
    leaf_mask: np.ndarray,
    local_kernel: int = 25,
    deficit_threshold: float = 28.0,
    neib_threshold: float = 12.0,
) -> np.ndarray:
    """
    Detect Gray Leaf Spot lesions via local HSV saturation contrast.

    GLS lesions are rectangular, tan/gray areas that are markedly LESS
    saturated than the surrounding green tissue.  They do not have a
    distinctive absolute colour (they're close to white/beige), but their
    saturation DEFICIT relative to healthy tissue is very consistent.

    Algorithm
    ─────────
    1. Compute local mean saturation via Gaussian blur (kernel 25 px — GLS
       lesions are larger than rust pustules so we use a wider window).

    2. Saturation deficit = local_mean_S − pixel_S.
       Positive values = pixel is less saturated than its surroundings.

    3. Neighbourhood average of the deficit map (kernel 13 px) for noise
       suppression.

    4. Flag pixels where both raw deficit AND neighbourhood average exceed
       thresholds.

    5. Hue guard: pixel must not be green-hued (H 28–88) and must be in the
       tan/straw/neutral range (H 0–35 OR very low S < 18).
       Brightness guard: V > 70 (not shadow).
       Safety cap: > 65 % leaf coverage → discard (over-detection).

    Returns binary uint8 mask (H, W).
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_ch    = hsv[:, :, 0].astype(np.uint8)
    s_ch    = hsv[:, :, 1]
    v_ch    = hsv[:, :, 2].astype(np.uint8)

    # ── Local mean saturation ─────────────────────────────────────────────
    ksize   = local_kernel | 1
    s_local = cv2.GaussianBlur(s_ch, (ksize, ksize), ksize / 3.0)

    # ── Saturation deficit ────────────────────────────────────────────────
    deficit = s_local - s_ch   # positive = less saturated than neighbourhood

    # ── Neighbourhood average (noise suppression) ─────────────────────────
    neib_k     = 13
    deficit_u8 = np.clip(deficit * 2 + 128, 0, 255).astype(np.uint8)
    neib_u8    = cv2.GaussianBlur(deficit_u8, (neib_k, neib_k), neib_k / 3.0)
    neib_score = (neib_u8.astype(np.float32) - 128) / 2.0

    # ── Threshold ─────────────────────────────────────────────────────────
    desaturated = ((deficit > deficit_threshold) & (neib_score > neib_threshold)
                   ).astype(np.uint8) * 255

    # ── Hue guard: tan/straw range only ───────────────────────────────────
    # Allow H 0–35 (tan/warm neutral), or near-zero saturation (any hue)
    tan_hue   = cv2.inRange(h_ch, np.array([0]),  np.array([35]))
    very_pale = (hsv[:, :, 1] < 18).astype(np.uint8) * 255
    hue_ok    = cv2.bitwise_or(tan_hue, very_pale)

    not_green = cv2.bitwise_not(cv2.inRange(h_ch, np.array([28]), np.array([88])))
    bright    = (v_ch > 70).astype(np.uint8) * 255

    result = cv2.bitwise_and(desaturated, hue_ok)
    result = cv2.bitwise_and(result,      not_green)
    result = cv2.bitwise_and(result,      bright)
    result = cv2.bitwise_and(result,      leaf_mask)

    # Morphological cleanup — GLS lesions are elongated; close fills gaps
    kern_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kern_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kern_c)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN,  kern_o)

    leaf_area = int(np.count_nonzero(leaf_mask))
    if leaf_area > 0 and int(np.count_nonzero(result)) / leaf_area > 0.65:
        return np.zeros(img_rgb.shape[:2], dtype=np.uint8)

    return result


def create_color_mask_within_leaf(
    img_rgb: np.ndarray,
    leaf_mask: np.ndarray,
    disease_class: str = "Common_Rust",
) -> np.ndarray:
    """
    Two-stage colour mask:
      Stage 1 — Fixed HSV + LAB ranges  (fast, handles clear-cut cases)
      Stage 2 — Adaptive local-contrast  (handles lighting variation + edge cases)
    Results are OR-ed, then green pixels removed, then morphological cleanup.
    """
    img_bgr  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    profile  = DISEASE_COLOR_PROFILES.get(disease_class, DISEASE_COLOR_PROFILES["Common_Rust"])
    combined = np.zeros(img_rgb.shape[:2], dtype=np.uint8)

    # ── Stage 1: fixed HSV + LAB ranges ──────────────────────────────────
    for lo, hi in profile["hsv_ranges"]:
        combined |= cv2.inRange(hsv, lo, hi)
    for lo, hi in profile["lab_ranges"]:
        combined |= cv2.inRange(lab, lo, hi)

    # ── Blight: adaptive desaturation detector ────────────────────────────
    if profile.get("use_green_departure", False):
        green_zone = cv2.inRange(hsv, np.array([35, 45, 40]), np.array([85, 255, 255]))
        green_zone = cv2.bitwise_and(green_zone, leaf_mask)
        green_sats = s_ch[green_zone > 0]
        if len(green_sats) > 100:
            s_thresh  = float(np.median(green_sats)) * profile.get("departure_sensitivity", 0.30)
            low_sat   = (s_ch.astype(np.float32) < s_thresh).astype(np.uint8) * 255
            not_green = cv2.bitwise_not(cv2.inRange(h_ch, np.array([35]), np.array([85])))
            bright    = (v_ch > 60).astype(np.uint8) * 255
            dep       = cv2.bitwise_and(cv2.bitwise_and(low_sat, bright), not_green)
            dep       = cv2.bitwise_and(dep, leaf_mask)
            kern      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dep       = cv2.morphologyEx(dep, cv2.MORPH_CLOSE, kern)
            dep       = cv2.morphologyEx(dep, cv2.MORPH_OPEN,  kern)
            leaf_area = np.count_nonzero(leaf_mask)
            if leaf_area > 0 and np.count_nonzero(dep) / leaf_area < 0.50:
                combined = cv2.bitwise_or(combined, dep)

    # ── Stage 2: adaptive local-contrast detectors ────────────────────────
    if profile.get("use_rust_local", False):
        combined |= _detect_rust_local_contrast(img_rgb, leaf_mask)

    if profile.get("use_gls_local", False):
        combined |= _detect_gls_local_contrast(img_rgb, leaf_mask)

    # ── Remove green pixels ───────────────────────────────────────────────
    s_min      = profile.get("exclude_green_s_min", 40)
    green_mask = cv2.inRange(hsv, np.array([28, s_min, 30]), np.array([90, 255, 255]))
    combined   = cv2.bitwise_and(combined, cv2.bitwise_not(green_mask))
    combined   = cv2.bitwise_and(combined, leaf_mask)

    # ── Morphological cleanup ─────────────────────────────────────────────
    close_k  = profile.get("morph_close_k", 5)
    open_k   = profile.get("morph_open_k",  3)
    kern_c   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kern_c)
    if open_k > 0:
        kern_o   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kern_o)

    return combined


def segment_full_leaf(img_rgb: np.ndarray) -> np.ndarray:
    h, w   = img_rgb.shape[:2]
    cx, cy = w // 2, h // 2
    pos = np.array([
        [cx, cy], [cx - w//6, cy], [cx + w//6, cy],
        [cx, cy - h//6], [cx, cy + h//6],
    ], dtype=np.float32)
    neg = np.array([[10,10],[w-10,10],[10,h-10],[w-10,h-10]], dtype=np.float32)
    pts = np.concatenate([pos, neg])
    lbs = np.array([1]*len(pos) + [0]*len(neg), dtype=np.int32)
    sam_predictor.set_image(img_rgb)
    with torch.inference_mode():
        masks, scores, _ = sam_predictor.predict(
            point_coords=pts, point_labels=lbs, multimask_output=True)
    return masks[np.argmax(scores)].astype(np.uint8)


# ── 6. SAM2 refinement helpers ─────────────────────────────────────────────────

def _mask_to_sam2_logit(binary_mask: np.ndarray) -> np.ndarray:
    """
    Convert a binary (H, W) uint8 mask → SAM2 mask_input tensor (1, 256, 256).

    Logit values:
      +10.0  (disease pixels)  → sigmoid ≈ 1.00   "definitely foreground"
       -5.0  (background)      → sigmoid ≈ 0.007  "likely background"

    Asymmetric values (+10 / -5 instead of ±10) intentionally let SAM2 grow
    past the colour mask where texture evidence warrants it, while still
    anchoring firmly to the colour mask interior.
    """
    resized = cv2.resize(binary_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    logit   = np.where(resized > 0, 10.0, -5.0).astype(np.float32)
    return logit[np.newaxis, :, :]   # (1, 256, 256)


def _sample_points_in_mask(binary: np.ndarray, n: int) -> np.ndarray:
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
    labels: np.ndarray, stats: np.ndarray, num_labels: int,
    gap_px: int = 40, min_area: int = 8,
) -> list[list[int]]:
    """Merge nearby connected components into spatial clusters."""
    valid = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not valid:
        return []

    def bbox(i):
        x, y, bw, bh = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                         stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        return x, y, x + bw, y + bh

    clusters = [[valid[0]]]
    for idx in valid[1:]:
        x0i, y0i, x1i, y1i = bbox(idx)
        merged = False
        for cluster in clusters:
            for j in cluster:
                x0j, y0j, x1j, y1j = bbox(j)
                if (max(0, max(x0i,x0j) - min(x1i,x1j)) <= gap_px and
                        max(0, max(y0i,y0j) - min(y1i,y1j)) <= gap_px):
                    cluster.append(idx)
                    merged = True
                    break
            if merged:
                break
        if not merged:
            clusters.append([idx])
    return clusters


def _cluster_bbox(cluster, stats, img_w, img_h, pad=24):
    x0 = min(stats[i, cv2.CC_STAT_LEFT] for i in cluster)
    y0 = min(stats[i, cv2.CC_STAT_TOP]  for i in cluster)
    x1 = max(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]  for i in cluster)
    y1 = max(stats[i, cv2.CC_STAT_TOP]  + stats[i, cv2.CC_STAT_HEIGHT] for i in cluster)
    return np.array([
        max(x0-pad,0), max(y0-pad,0),
        min(x1+pad, img_w-1), min(y1+pad, img_h-1),
    ], dtype=np.float32)


# Per-disease thresholds for spot vs region classification
# Rust pustules are inherently small — a 50 px pustule is a legitimate "spot",
# not a region, so we use a lower threshold to avoid sending tiny areas through
# SAM2 where it tends to over-expand.
_MIN_AREAS = {
    "Common_Rust":    3,   # pustules can be 3-5 px on high-res images
    "Blight":         8,
    "Gray_Leaf_Spot": 6,
}
    "Common_Rust":   60,   # pustules are small; keep more as verbatim spots
    "Blight":       200,   # blight lesions are large
    "Gray_Leaf_Spot": 120,
}
_CLUSTER_GAPS = {
    "Common_Rust":   25,   # dense pustule fields — cluster tightly
    "Blight":        50,
    "Gray_Leaf_Spot": 40,
}


def refine_disease_mask_with_sam2(
    img_rgb: np.ndarray,
    rough_mask: np.ndarray,
    leaf_mask: np.ndarray,
    disease_class: str = "Common_Rust",
) -> tuple[np.ndarray, np.ndarray]:
    """
    3-pass mask-guided SAM2 refinement with disease-specific thresholds.

    Pass 1 — box + distance-transform points (baseline, no colour bias).
    Pass 2 — box + points + rough colour mask as mask_input (logit form).
             The colour mask acts as a dense spatial prior.  SAM2 can grow
             past it where texture warrants, and trim false-positives where it
             doesn't.
    Pass 3 — box + points + pass-2 logits (iterative edge sharpening).

    Selection (best-first):
      pass-3 → pass-2 → pass-1 → verbatim colour pixels
    Acceptance criteria: IoU with rough mask ≥ 0.25 AND expansion ≤ 2.0×.

    Components smaller than spot_threshold are kept verbatim (SAM2
    over-expands on sub-pixel-cluster pustules).
    """
    h, w = img_rgb.shape[:2]
    min_area       = _MIN_AREAS.get(disease_class, 6)
    spot_threshold = _SPOT_THRESHOLDS.get(disease_class, 120)
    cluster_gap    = _CLUSTER_GAPS.get(disease_class, 40)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        rough_mask, connectivity=8)

    spot_mask   = np.zeros((h, w), dtype=np.uint8)
    region_mask = np.zeros((h, w), dtype=np.uint8)

    if num_labels <= 1:
        return spot_mask, region_mask

    spot_ids, region_ids = [], []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        (spot_ids if area < spot_threshold else region_ids).append(i)

    # Small spots → keep verbatim
    for i in spot_ids:
        spot_mask |= (labels == i).astype(np.uint8)

    if not region_ids:
        return cv2.bitwise_and(spot_mask, leaf_mask), region_mask

    # Inset leaf mask to prevent SAM2 bleeding onto the leaf boundary
    leaf_inset = cv2.erode(
        leaf_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    if np.count_nonzero(leaf_inset) < np.count_nonzero(leaf_mask) * 0.5:
        leaf_inset = leaf_mask

    # Cluster nearby regions together
    clusters      = _cluster_components(labels, stats, num_labels,
                                        gap_px=cluster_gap, min_area=spot_threshold)
    clustered_ids = {idx for c in clusters for idx in c}
    for rid in region_ids:
        if rid not in clustered_ids:
            clusters.append([rid])

    sam_predictor.set_image(img_rgb)

    def _score_mask(m: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
        m    = cv2.bitwise_and(m, leaf_inset)
        area = int(np.count_nonzero(m))
        ref_area = int(np.count_nonzero(ref))
        inter = int(np.count_nonzero(m & ref))
        union = int(np.count_nonzero(m | ref))
        iou   = inter / union if union > 0 else 0.0
        ratio = area / ref_area if ref_area > 0 else 999.0
        return iou, ratio

    def _ok(iou, ratio):
        return iou >= 0.25 and ratio <= 2.0

    for cluster in clusters:
        cluster_bin  = np.zeros((h, w), dtype=np.uint8)
        for i in cluster:
            cluster_bin |= (labels == i).astype(np.uint8)
        if not np.count_nonzero(cluster_bin):
            continue

        box        = _cluster_bbox(cluster, stats, w, h)
        pos_points = _sample_points_in_mask(cluster_bin, n=6)
        if len(pos_points) == 0:
            M = cv2.moments(cluster_bin)
            if M["m00"] > 0:
                pos_points = np.array([[int(M["m10"]/M["m00"]),
                                        int(M["m01"]/M["m00"])]], dtype=np.float32)
            else:
                region_mask |= cluster_bin
                continue
        pos_labels  = np.ones(len(pos_points), dtype=np.int32)
        rough_logit = _mask_to_sam2_logit(cluster_bin)

        # Pass 1 — baseline
        try:
            with torch.inference_mode():
                m1, s1, l1 = sam_predictor.predict(
                    point_coords=pos_points, point_labels=pos_labels,
                    box=box, multimask_output=True)
            p1      = m1[np.argmax(s1)].astype(np.uint8)
            p1_logit = l1[np.argmax(s1)]
        except Exception:
            region_mask |= cluster_bin
            continue

        # Pass 2 — colour mask as spatial prior
        try:
            with torch.inference_mode():
                m2, _, l2 = sam_predictor.predict(
                    point_coords=pos_points, point_labels=pos_labels,
                    box=box, mask_input=rough_logit, multimask_output=False)
            p2      = m2[0].astype(np.uint8)
            p2_logit = l2[0]
        except Exception:
            p2 = p1;  p2_logit = p1_logit

        # Pass 3 — SAM2's own logits (iterative sharpening)
        try:
            with torch.inference_mode():
                m3, _, _ = sam_predictor.predict(
                    point_coords=pos_points, point_labels=pos_labels,
                    box=box, mask_input=p2_logit, multimask_output=False)
            p3 = m3[0].astype(np.uint8)
        except Exception:
            p3 = p2

        p3_iou, p3_r = _score_mask(p3, cluster_bin)
        p2_iou, p2_r = _score_mask(p2, cluster_bin)
        p1_iou, p1_r = _score_mask(p1, cluster_bin)

        if   _ok(p3_iou, p3_r): chosen = cv2.bitwise_and(p3, leaf_inset)
        elif _ok(p2_iou, p2_r): chosen = cv2.bitwise_and(p2, leaf_inset)
        elif _ok(p1_iou, p1_r): chosen = cv2.bitwise_and(p1, leaf_inset)
        else:                   chosen = cv2.bitwise_and(cluster_bin, leaf_inset)

        region_mask |= chosen

    return (cv2.bitwise_and(spot_mask,   leaf_mask),
            cv2.bitwise_and(region_mask, leaf_mask))


# ── 7. Severity Calculation ────────────────────────────────────────────────────

_HB_ANCHORS = np.array([0, 3, 6, 12, 25, 50, 75, 87, 94, 97, 100], dtype=np.float32)

def _hb_grade(pct: float) -> int:
    return int(np.argmin(np.abs(_HB_ANCHORS - pct)))


def _distribution_label(centroids: np.ndarray, leaf_mask: np.ndarray) -> str:
    if len(centroids) == 0:
        return "None"
    ys, xs = np.where(leaf_mask > 0)
    if not len(xs):
        return "Unknown"
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    bw, bh = max(x_max - x_min, 1), max(y_max - y_min, 1)
    zones = set()
    for cx, cy in centroids:
        zones.add((min(int((cy-y_min)/bh*3), 2), min(int((cx-x_min)/bw*3), 2)))
    n = len(zones)
    return "Dispersed" if n >= 7 else ("Scattered" if n >= 4 else "Localised")


def calculate_severity(
    leaf_mask: np.ndarray,
    spot_mask: np.ndarray,
    region_mask: np.ndarray,
    disease_class: str = "Unknown",
) -> dict:
    """
    Compute a rich, multi-dimensional severity score.

    The core insight is that raw area percentage alone is a poor proxy for
    agronomic impact.  Two images can have 5 % disease area but represent
    vastly different situations:
      • 200 tiny rust pustules (early, localised) → low urgency
      • One large blight lesion spreading across the midrib → high urgency

    We therefore compute a weighted severity score from four components:

      area_score        (weight 0.45)
        Log-scaled disease area: 100 × log(1+pct) / log(101)
        Log scaling prevents early-stage detections from collapsing to ~0.

      density_score     (weight 0.20)
        Lesion density = lesions per 10 000 leaf pixels, normalised to 0-100.
        Normalisation anchor: 8 lesions/10 000 px ≈ "very dense" → score 100.
        Captures severity from many small lesions that area alone under-weights.

      distribution_score (weight 0.20)
        How spread the lesions are across the leaf surface:
          Localised  → 20   (early / isolated)
          Scattered  → 55   (progressing)
          Dispersed  → 100  (advanced / systemic)

      progression_score  (weight 0.15)
        Region area / total disease area.
        Regions (large, coalescing lesions) indicate more advanced infection
        than discrete spots of the same total area.
        Ranges 0–100 (0 = all spots, 100 = all regions).

    Final weighted_score = Σ(weight_i × component_i)  ∈ [0, 100]

    Labels derived from weighted_score:
      0–8   → Trace
      8–22  → Mild
      22–42 → Moderate
      42–65 → Severe
      65+   → Very Severe
    """
    leaf_u8   = (leaf_mask > 0).astype(np.uint8)
    spot_u8   = cv2.bitwise_and((spot_mask > 0).astype(np.uint8),   leaf_u8)
    region_u8 = cv2.bitwise_and((region_mask > 0).astype(np.uint8), leaf_u8)
    union_u8  = cv2.bitwise_or(spot_u8, region_u8)

    leaf_area     = int(np.count_nonzero(leaf_u8))
    spot_area     = int(np.count_nonzero(spot_u8))
    region_area   = int(np.count_nonzero(region_u8))
    total_disease = int(np.count_nonzero(union_u8))

    severity_pct = round(total_disease / leaf_area * 100, 2) if leaf_area > 0 else 0.0
    spot_pct     = round(spot_area    / leaf_area * 100, 2) if leaf_area > 0 else 0.0
    region_pct   = round(region_area  / leaf_area * 100, 2) if leaf_area > 0 else 0.0

    # ── Lesion counts + centroids ─────────────────────────────────────────
    n_s, s_lbl, s_stats, s_ct = cv2.connectedComponentsWithStats(spot_u8,   8)
    n_r, r_lbl, r_stats, r_ct = cv2.connectedComponentsWithStats(region_u8, 8)

    spot_count   = max(n_s - 1, 0)
    region_count = max(n_r - 1, 0)
    total_lesions = spot_count + region_count

    centroids = []
    for i in range(1, n_s): centroids.append((s_ct[i][0], s_ct[i][1]))
    for i in range(1, n_r): centroids.append((r_ct[i][0], r_ct[i][1]))
    centroids = np.array(centroids, dtype=np.float32) if centroids else np.empty((0, 2))

    # ── Mean lesion size ──────────────────────────────────────────────────
    areas = ([int(s_stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_s)] +
             [int(r_stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_r)])
    mean_lesion_size = round(float(np.mean(areas)), 1) if areas else 0.0

    # ── Distribution ─────────────────────────────────────────────────────
    distribution = _distribution_label(centroids, leaf_u8)

    # ── HB grade ─────────────────────────────────────────────────────────
    hb = _hb_grade(severity_pct)

    # ── Four severity components ──────────────────────────────────────────
    # Component 1: log-scaled area (0-100)
    area_score = 100.0 * np.log1p(severity_pct) / np.log(101)

    # Component 2: lesion density, normalised to 0-100
    # Anchor: 8 lesions / 10 000 leaf px = score 100
    lesion_density   = round(total_lesions / leaf_area * 10_000, 3) if leaf_area > 0 else 0.0
    density_score    = min(lesion_density / 8.0 * 100.0, 100.0)

    # Component 3: distribution (0-100)
    dist_scores = {"None": 0.0, "Unknown": 0.0, "Localised": 20.0,
                   "Scattered": 55.0, "Dispersed": 100.0}
    distribution_score = dist_scores.get(distribution, 0.0)

    # Component 4: progression — how much of the disease is coalescing regions
    progression_score = (region_area / total_disease * 100.0) if total_disease > 0 else 0.0

    # ── Weighted combination ──────────────────────────────────────────────
    weighted_score = round(
        0.45 * area_score +
        0.20 * density_score +
        0.20 * distribution_score +
        0.15 * progression_score,
        1,
    )

    # ── Label from weighted score ─────────────────────────────────────────
    if   weighted_score < 8:   label = "Trace"
    elif weighted_score < 22:  label = "Mild"
    elif weighted_score < 42:  label = "Moderate"
    elif weighted_score < 65:  label = "Severe"
    else:                      label = "Very Severe"

    return {
        "leaf_area_px":        leaf_area,
        "disease_area_px":     total_disease,
        "severity_pct":        severity_pct,
        "severity_label":      label,
        "hb_grade":            hb,
        "severity_score":      weighted_score,
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


# ── 8. Overlay Encoder ────────────────────────────────────────────────────────
def encode_overlay(
    img_rgb: np.ndarray,
    spot_mask: np.ndarray,
    region_mask: np.ndarray,
    leaf_mask: np.ndarray,
) -> str:
    overlay = img_rgb.copy()
    alpha   = 0.55

    if np.any(spot_mask > 0):
        overlay[spot_mask > 0] = (
            overlay[spot_mask > 0] * (1 - alpha) + np.array([255, 165, 0]) * alpha
        ).astype(np.uint8)
        conts, _ = cv2.findContours(spot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, conts, -1, (255, 140, 0), 1)

    if np.any(region_mask > 0):
        overlay[region_mask > 0] = (
            overlay[region_mask > 0] * (1 - alpha) + np.array([220, 30, 30]) * alpha
        ).astype(np.uint8)
        conts, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, conts, -1, (180, 0, 0), 2)

    leaf_c, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, leaf_c, -1, (0, 200, 0), 1)

    h, w = overlay.shape[:2]
    lx, ly, lh, lw = 10, h - 60, 50, 180
    roi = overlay[ly:ly+lh, lx:lx+lw].copy()
    cv2.rectangle(overlay, (lx, ly), (lx+lw, ly+lh), (0,0,0), -1)
    overlay[ly:ly+lh, lx:lx+lw] = (roi*0.3 + overlay[ly:ly+lh, lx:lx+lw]*0.7).astype(np.uint8)
    cv2.rectangle(overlay, (lx+8, ly+8),  (lx+22, ly+22), (255,165,0), -1)
    cv2.putText(overlay, "Spots",   (lx+28, ly+20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    cv2.rectangle(overlay, (lx+8, ly+28), (lx+22, ly+42), (220,30,30),  -1)
    cv2.putText(overlay, "Regions", (lx+28, ly+40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── 9. Prediction Endpoint ─────────────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr    = np.frombuffer(contents, np.uint8)
        img_bgr  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        prediction, confidence = classify_image_with_tta(img_rgb, n_augments=6)

        severity_pct = severity_label = hb_grade = severity_score = 0.0
        overlay_b64 = None
        spot_count = region_count = 0
        spot_severity_pct = region_severity_pct = 0.0
        mean_lesion_size_px = lesion_density = 0.0
        distribution = "None"
        severity_label = "N/A"
        hb_grade = 0

        if prediction != "Healthy":
            leaf_mask = segment_full_leaf(img_rgb)

            if np.count_nonzero(leaf_mask) > 0:
                rough_mask = create_color_mask_within_leaf(
                    img_rgb, leaf_mask, disease_class=prediction)

                spot_mask, region_mask = refine_disease_mask_with_sam2(
                    img_rgb, rough_mask, leaf_mask, disease_class=prediction)

                m = calculate_severity(leaf_mask, spot_mask, region_mask,
                                       disease_class=prediction)

                severity_pct        = m["severity_pct"]
                severity_label      = m["severity_label"]
                hb_grade            = m["hb_grade"]
                severity_score      = m["severity_score"]
                spot_count          = m["spot_count"]
                region_count        = m["region_count"]
                spot_severity_pct   = m["spot_severity_pct"]
                region_severity_pct = m["region_severity_pct"]
                mean_lesion_size_px = m["mean_lesion_size_px"]
                lesion_density      = m["lesion_density"]
                distribution        = m["distribution"]

                overlay_b64 = encode_overlay(
                    img_rgb, m["spot_mask"], m["region_mask"], leaf_mask)

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


# ── 10. Entry Point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))