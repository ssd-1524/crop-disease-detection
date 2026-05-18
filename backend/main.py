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

# â”€â”€ 1. App Setup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# â”€â”€ 2. Model Definition â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CLASS_NAMES = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]


class CustomMobileNetV2_3(nn.Module):
    def __init__(self, num_classes: int, dropout_rate: float = 0.2):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT
        self.model = mobilenet_v2(weights=weights)
        self.transforms = weights.transforms()
        self.model.features.requires_grad_(False)
        in_features = self.model.classifier[1].in_features
        self.head = nn.Linear(in_features, num_classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.model.classifier = nn.Sequential(nn.Dropout(p=dropout_rate), self.head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def freeze_base(self) -> None:
        self.model.features.requires_grad_(False)

    def unfreeze_base(self) -> None:
        self.model.features.requires_grad_(True)

    def unfreeze_last_n_layers(self, n: int = 10) -> None:
        layers = list(self.model.features.children())
        for layer in layers[-n:]:
            layer.requires_grad_(True)

    @property
    def param_summary(self) -> dict:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {
            "trainable": trainable,
            "frozen": total - trainable,
            "total": total,
            "trainable_pct": f"{100 * trainable / total:.1f}%",
        }


# â”€â”€ 3. Load Models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CLASSIFIER_FILENAME = "CustomMobileNetV2_2_best.pth"
SAM2_CHECKPOINT     = "sam2.1_hiera_large.pt"
SAM2_CONFIG         = "configs/sam2.1/sam2.1_hiera_l.yaml"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classifier = CustomMobileNetV2_3(num_classes=len(CLASS_NAMES))
classifier.load_state_dict(
    torch.load(CLASSIFIER_FILENAME, map_location=device, weights_only=False),
    strict=False,
)
classifier.to(device)
classifier.eval()
print(f"Classifier loaded successfully. {classifier.param_summary}")

sam2_model    = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
sam_predictor = SAM2ImagePredictor(sam2_model)
print("SAM2 loaded successfully.")


# â”€â”€ 4. Inference helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def classify_image_with_tta(img_rgb: np.ndarray, n_augments: int = 6) -> str:
    tta_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_pil   = Image.fromarray(img_rgb)
    all_probs = []

    with torch.inference_mode():
        base_tensor = classifier.transforms(img_pil).unsqueeze(0).to(device)
        all_probs.append(torch.softmax(classifier(base_tensor), dim=1))

    for _ in range(n_augments):
        with torch.inference_mode():
            all_probs.append(
                torch.softmax(classifier(tta_tf(img_pil).unsqueeze(0).to(device)), dim=1)
            )

    avg = torch.stack(all_probs).mean(0)[0]
    idx = avg.argmax().item()
    return CLASS_NAMES[idx]


# â”€â”€ 5. Otsu-based dynamic colour mask â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _otsu_1d(values: np.ndarray) -> float:
    """Otsu threshold on a flat uint8 array."""
    if len(values) < 10:
        return 128.0
    hist, _ = np.histogram(values, bins=256, range=(0, 256))
    total    = int(hist.sum())
    if total == 0:
        return 128.0
    sum_all  = float(np.dot(np.arange(256, dtype=np.float64), hist))
    sum_bg, w_bg, max_var, thresh = 0.0, 0, 0.0, 128.0
    for t in range(256):
        w_bg += hist[t]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += t * hist[t]
        mu_bg   = sum_bg / w_bg
        mu_fg   = (sum_all - sum_bg) / w_fg
        var     = w_bg * w_fg * (mu_bg - mu_fg) ** 2
        if var > max_var:
            max_var = var
            thresh  = float(t)
    return thresh


def create_color_mask_within_leaf(
    img_rgb: np.ndarray,
    leaf_mask: np.ndarray,
    disease_class: str = "Common_Rust",
) -> np.ndarray:
    """
    Otsu-driven rough disease mask.

    Instead of fixed HSV/LAB ranges the discriminating thresholds are computed
    from the histogram of leaf pixels in the uploaded image, so they adapt to
    each image's lighting and colour balance.

    Common_Rust  â€” LAB a* warmth + HSV orange-hue band
    Blight       â€” local lightness deficit + saturation drop + warm-hue band
    Gray_Leaf_Spot â€” local saturation deficit + pale/grey pixels
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv_f   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab_f   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv_u8  = hsv_f.astype(np.uint8)

    h_ch = hsv_f[:, :, 0]
    s_ch = hsv_f[:, :, 1]
    v_ch = hsv_f[:, :, 2]
    l_ch = lab_f[:, :, 0]
    a_ch = lab_f[:, :, 1]

    leaf_px  = leaf_mask > 0
    h, w     = img_rgb.shape[:2]
    combined = np.zeros((h, w), dtype=np.uint8)

    # â”€â”€ Otsu on leaf saturation â†’ dynamic green-exclusion threshold â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    s_leaf = s_ch[leaf_px].astype(np.uint8)
    s_otsu = _otsu_1d(s_leaf)
    green_mask = cv2.inRange(
        hsv_u8,
        np.array([28, max(15, int(s_otsu * 0.35)), 25], dtype=np.uint8),
        np.array([90, 255, 255], dtype=np.uint8),
    )

    if disease_class == "Common_Rust":
        # LAB a* warmth (rust = elevated red-green axis)
        a_leaf  = a_ch[leaf_px].astype(np.uint8)
        a_otsu  = _otsu_1d(a_leaf)
        warm    = (a_ch > max(a_otsu, 132.0)).astype(np.uint8) * 255

        # HSV hue: orange-brown band 0-25 + wrap-around 168-179
        hue_r1  = cv2.inRange(hsv_u8, np.array([0,  15, 15], np.uint8),
                                       np.array([25, 255, 255], np.uint8))
        hue_r2  = cv2.inRange(hsv_u8, np.array([168, 15, 15], np.uint8),
                                       np.array([179, 255, 255], np.uint8))
        combined = cv2.bitwise_or(warm, cv2.bitwise_or(hue_r1, hue_r2))

        # Gate: not very dark (shadow)
        v_leaf  = v_ch[leaf_px].astype(np.uint8)
        v_otsu  = _otsu_1d(v_leaf)
        bright  = (v_ch > max(v_otsu * 0.30, 18.0)).astype(np.uint8) * 255
        combined = cv2.bitwise_and(combined, bright)
        close_k, open_k = 3, 0

    elif disease_class == "Blight":
        # Signal 1 â€” local lightness deficit
        l_u8    = np.clip(l_ch, 0, 255).astype(np.uint8)
        l_blur  = cv2.GaussianBlur(l_u8, (31, 31), 10)
        l_diff  = cv2.subtract(l_blur, l_u8)
        ld_otsu = _otsu_1d(l_diff[leaf_px])
        dark    = (l_diff.astype(np.float32) > max(ld_otsu * 0.55, 6.0)
                   ).astype(np.uint8) * 255

        # Signal 2 â€” saturation drop (necrotic / water-soaked tissue)
        s_otsu2 = _otsu_1d(s_leaf)
        low_sat = (s_ch < max(s_otsu2 * 0.55, 25.0)).astype(np.uint8) * 255

        # Signal 3 â€” warm hue (tan-brown H 5-38)
        warm_hue = cv2.inRange(hsv_u8, np.array([5, 18, 18], np.uint8),
                                        np.array([38, 230, 240], np.uint8))

        combined = cv2.bitwise_or(dark, cv2.bitwise_or(low_sat, warm_hue))

        # Gates
        v_leaf  = v_ch[leaf_px].astype(np.uint8)
        v_otsu  = _otsu_1d(v_leaf)
        bright  = (v_ch > max(v_otsu * 0.25, 20.0)).astype(np.uint8) * 255
        not_dark = (l_ch > 22.0).astype(np.uint8) * 255
        combined = cv2.bitwise_and(combined, cv2.bitwise_and(bright, not_dark))
        close_k, open_k = 9, 5

    else:  # Gray_Leaf_Spot
        # Signal 1 â€” local saturation deficit (grey streak vs green surroundings)
        s_u8    = np.clip(s_ch, 0, 255).astype(np.uint8)
        s_blur  = cv2.GaussianBlur(s_u8, (21, 21), 7)
        s_diff  = cv2.subtract(s_blur, s_u8)
        sd_otsu = _otsu_1d(s_diff[leaf_px])
        desat   = (s_diff.astype(np.float32) > max(sd_otsu * 0.50, 8.0)
                   ).astype(np.uint8) * 255

        # Signal 2 â€” pale/grey pixels with very low absolute saturation
        s_otsu3 = _otsu_1d(s_leaf)
        pale    = (s_ch < max(s_otsu3 * 0.45, 20.0)).astype(np.uint8) * 255
        combined = cv2.bitwise_or(desat, pale)

        # Gates: tan/grey hue OK, bright enough, not dark
        tan_hue  = cv2.inRange(hsv_u8, np.array([0,  0,  40], np.uint8),
                                        np.array([38, 255, 255], np.uint8))
        very_pale = (s_ch < 50.0).astype(np.uint8) * 255
        hue_ok   = cv2.bitwise_or(tan_hue, very_pale)
        v_leaf   = v_ch[leaf_px].astype(np.uint8)
        v_otsu   = _otsu_1d(v_leaf)
        bright   = (v_ch > max(v_otsu * 0.30, 35.0)).astype(np.uint8) * 255
        not_dark = (l_ch > 30.0).astype(np.uint8) * 255
        combined = cv2.bitwise_and(combined,
                   cv2.bitwise_and(hue_ok,
                   cv2.bitwise_and(bright, not_dark)))
        close_k, open_k = 7, 2

    # â”€â”€ Remove green, restrict to leaf â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    combined = cv2.bitwise_and(combined, cv2.bitwise_not(green_mask))
    combined = cv2.bitwise_and(combined, leaf_mask)

    # â”€â”€ Morphological cleanup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    kern_c   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kern_c)
    if open_k > 0:
        kern_o   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kern_o)

    # â”€â”€ Flood guard â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    leaf_area = int(np.count_nonzero(leaf_mask))
    if leaf_area > 0 and int(np.count_nonzero(combined)) / leaf_area > 0.72:
        combined = np.zeros((h, w), dtype=np.uint8)

    return combined


def segment_full_leaf(img_rgb: np.ndarray) -> np.ndarray:
    """
    FIX: Added more positive points spread across the leaf (9-point grid instead
    of 5) so SAM2 doesn't miss leaf area near the edges.
    Negative corner points kept but shifted inward slightly to avoid cutting the
    leaf boundary when it is near the image edge.
    """
    h, w   = img_rgb.shape[:2]
    cx, cy = w // 2, h // 2

    # 3Ã—3 grid of positive points covering the leaf body
    pos = np.array([
        [cx,        cy       ],
        [cx - w//5, cy       ],
        [cx + w//5, cy       ],
        [cx,        cy - h//5],
        [cx,        cy + h//5],
        [cx - w//5, cy - h//5],
        [cx + w//5, cy - h//5],
        [cx - w//5, cy + h//5],
        [cx + w//5, cy + h//5],
    ], dtype=np.float32)

    # Inward negative points â€” avoid clipping real leaf at frame edges
    margin = 20
    neg = np.array([
        [margin,     margin   ],
        [w - margin, margin   ],
        [margin,     h-margin ],
        [w - margin, h-margin ],
    ], dtype=np.float32)

    pts = np.concatenate([pos, neg])
    lbs = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int32)

    sam_predictor.set_image(img_rgb)
    with torch.inference_mode():
        masks, scores, _ = sam_predictor.predict(
            point_coords=pts, point_labels=lbs, multimask_output=True)
    return masks[np.argmax(scores)].astype(np.uint8)


# â”€â”€ 7. SAM2 refinement helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _mask_to_sam2_logit(binary_mask: np.ndarray) -> np.ndarray:
    """
    Convert a binary (H, W) uint8 mask â†’ SAM2 mask_input tensor (1, 256, 256).
    Logit values: +10 = definitely foreground, -5 = likely background.
    """
    resized = cv2.resize(binary_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    logit   = np.where(resized > 0, 10.0, -5.0).astype(np.float32)
    return logit[np.newaxis, :, :]


def _sample_points_in_mask(binary: np.ndarray, n: int) -> np.ndarray:
    """
    FIX: Improved to cover elongated shapes (GLS streaks).

    Original greedy distance-transform approach clusters all points near the
    centroid of round blobs. For long narrow GLS streaks the first point sits
    near the centre and subsequent calls find nothing new because the radius
    exclusion covers most of the thin stripe.

    New approach: skeletonise the mask first, then sample evenly along the
    skeleton so points are distributed across the full length of the lesion.
    Falls back to the original distance-transform method for round/compact blobs.
    """
    if not np.any(binary):
        return np.empty((0, 2), dtype=np.float32)

    # Compute aspect ratio to decide strategy
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    use_skeleton = False
    if contours:
        rect = cv2.minAreaRect(np.vstack(contours))
        rw, rh = rect[1]
        aspect = max(rw, rh) / (min(rw, rh) + 1e-6)
        use_skeleton = aspect > 2.5   # elongated shape

    if use_skeleton:
        # Thin the mask to a 1-pixel skeleton and sample evenly along it
        from skimage.morphology import skeletonize
        skel = skeletonize(binary > 0).astype(np.uint8)
        ys, xs = np.where(skel > 0)
        if len(xs) >= n:
            indices = np.linspace(0, len(xs) - 1, n, dtype=int)
            return np.stack([xs[indices], ys[indices]], axis=1).astype(np.float32)
        elif len(xs) > 0:
            return np.stack([xs, ys], axis=1).astype(np.float32)

    # Fallback: distance-transform greedy for compact blobs
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    used = np.zeros_like(dist)
    pts  = []
    for _ in range(n):
        masked = dist * (1 - used)
        if masked.max() < 1:
            break
        yx = np.unravel_index(masked.argmax(), dist.shape)
        pts.append([yx[1], yx[0]])
        cv2.circle(used, (int(yx[1]), int(yx[0])), max(int(dist[yx]), 8), 1, -1)
    return np.array(pts, dtype=np.float32) if pts else np.empty((0, 2), dtype=np.float32)


def _sample_background_points(
    binary: np.ndarray, leaf_mask: np.ndarray, n: int = 3
) -> np.ndarray:
    """
    NEW: Sample negative (background) points from leaf tissue that is NOT
    disease.  Feeding these to SAM2 prevents mask bleeding into healthy tissue.
    """
    bg_mask = cv2.bitwise_and(leaf_mask, cv2.bitwise_not(binary))
    # Erode slightly so we don't accidentally land on the lesion border
    kern   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    bg_safe = cv2.erode(bg_mask, kern)
    if not np.any(bg_safe):
        bg_safe = bg_mask
    dist   = cv2.distanceTransform(bg_safe, cv2.DIST_L2, 5)
    used   = np.zeros_like(dist)
    pts    = []
    for _ in range(n):
        masked = dist * (1 - used)
        if masked.max() < 1:
            break
        yx = np.unravel_index(masked.argmax(), dist.shape)
        pts.append([yx[1], yx[0]])
        cv2.circle(used, (int(yx[1]), int(yx[0])), max(int(dist[yx]), 20), 1, -1)
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
                if (max(0, max(x0i, x0j) - min(x1i, x1j)) <= gap_px and
                        max(0, max(y0i, y0j) - min(y1i, y1j)) <= gap_px):
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
        max(x0 - pad, 0), max(y0 - pad, 0),
        min(x1 + pad, img_w - 1), min(y1 + pad, img_h - 1),
    ], dtype=np.float32)


# Disease-specific SAM2 thresholds
#
# Blight:
#   _SPOT_THRESHOLDS: 200 â†’ 80 â€” early small NCLB lesions get SAM2 refinement
#   _CLUSTER_GAPS:     50 â†’ 70 â€” yellow halo + necrotic core often have a gap
#   _IOU_THRESHOLDS:  0.25 â†’ 0.18 â€” blight rough masks are patchy across the
#     yellow-tan gradient; correct SAM2 masks score low IoU against them
#   _EXPANSION_LIMITS: 2.0 â†’ 2.8 â€” SAM2 is expected to expand to full lesion
#
# Common_Rust:
#   _SPOT_THRESHOLDS: 60 â†’ 40 â€” catch smaller/younger pustules
#   _IOU_THRESHOLDS: 0.25 â†’ 0.20 â€” rust clusters are slightly fragmented
#   _EXPANSION_LIMITS: 2.0 â†’ 2.5 â€” minor allowance for cluster expansion
#
# Gray_Leaf_Spot: unchanged from previous fix pass
_MIN_AREAS = {
    "Common_Rust":    3,
    "Blight":         8,
    "Gray_Leaf_Spot": 4,
}
_SPOT_THRESHOLDS = {
    "Common_Rust":    40,     # was 60
    "Blight":         80,     # was 200
    "Gray_Leaf_Spot": 50,
}
_CLUSTER_GAPS = {
    "Common_Rust":    25,
    "Blight":         70,     # was 50
    "Gray_Leaf_Spot": 65,
}
_IOU_THRESHOLDS = {
    "Common_Rust":    0.20,   # was 0.25
    "Blight":         0.18,   # was 0.25
    "Gray_Leaf_Spot": 0.12,
}
_EXPANSION_LIMITS = {
    "Common_Rust":    2.5,    # was 2.0
    "Blight":         2.8,    # was 2.0
    "Gray_Leaf_Spot": 3.5,
}


def refine_disease_mask_with_sam2(
    img_rgb: np.ndarray,
    rough_mask: np.ndarray,
    leaf_mask: np.ndarray,
    disease_class: str = "Common_Rust",
) -> tuple[np.ndarray, np.ndarray]:
    """
    3-pass mask-guided SAM2 refinement with disease-specific thresholds.

    Changes vs original:
    - Negative background points fed to SAM2 (prevents mask bleed).
    - Positive points increased to n=9 and use skeleton-aware sampling for
      elongated GLS streaks.
    - Per-disease IoU and expansion thresholds (see _IOU_THRESHOLDS /
      _EXPANSION_LIMITS) â€” GLS uses 0.12 / 3.5x instead of 0.25 / 2.0x.
    - If all three SAM2 passes fail the acceptance test, the rough cluster_bin
      is used BUT it is also AND-ed with leaf_inset (was missing before).
    """
    h, w = img_rgb.shape[:2]
    min_area       = _MIN_AREAS.get(disease_class, 6)
    spot_threshold = _SPOT_THRESHOLDS.get(disease_class, 120)
    cluster_gap    = _CLUSTER_GAPS.get(disease_class, 40)
    iou_min        = _IOU_THRESHOLDS.get(disease_class, 0.25)
    expansion_max  = _EXPANSION_LIMITS.get(disease_class, 2.0)

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

    for i in spot_ids:
        spot_mask |= (labels == i).astype(np.uint8)

    if not region_ids:
        return cv2.bitwise_and(spot_mask, leaf_mask), region_mask

    leaf_inset = cv2.erode(
        leaf_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    if np.count_nonzero(leaf_inset) < np.count_nonzero(leaf_mask) * 0.5:
        leaf_inset = leaf_mask

    clusters      = _cluster_components(labels, stats, num_labels,
                                        gap_px=cluster_gap, min_area=spot_threshold)
    clustered_ids = {idx for c in clusters for idx in c}
    for rid in region_ids:
        if rid not in clustered_ids:
            clusters.append([rid])

    sam_predictor.set_image(img_rgb)

    def _score_mask(m: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
        m        = cv2.bitwise_and(m, leaf_inset)
        area     = int(np.count_nonzero(m))
        ref_area = int(np.count_nonzero(ref))
        inter    = int(np.count_nonzero(m & ref))
        union    = int(np.count_nonzero(m | ref))
        iou      = inter / union  if union    > 0 else 0.0
        ratio    = area  / ref_area if ref_area > 0 else 999.0
        return iou, ratio

    def _ok(iou, ratio):
        return iou >= iou_min and ratio <= expansion_max

    for cluster in clusters:
        cluster_bin = np.zeros((h, w), dtype=np.uint8)
        for i in cluster:
            cluster_bin |= (labels == i).astype(np.uint8)
        if not np.count_nonzero(cluster_bin):
            continue

        box = _cluster_bbox(cluster, stats, w, h)

        # FIX: more points (9) with skeleton-aware sampling for elongated shapes
        pos_points = _sample_points_in_mask(cluster_bin, n=9)
        if len(pos_points) == 0:
            M = cv2.moments(cluster_bin)
            if M["m00"] > 0:
                pos_points = np.array([[int(M["m10"] / M["m00"]),
                                        int(M["m01"] / M["m00"])]], dtype=np.float32)
            else:
                region_mask |= cv2.bitwise_and(cluster_bin, leaf_inset)
                continue

        # FIX: add negative background points
        neg_points = _sample_background_points(cluster_bin, leaf_mask, n=3)
        if len(neg_points) > 0:
            all_points = np.vstack([pos_points, neg_points])
            all_labels = np.array(
                [1] * len(pos_points) + [0] * len(neg_points), dtype=np.int32)
        else:
            all_points = pos_points
            all_labels = np.ones(len(pos_points), dtype=np.int32)

        rough_logit = _mask_to_sam2_logit(cluster_bin)

        try:
            with torch.inference_mode():
                m1, s1, l1 = sam_predictor.predict(
                    point_coords=all_points, point_labels=all_labels,
                    box=box, multimask_output=True)
            p1       = m1[np.argmax(s1)].astype(np.uint8)
            p1_logit = l1[np.argmax(s1)]
        except Exception:
            region_mask |= cv2.bitwise_and(cluster_bin, leaf_inset)
            continue

        try:
            with torch.inference_mode():
                m2, _, l2 = sam_predictor.predict(
                    point_coords=all_points, point_labels=all_labels,
                    box=box, mask_input=rough_logit, multimask_output=False)
            p2       = m2[0].astype(np.uint8)
            p2_logit = l2[0]
        except Exception:
            p2 = p1; p2_logit = p1_logit

        try:
            with torch.inference_mode():
                m3, _, _ = sam_predictor.predict(
                    point_coords=all_points, point_labels=all_labels,
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
        else:
            # FIX: was missing leaf_inset AND â€” fallback now properly constrained
            chosen = cv2.bitwise_and(cluster_bin, leaf_inset)

        region_mask |= chosen

    return (cv2.bitwise_and(spot_mask,   leaf_mask),
            cv2.bitwise_and(region_mask, leaf_mask))


# â”€â”€ 8. Severity Calculation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
        zones.add((min(int((cy - y_min) / bh * 3), 2), min(int((cx - x_min) / bw * 3), 2)))
    n = len(zones)
    return "Dispersed" if n >= 7 else ("Scattered" if n >= 4 else "Localised")


def calculate_severity(
    leaf_mask: np.ndarray,
    spot_mask: np.ndarray,
    region_mask: np.ndarray,
    disease_class: str = "Unknown",
) -> dict:
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

    n_s, s_lbl, s_stats, s_ct = cv2.connectedComponentsWithStats(spot_u8,   8)
    n_r, r_lbl, r_stats, r_ct = cv2.connectedComponentsWithStats(region_u8, 8)

    spot_count    = max(n_s - 1, 0)
    region_count  = max(n_r - 1, 0)
    total_lesions = spot_count + region_count

    centroids = []
    for i in range(1, n_s): centroids.append((s_ct[i][0], s_ct[i][1]))
    for i in range(1, n_r): centroids.append((r_ct[i][0], r_ct[i][1]))
    centroids = np.array(centroids, dtype=np.float32) if centroids else np.empty((0, 2))

    areas = ([int(s_stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_s)] +
             [int(r_stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_r)])
    mean_lesion_size = round(float(np.mean(areas)), 1) if areas else 0.0

    distribution = _distribution_label(centroids, leaf_u8)
    hb           = _hb_grade(severity_pct)

    area_score         = 100.0 * np.log1p(severity_pct) / np.log(101)
    lesion_density     = round(total_lesions / leaf_area * 10_000, 3) if leaf_area > 0 else 0.0
    density_score      = min(lesion_density / 8.0 * 100.0, 100.0)
    dist_scores        = {"None": 0.0, "Unknown": 0.0, "Localised": 20.0,
                          "Scattered": 55.0, "Dispersed": 100.0}
    distribution_score = dist_scores.get(distribution, 0.0)
    progression_score  = (region_area / total_disease * 100.0) if total_disease > 0 else 0.0

    weighted_score = round(
        0.55 * area_score +
        0.15 * density_score +
        0.15 * distribution_score +
        0.15 * progression_score, 1
    )

    if   weighted_score < 10: label = "Trace"
    elif weighted_score < 30: label = "Mild"
    elif weighted_score < 52: label = "Moderate"
    elif weighted_score < 72: label = "Severe"
    else:                     label = "Very Severe"

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


# â”€â”€ 9. Overlay Encoder â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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


    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def encode_rough_mask(img_rgb: np.ndarray, rough_mask: np.ndarray) -> str:
    """Cyan tint overlay showing the Otsu rough colour mask."""
    overlay = img_rgb.copy()
    alpha   = 0.55
    if np.any(rough_mask > 0):
        overlay[rough_mask > 0] = (
            overlay[rough_mask > 0] * (1 - alpha) + np.array([0, 220, 220]) * alpha
        ).astype(np.uint8)
        conts, _ = cv2.findContours(rough_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, conts, -1, (0, 180, 180), 1)
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def encode_bboxes(img_rgb: np.ndarray, rough_mask: np.ndarray, leaf_mask: np.ndarray) -> str:
    """Yellow bounding boxes around each connected component in the rough mask."""
    overlay = img_rgb.copy()
    leaf_c, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, leaf_c, -1, (0, 200, 0), 1)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(rough_mask, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 6:
            continue
        x  = stats[i, cv2.CC_STAT_LEFT]
        y  = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (255, 220, 0), 2)
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# â”€â”€ 10. Prediction Endpoint â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr    = np.frombuffer(contents, np.uint8)
        img_bgr  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        prediction = classify_image_with_tta(img_rgb, n_augments=6)

        severity_pct        = 0.0
        severity_label      = "N/A"
        hb_grade            = 0
        severity_score      = 0.0
        overlay_b64         = None
        rough_mask_b64      = None
        bboxes_b64          = None
        spot_count          = 0
        region_count        = 0
        spot_severity_pct   = 0.0
        region_severity_pct = 0.0
        mean_lesion_size_px = 0.0
        lesion_density      = 0.0
        distribution        = "None"

        if prediction != "Healthy":
            leaf_mask = segment_full_leaf(img_rgb)

            if np.count_nonzero(leaf_mask) > 0:
                rough_mask = create_color_mask_within_leaf(
                    img_rgb, leaf_mask, disease_class=prediction)

                rough_mask_b64 = encode_rough_mask(img_rgb, rough_mask)
                bboxes_b64     = encode_bboxes(img_rgb, rough_mask, leaf_mask)

                spot_mask, region_mask = refine_disease_mask_with_sam2(
                    img_rgb, rough_mask, leaf_mask, disease_class=prediction)

                m = calculate_severity(
                    leaf_mask, spot_mask, region_mask, disease_class=prediction)

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
            "severity_percentage":  severity_pct,
            "severity_label":       severity_label,
            "hb_grade":             hb_grade,
            "severity_score":       severity_score,
            "rough_mask_image":     rough_mask_b64,
            "bboxes_image":         bboxes_b64,
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


# â”€â”€ 11. Entry Point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
