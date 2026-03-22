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

# ── 2. Model Definition ────────────────────────────────────────────────────────
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


# ── 3. Load Models ─────────────────────────────────────────────────────────────
CLASSIFIER_FILENAME = "CustomMobileNetV2_2_best.pth"
SAM2_CHECKPOINT     = "sam2.1_hiera_large.pt"
SAM2_CONFIG         = "configs/sam2.1/sam2.1_hiera_l.yaml"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classifier = CustomMobileNetV2_3(num_classes=len(CLASS_NAMES))
classifier.load_state_dict(
    torch.load(CLASSIFIER_FILENAME, map_location=device, weights_only=False)
)
classifier.to(device)
classifier.eval()
print(f"Classifier loaded successfully. {classifier.param_summary}")

sam2_model    = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
sam_predictor = SAM2ImagePredictor(sam2_model)
print("SAM2 loaded successfully.")


# ── 4. Inference helper ────────────────────────────────────────────────────────
def classify_image_with_tta(img_rgb: np.ndarray, n_augments: int = 6) -> tuple[str, str]:
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
    return CLASS_NAMES[idx], f"{avg[idx].item() * 100:.2f}%"


# ── 5. Disease colour profiles ─────────────────────────────────────────────────
#
# Blight — expanded to cover all NCLB appearance stages:
#   • Water-soaked gray-green (early)
#   • Tan/straw cigar lesion body (mid)
#   • Dark brown necrotic centre (mature)
#   • Yellow-green halo zone
#   • Near-white bleached necrotic core
#   departure_sensitivity raised 0.30 → 0.45 to flag moderate-saturation tan tissue
#
# Common_Rust — expanded to cover late-stage pustules:
#   • Dark brown/black aged urediniospores (V 15-80)
#   • Faded old pustules with yellow halo (H 20-35)
#   LAB range added for dark mature rust
#
# Gray_Leaf_Spot — unchanged from previous fix pass

DISEASE_COLOR_PROFILES = {
    "Blight": {
        "hsv_ranges": [
            # Original tan/brown lesion body
            (np.array([10, 50,  80]),  np.array([25, 200, 220])),
            (np.array([5,  40,  40]),  np.array([15, 180, 160])),
            # NEW: water-soaked early lesion — gray-green, low-moderate S
            (np.array([35, 20,  60]),  np.array([75, 70,  160])),
            # NEW: dark brown necrotic centre — low V, moderate-high S
            (np.array([5,  55,  20]),  np.array([22, 200,  90])),
            # NEW: yellow-green halo zone around lesion
            (np.array([22, 60,  130]), np.array([38, 200, 240])),
            # NEW: near-white bleached necrotic core
            (np.array([0,  0,   180]), np.array([30, 35,  255])),
        ],
        "lab_ranges": [
            # Original buff/tan
            (np.array([60,  133, 145]), np.array([170, 155, 180])),
            # NEW: dark necrotic tissue — low L, warm a/b
            (np.array([20,  130, 135]), np.array([85,  152, 162])),
            # NEW: near-white necrotic — high L, near-neutral
            (np.array([155, 120, 128]), np.array([230, 134, 142])),
        ],
        "exclude_green_s_min":   45,   # was 55 — some blight-adjacent tissue has S 45-55
        "morph_close_k":          7,   # was 5 — larger close to bridge yellow-halo to necrotic core
        "morph_open_k":           3,
        "use_green_departure":   True,
        "departure_sensitivity": 0.45, # was 0.30 — flags moderate-S tan tissue (S < 45% of green median)
        "use_blight_local":      True, # NEW
    },
    "Common_Rust": {
        "hsv_ranges": [
            # Fresh orange-brown pustules
            (np.array([5,  25,  30]),  np.array([25, 255, 255])),
            (np.array([0,  20,  25]),  np.array([8,  200, 180])),
            (np.array([170, 20, 25]),  np.array([180, 200, 180])),
            # NEW: late-stage dark brown/black urediniospores
            (np.array([5,  60,  15]),  np.array([20, 220,  80])),
            # NEW: faded old pustules with yellow-brown halo
            (np.array([18, 40,  100]), np.array([35, 160, 200])),
        ],
        "lab_ranges": [
            (np.array([30, 135, 140]), np.array([220, 185, 210])),
            (np.array([15, 128, 130]), np.array([130, 162, 172])),
            # NEW: dark mature rust — low L, warm tones
            (np.array([10, 130, 132]), np.array([80,  158, 160])),
        ],
        "exclude_green_s_min":  40,
        "morph_close_k":         3,
        "morph_open_k":          0,   # MUST stay 0 — opening kills tiny pustules
        "use_green_departure":  False,
        "use_rust_local":       True,
    },
    "Gray_Leaf_Spot": {
        "hsv_ranges": [
            # Mature gray lesions — very low saturation, wide hue
            (np.array([0,  0,   100]), np.array([35, 40,  245])),
            # Mid-stage buff/straw — moderate S, tan hue
            (np.array([10, 20,  70]),  np.array([35, 110, 220])),
            # Early tan-yellow lesions — higher S, lighter V
            (np.array([15, 30,  130]), np.array([38, 160, 255])),
            # Dark brown edges of lesions
            (np.array([5,  25,  45]),  np.array([22, 160, 165])),
            # Near-white necrotic centres
            (np.array([0,  0,   180]), np.array([30, 25,  255])),
        ],
        "lab_ranges": [
            # Primary buff/gray range
            (np.array([75, 124, 132]), np.array([175, 150, 168])),
            # Darker lesion edges
            (np.array([40, 124, 130]), np.array([120, 148, 162])),
            # Near-neutral pale tissue
            (np.array([140, 122, 126]), np.array([220, 136, 140])),
        ],
        "exclude_green_s_min":  28,   # was 30 — keep slightly permissive
        "morph_close_k":        7,    # was 5 — close gaps between streak fragments
        "morph_open_k":         2,    # was 3 — smaller open to keep thin streaks
        "use_green_departure":  False,
        "use_gls_local":        True,
    },
}


# ── 6. Adaptive local-contrast detectors ──────────────────────────────────────

def _detect_rust_local_contrast(
    img_rgb: np.ndarray,
    leaf_mask: np.ndarray,
) -> np.ndarray:
    """
    Multi-scale local LAB warmth detector for Common Rust pustules.

    Warmth = (a* − local_a*) + 0.6 × (b* − local_b*)

    Changes vs original:
    - 7px threshold 5.0 → 4.0: catches faint early-stage and light-coloured
      pustules that the original missed.
    - V floor 25 → 18: late-stage dark brown/black urediniospores have
      V as low as 15–20; the original floor excluded them entirely.
    - Added 35px scale at threshold 9.5: catches mature coalescing pustule
      clusters that span a wider neighbourhood.
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    lab     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_ch    = hsv[:, :, 0]
    v_ch    = hsv[:, :, 2]

    a_ch = lab[:, :, 1]
    b_ch = lab[:, :, 2]

    combined = np.zeros(img_rgb.shape[:2], dtype=np.uint8)

    # (kernel_size, warmth_threshold)
    # 7px  — micro-pustules (2-8 px)
    # 15px — typical pustules
    # 25px — larger mature pustules
    # 35px — coalescing clusters (NEW)
    for ksize, threshold in [(7, 4.0), (15, 6.5), (25, 8.0), (35, 9.5)]:
        ksize   = ksize | 1
        a_local = cv2.GaussianBlur(a_ch, (ksize, ksize), ksize / 3.0)
        b_local = cv2.GaussianBlur(b_ch, (ksize, ksize), ksize / 3.0)
        warmth  = (a_ch - a_local) + 0.6 * (b_ch - b_local)
        hot     = (warmth > threshold).astype(np.uint8) * 255
        combined = cv2.bitwise_or(combined, hot)

    not_green = cv2.bitwise_not(cv2.inRange(h_ch, np.array([28]), np.array([90])))
    # FIX: lowered 25 → 18 — dark mature pustules have V as low as 15-20
    bright    = (v_ch > 18).astype(np.uint8) * 255
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
    local_kernel: int = 21,        # was 25 — tighter local window finds smaller lesions
    deficit_threshold: float = 14.0,  # was 28.0 — GLS has subtle dips (~15 units)
    neib_threshold: float = 6.0,   # was 12.0 — less strict neighbourhood confirmation
) -> np.ndarray:
    """
    Detect Gray Leaf Spot lesions via local HSV saturation contrast.

    GLS lesions are elongated, tan/gray areas that are markedly LESS saturated
    than surrounding green tissue.

    Key changes vs original:
    - `deficit_threshold` lowered 28 → 14: GLS saturation dips are subtle (~15–20
      units), the original threshold missed most lesions entirely.
    - `neib_threshold` lowered 12 → 6: neighbourhood confirmation was rejecting
      real lesions because fragmented patches fail the neighbourhood check.
    - `very_pale` S threshold raised 18 → 45: real GLS tissue has S ≈ 20–55;
      original cutoff was too aggressive.
    - Multi-scale: three kernel sizes to catch both narrow streaks and wide patches.
    - Saturation+lightness dual-channel: adds an L-channel check so very dark
      non-green pixels (soil, shadow) are excluded.
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    h_ch = hsv[:, :, 0].astype(np.uint8)
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2].astype(np.uint8)
    l_ch = lab[:, :, 0]   # lightness — exclude very dark pixels

    combined = np.zeros(img_rgb.shape[:2], dtype=np.uint8)

    # Multi-scale: tight kernel catches thin streaks; wide kernel catches patches
    for ksize, d_thresh, n_thresh in [
        (11, deficit_threshold,        neib_threshold),
        (21, deficit_threshold * 1.2,  neib_threshold * 1.1),
        (35, deficit_threshold * 1.5,  neib_threshold * 1.3),
    ]:
        ksize   = ksize | 1
        s_local = cv2.GaussianBlur(s_ch, (ksize, ksize), ksize / 3.0)
        deficit = s_local - s_ch   # positive where pixel is LESS saturated than surroundings

        neib_k     = 13 | 1
        deficit_u8 = np.clip(deficit * 2 + 128, 0, 255).astype(np.uint8)
        neib_u8    = cv2.GaussianBlur(deficit_u8, (neib_k, neib_k), neib_k / 3.0)
        neib_score = (neib_u8.astype(np.float32) - 128) / 2.0

        desaturated = ((deficit > d_thresh) & (neib_score > n_thresh)).astype(np.uint8) * 255
        combined    = cv2.bitwise_or(combined, desaturated)

    tan_hue   = cv2.inRange(h_ch, np.array([0]),  np.array([38]))
    # FIX: raised threshold 18 → 45 — real GLS tissue has S up to ~55
    very_pale = (hsv[:, :, 1] < 45).astype(np.uint8) * 255
    hue_ok    = cv2.bitwise_or(tan_hue, very_pale)

    not_green = cv2.bitwise_not(cv2.inRange(h_ch, np.array([28]), np.array([88])))
    # FIX: lowered brightness floor 70 → 45 — some GLS lesions are darker
    bright    = (v_ch > 45).astype(np.uint8) * 255
    # Exclude very dark pixels (shadow / soil that sneaked through leaf mask)
    not_dark  = (l_ch > 35).astype(np.uint8) * 255

    result = cv2.bitwise_and(combined,  hue_ok)
    result = cv2.bitwise_and(result,    not_green)
    result = cv2.bitwise_and(result,    bright)
    result = cv2.bitwise_and(result,    not_dark)
    result = cv2.bitwise_and(result,    leaf_mask)

    kern_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kern_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kern_c)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN,  kern_o)

    leaf_area = int(np.count_nonzero(leaf_mask))
    if leaf_area > 0 and int(np.count_nonzero(result)) / leaf_area > 0.65:
        return np.zeros(img_rgb.shape[:2], dtype=np.uint8)

    return result


def _detect_blight_local_contrast(
    img_rgb: np.ndarray,
    leaf_mask: np.ndarray,
) -> np.ndarray:
    """
    Detect Blight (NCLB) lesions via dual local-contrast signals in LAB space.

    NCLB presents three co-occurring visual signals that fixed HSV ranges miss:
      1. Lightness deficit — lesion cores are locally darker than surrounding
         healthy green tissue (L drops 8–25 units below neighbourhood mean).
      2. Warmth shift — tan/brown tissue has elevated a* and b* relative to
         the local green neighbourhood (same direction as rust but weaker signal
         spread over a much larger area).
      3. Saturation departure — necrotic and water-soaked tissue is less
         saturated than neighbouring healthy leaf (S deficit ≥ 10).

    All three signals are OR-ed so that any stage of the disease is caught:
      - Water-soaked early stage: mainly S deficit + weak warmth
      - Cigar-body mid stage: mainly lightness deficit + moderate warmth
      - Necrotic mature stage: strong lightness deficit + S deficit

    Gates:
      - not_green: hue must not be in healthy green range
      - bright: V > 35 (avoids camera noise in deep shadows)
      - not_dark (L > 25): avoids picking up dark background/soil

    Flood guard: >70 % leaf coverage → discard (likely to be a healthy
    shaded leaf, not actual blight).

    Kernel sizes use wide windows (21–55 px) because NCLB lesions are large
    (2–15 cm) and the local neighbourhood must be big enough to span across
    healthy tissue on both sides.
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    lab     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    h_ch = hsv[:, :, 0].astype(np.uint8)
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2].astype(np.uint8)
    l_ch = lab[:, :, 0]
    a_ch = lab[:, :, 1]
    b_ch = lab[:, :, 2]

    combined = np.zeros(img_rgb.shape[:2], dtype=np.uint8)

    # ── Signal 1: local lightness deficit ─────────────────────────────────────
    # Large kernels needed — NCLB lesions are wide and the neighbourhood must
    # span into adjacent healthy tissue to compute a meaningful local mean.
    for ksize, l_thresh in [(21, 8.0), (35, 11.0), (55, 14.0)]:
        ksize   = ksize | 1
        l_local = cv2.GaussianBlur(l_ch, (ksize, ksize), ksize / 3.0)
        deficit = l_local - l_ch          # positive where pixel is DARKER
        dark    = (deficit > l_thresh).astype(np.uint8) * 255
        combined = cv2.bitwise_or(combined, dark)

    # ── Signal 2: local warmth boost (tan/brown hue shift) ────────────────────
    for ksize, w_thresh in [(21, 4.0), (35, 6.0), (55, 8.0)]:
        ksize   = ksize | 1
        a_local = cv2.GaussianBlur(a_ch, (ksize, ksize), ksize / 3.0)
        b_local = cv2.GaussianBlur(b_ch, (ksize, ksize), ksize / 3.0)
        warmth  = (a_ch - a_local) + 0.5 * (b_ch - b_local)
        warm    = (warmth > w_thresh).astype(np.uint8) * 255
        combined = cv2.bitwise_or(combined, warm)

    # ── Signal 3: local saturation departure (necrotic / water-soaked) ────────
    for ksize, s_thresh in [(21, 10.0), (35, 13.0)]:
        ksize   = ksize | 1
        s_local = cv2.GaussianBlur(s_ch, (ksize, ksize), ksize / 3.0)
        s_def   = s_local - s_ch          # positive where pixel is LESS saturated
        desat   = (s_def > s_thresh).astype(np.uint8) * 255
        combined = cv2.bitwise_or(combined, desat)

    # ── Gates ─────────────────────────────────────────────────────────────────
    not_green  = cv2.bitwise_not(cv2.inRange(h_ch, np.array([30]), np.array([88])))
    bright     = (v_ch > 35).astype(np.uint8) * 255
    not_dark   = (l_ch > 25).astype(np.uint8) * 255
    # Must be tan/brown/neutral — exclude vivid non-green hues (e.g. flower)
    tan_range  = cv2.inRange(h_ch, np.array([0]),  np.array([38]))
    near_neut  = (s_ch < 90).astype(np.uint8) * 255
    hue_ok     = cv2.bitwise_or(tan_range, near_neut)

    result = cv2.bitwise_and(combined, not_green)
    result = cv2.bitwise_and(result,   bright)
    result = cv2.bitwise_and(result,   not_dark)
    result = cv2.bitwise_and(result,   hue_ok)
    result = cv2.bitwise_and(result,   leaf_mask)

    kern_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kern_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kern_c)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN,  kern_o)

    leaf_area = int(np.count_nonzero(leaf_mask))
    if leaf_area > 0 and int(np.count_nonzero(result)) / leaf_area > 0.70:
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

    for lo, hi in profile["hsv_ranges"]:
        combined |= cv2.inRange(hsv, lo, hi)
    for lo, hi in profile["lab_ranges"]:
        combined |= cv2.inRange(lab, lo, hi)

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

    if profile.get("use_rust_local", False):
        combined |= _detect_rust_local_contrast(img_rgb, leaf_mask)

    if profile.get("use_blight_local", False):
        combined |= _detect_blight_local_contrast(img_rgb, leaf_mask)

    if profile.get("use_gls_local", False):
        combined |= _detect_gls_local_contrast(img_rgb, leaf_mask)

    s_min      = profile.get("exclude_green_s_min", 40)
    green_mask = cv2.inRange(hsv, np.array([28, s_min, 30]), np.array([90, 255, 255]))
    combined   = cv2.bitwise_and(combined, cv2.bitwise_not(green_mask))
    combined   = cv2.bitwise_and(combined, leaf_mask)

    close_k  = profile.get("morph_close_k", 5)
    open_k   = profile.get("morph_open_k",  3)
    kern_c   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kern_c)
    if open_k > 0:
        kern_o   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kern_o)

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

    # 3×3 grid of positive points covering the leaf body
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

    # Inward negative points — avoid clipping real leaf at frame edges
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


# ── 7. SAM2 refinement helpers ─────────────────────────────────────────────────

def _mask_to_sam2_logit(binary_mask: np.ndarray) -> np.ndarray:
    """
    Convert a binary (H, W) uint8 mask → SAM2 mask_input tensor (1, 256, 256).
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
#   _SPOT_THRESHOLDS: 200 → 80 — early small NCLB lesions get SAM2 refinement
#   _CLUSTER_GAPS:     50 → 70 — yellow halo + necrotic core often have a gap
#   _IOU_THRESHOLDS:  0.25 → 0.18 — blight rough masks are patchy across the
#     yellow-tan gradient; correct SAM2 masks score low IoU against them
#   _EXPANSION_LIMITS: 2.0 → 2.8 — SAM2 is expected to expand to full lesion
#
# Common_Rust:
#   _SPOT_THRESHOLDS: 60 → 40 — catch smaller/younger pustules
#   _IOU_THRESHOLDS: 0.25 → 0.20 — rust clusters are slightly fragmented
#   _EXPANSION_LIMITS: 2.0 → 2.5 — minor allowance for cluster expansion
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
      _EXPANSION_LIMITS) — GLS uses 0.12 / 3.5x instead of 0.25 / 2.0x.
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
            # FIX: was missing leaf_inset AND — fallback now properly constrained
            chosen = cv2.bitwise_and(cluster_bin, leaf_inset)

        region_mask |= chosen

    return (cv2.bitwise_and(spot_mask,   leaf_mask),
            cv2.bitwise_and(region_mask, leaf_mask))


# ── 8. Severity Calculation ────────────────────────────────────────────────────

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


# ── 9. Overlay Encoder ────────────────────────────────────────────────────────
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
    cv2.rectangle(overlay, (lx, ly), (lx+lw, ly+lh), (0, 0, 0), -1)
    overlay[ly:ly+lh, lx:lx+lw] = (roi*0.3 + overlay[ly:ly+lh, lx:lx+lw]*0.7).astype(np.uint8)
    cv2.rectangle(overlay, (lx+8,  ly+8),  (lx+22, ly+22), (255, 165, 0), -1)
    cv2.putText(overlay, "Spots",   (lx+28, ly+20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    cv2.rectangle(overlay, (lx+8,  ly+28), (lx+22, ly+42), (220,  30, 30), -1)
    cv2.putText(overlay, "Regions", (lx+28, ly+40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── 10. Prediction Endpoint ────────────────────────────────────────────────────
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
            leaf_mask = segment_full_leaf(img_rgb)

            if np.count_nonzero(leaf_mask) > 0:
                rough_mask = create_color_mask_within_leaf(
                    img_rgb, leaf_mask, disease_class=prediction)

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


# ── 11. Entry Point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))