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
from PIL import Image
import uvicorn

# ── 1. App Setup ───────────────────────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://crop-disease-detection-tau.vercel.app/",
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
SAM2_CHECKPOINT = "sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classification model
classifier = CustomMobileNetV2_3(num_classes=len(CLASS_NAMES))
classifier.load_state_dict(torch.load(CLASSIFIER_FILENAME, map_location=device))
classifier.to(device)
classifier.eval()
print("Classifier loaded successfully.")

# SAM2 predictor
sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
sam_predictor = SAM2ImagePredictor(sam2_model)
print("SAM2 loaded successfully.")

# Preprocessing transform for classifier
preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# ── 4. Helper Functions ────────────────────────────────────────────────────────
def classify_image(img_rgb: np.ndarray) -> tuple[str, str]:
    """Run classifier and return (class_name, confidence_str)."""
    img_pil = Image.fromarray(img_rgb)
    tensor = preprocess(img_pil).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = classifier(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    pred_idx = probs.argmax().item()
    confidence = f"{probs[pred_idx].item() * 100:.2f}%"
    return CLASS_NAMES[pred_idx], confidence


def segment_full_leaf(img_rgb: np.ndarray) -> np.ndarray:
    """
    SAM2: segment the entire leaf using center positive points
    and corner negative points.
    Returns binary uint8 mask (H, W).
    """
    h, w = img_rgb.shape[:2]
    cx, cy = w // 2, h // 2

    pos_points = np.array(
        [
            [cx, cy],
            [cx - w // 6, cy],
            [cx + w // 6, cy],
            [cx, cy - h // 6],
            [cx, cy + h // 6],
        ],
        dtype=np.float32,
    )
    neg_points = np.array(
        [
            [10, 10],
            [w - 10, 10],
            [10, h - 10],
            [w - 10, h - 10],
        ],
        dtype=np.float32,
    )

    points = np.concatenate([pos_points, neg_points], axis=0)
    labels = np.array([1] * len(pos_points) + [0] * len(neg_points), dtype=np.int32)

    sam_predictor.set_image(img_rgb)
    with torch.inference_mode():
        masks, scores, _ = sam_predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
    return masks[np.argmax(scores)].astype(np.uint8)


def create_color_mask_within_leaf(
    img_rgb: np.ndarray, leaf_mask: np.ndarray
) -> np.ndarray:
    """
    Colour threshold for yellow/orange/brown/red disease regions,
    constrained inside the leaf boundary.
    Returns binary uint8 mask (H, W).
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    yellow = cv2.inRange(hsv, np.array([15, 50, 50]), np.array([35, 255, 255]))
    brown = cv2.inRange(hsv, np.array([5, 50, 20]), np.array([20, 255, 200]))
    red1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))

    combined = yellow | brown | red1 | red2

    # Constrain to leaf only
    combined = cv2.bitwise_and(combined, combined, mask=leaf_mask)

    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    return combined


def calculate_severity(leaf_mask: np.ndarray, disease_mask: np.ndarray) -> dict:
    """
    Compute severity and return metrics + disease-on-leaf mask.
    """
    leaf_u8 = (leaf_mask > 0).astype(np.uint8)
    disease_u8 = (disease_mask > 0).astype(np.uint8)

    disease_on_leaf = cv2.bitwise_and(disease_u8, leaf_u8)
    leaf_area = int(np.count_nonzero(leaf_u8))
    disease_area = int(np.count_nonzero(disease_on_leaf))
    severity_pct = round(disease_area / leaf_area * 100, 2) if leaf_area > 0 else 0.0

    if severity_pct < 5:
        label = "Mild"
    elif severity_pct < 15:
        label = "Moderate"
    else:
        label = "Severe"

    return {
        "leaf_area_px": leaf_area,
        "disease_area_px": disease_area,
        "severity_pct": severity_pct,
        "severity_label": label,
        "disease_on_leaf_mask": disease_on_leaf,
    }


def encode_overlay(img_rgb: np.ndarray, disease_mask: np.ndarray) -> str:
    """Burn disease mask onto image and return base64-encoded JPEG."""
    overlay = img_rgb.copy()
    overlay[disease_mask > 0] = (
        overlay[disease_mask > 0] * 0.4 + np.array([255, 0, 0]) * 0.6
    ).astype(np.uint8)
    pil_img = Image.fromarray(overlay)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG", quality=90)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# ── 5. Prediction Endpoint ─────────────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Step 1 — Classification
        prediction, confidence = classify_image(img_rgb)

        # Defaults for healthy leaf
        severity_pct = 0.0
        severity_label = "N/A"
        overlay_b64 = None

        if prediction != "Healthy":
            # Step 2 — SAM2 leaf segmentation
            leaf_mask = segment_full_leaf(img_rgb)

            if np.count_nonzero(leaf_mask) > 0:
                # Step 3 — Colour threshold inside leaf
                disease_mask = create_color_mask_within_leaf(img_rgb, leaf_mask)

                # Step 4 — Severity
                metrics = calculate_severity(leaf_mask, disease_mask)
                severity_pct = metrics["severity_pct"]
                severity_label = metrics["severity_label"]

                # Step 5 — Overlay image
                overlay_b64 = encode_overlay(img_rgb, metrics["disease_on_leaf_mask"])

        return {
            "prediction": prediction,
            "confidence": confidence,
            "severity_percentage": severity_pct,
            "severity_label": severity_label,
            "sam_mask_image": overlay_b64,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 6. Entry Point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
