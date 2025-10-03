import os
import cv2
import numpy as np
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from segment_anything import sam_model_registry, SamPredictor
import onnxruntime as ort
from PIL import Image
import uvicorn
import io

# --- 1. Application Setup ---
app = FastAPI()

origins = [
    "http://localhost:3000",
    "https://crop-disease-detection-tau.vercel.app/"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. Load Models (Assuming they were downloaded during the build step) ---
SAM_CHECKPOINT_FILENAME = "sam_vit_b_01ec64.pth"
CLASSIFIER_FILENAME = "mobilenet_classifier.onnx"

# Load SAM model
device = "cuda" if torch.cuda.is_available() else "cpu"
sam_model = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT_FILENAME)
sam_model.to(device=device)
sam_predictor = SamPredictor(sam_model)
print("SAM model loaded successfully.")

# Load ONNX Classification model
ort_session = ort.InferenceSession(CLASSIFIER_FILENAME)
print("ONNX classification model loaded successfully.")

CLASS_NAMES = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

# --- 3. Helper Functions (These remain the same) ---
def create_color_mask(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    yellow_lower = np.array([15, 50, 50])
    yellow_upper = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    brown_lower = np.array([5, 50, 20])
    brown_upper = np.array([20, 255, 200])
    brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)
    red_lower1 = np.array([0, 120, 70])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 120, 70])
    red_upper2 = np.array([180, 255, 255])
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    diseased_mask = cv2.bitwise_or(yellow_mask, brown_mask)
    diseased_mask = cv2.bitwise_or(diseased_mask, red_mask)
    kernel = np.ones((3,3), np.uint8)
    diseased_mask = cv2.morphologyEx(diseased_mask, cv2.MORPH_CLOSE, kernel)
    diseased_mask = cv2.morphologyEx(diseased_mask, cv2.MORPH_OPEN, kernel)
    return diseased_mask

def mask_to_boxes(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    return np.array([[x, y, x + w, y + h] for x, y, w, h in boxes]) if boxes else None

def segment_leaf_with_sam(image_rgb, predictor):
    predictor.set_image(image_rgb)
    h, w, _ = image_rgb.shape
    input_point = np.array([[w // 2, h // 2]])
    input_label = np.array([1])
    masks, scores, _ = predictor.predict(
        point_coords=input_point, point_labels=input_label, multimask_output=True
    )
    return masks[np.argmax(scores)]

# --- 4. Main Prediction Endpoint ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Classification with ONNX model
        img_pil = Image.fromarray(img_rgb).resize((224, 224))
        img_array = np.array(img_pil, dtype=np.float32)[np.newaxis, ...]
        input_name = ort_session.get_inputs()[0].name
        ort_outs = ort_session.run(None, {input_name: img_array})
        scores = ort_outs[0][0]
        prediction = CLASS_NAMES[np.argmax(scores)]
        confidence = f"{np.max(scores) * 100:.2f}%"

        severity_percentage, severity_label, final_overlay_encoded = 0, "N/A", None

        if prediction != 'Healthy':
            leaf_mask = segment_leaf_with_sam(img_rgb, sam_predictor)
            leaf_area = np.sum(leaf_mask)
            if leaf_area > 0:
                initial_disease_mask = create_color_mask(img_bgr)
                disease_boxes = mask_to_boxes(initial_disease_mask)
                combined_disease_mask = np.zeros_like(leaf_mask, dtype=bool)
                if disease_boxes is not None:
                    sam_predictor.set_image(img_rgb)
                    for box in disease_boxes:
                        masks, _, _ = sam_predictor.predict(box=box, multimask_output=False)
                        combined_disease_mask = np.logical_or(combined_disease_mask, masks[0])
                
                disease_on_leaf_mask = np.logical_and(leaf_mask, combined_disease_mask)
                disease_area = np.sum(disease_on_leaf_mask)
                severity_percentage = round((disease_area / leaf_area) * 100, 2)
                
                if severity_percentage < 5: severity_label = "Mild"
                elif 5 <= severity_percentage < 15: severity_label = "Moderate"
                else: severity_label = "Severe"

                overlay = img_rgb.copy()
                overlay[disease_on_leaf_mask] = [255, 0, 0]
                final_overlay_pil = Image.fromarray(overlay)
                buffered = io.BytesIO()
                final_overlay_pil.save(buffered, format="JPEG")
                final_overlay_encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "prediction": prediction, "confidence": confidence,
            "severity_percentage": severity_percentage, "severity_label": severity_label,
            "sam_mask_image": final_overlay_encoded,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# For Render deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

