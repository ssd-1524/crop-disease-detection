import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import tempfile
import os
from segment_anything import sam_model_registry, SamPredictor
import base64
from io import BytesIO
import traceback
import shutil

# --- Application and Model Configuration ---
app = FastAPI(
    title="Maize Disease Detection API v3",
    description="API for multi-stage maize disease classification and severity analysis using MobileNet, U-Net, and SAM.",
    version="3.0.0"
)

# --- CORS Configuration ---
origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Variables for Models ---
classification_model = None
segmentation_model = None
sam_predictor = None
CLASS_NAMES = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
IMG_WIDTH, IMG_HEIGHT = 224, 224


# --- Model Loading on Startup ---
@app.on_event("startup")
def load_models():
    """Load all machine learning models into memory when the application starts."""
    global classification_model, segmentation_model, sam_predictor
    
    print("Loading machine learning models...")
    
    classification_model = tf.keras.models.load_model("mobilenet_classification_model.h5")
    segmentation_model = tf.keras.models.load_model('unet_segmentation_model.h5')
    
    # Use the smaller, more memory-efficient SAM model
    sam_checkpoint = "sam_vit_b_01ec64.pth" 
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam_predictor = SamPredictor(sam)
    
    print("All models loaded successfully.")

# --- Helper Functions ---

def mask_to_boxes(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append([x, y, x + w, y + h])
    return np.array(boxes) if boxes else None

def sam_segment_with_boxes(image: np.ndarray, boxes: np.ndarray, predictor: SamPredictor):
    predictor.set_image(image)
    all_masks = []
    if boxes is not None:
        for box in boxes:
            masks, _, _ = predictor.predict(
                point_coords=None, point_labels=None, box=box[None, :], multimask_output=False,
            )
            all_masks.append(masks[0])
    return all_masks

def segment_leaf(image_path: str):
    image = cv2.imread(image_path)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    leaf_mask = cv2.inRange(image_hsv, lower_green, upper_green)
    kernel = np.ones((5, 5), np.uint8)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)
    return leaf_mask

def calculate_severity(mask, total_area):
    if total_area == 0:
        return 0.0
    diseased_area = np.sum(mask > 0)
    severity = (diseased_area / total_area) * 100
    return severity

# --- NEW: Function to create a colored overlay ---
def create_colored_overlay(image, mask, color=(255, 0, 0), alpha=0.5):
    """
    Overlays a semi-transparent colored mask onto an image.
    """
    # Create a colored version of the mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color
    
    # Blend the original image and the colored mask
    # cv2.addWeighted calculates: dst = src1*alpha + src2*beta + gamma
    overlay = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
    
    return overlay

# --- Main Prediction Pipeline ---
def predict_disease_and_severity(image_path, classification_model, segmentation_model, sam_predictor, class_names):
    
    # 1. Image Preprocessing
    img = cv2.imread(image_path)
    original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(original_image, (IMG_WIDTH, IMG_HEIGHT))
    img_processed = img_resized.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_processed, axis=0)

    # 2. Disease Classification
    predictions = classification_model.predict(img_input)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_names[predicted_class_index]
    confidence_score = float(np.max(predictions, axis=1)[0])

    if predicted_class == 'Healthy':
        return (predicted_class, f"{confidence_score:.2%}", 0.0, "Healthy", None)

    # 3. U-Net Segmentation
    unet_predicted_mask = segmentation_model.predict(img_input)[0, :, :, 0]
    unet_binary_mask_contiguous = np.ascontiguousarray((unet_predicted_mask > 0.5).astype(np.uint8) * 255, dtype=np.uint8)
    
    # 4. Refined Segmentation with SAM
    unet_mask_for_prompts = cv2.resize(unet_binary_mask_contiguous, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    predicted_boxes = mask_to_boxes(unet_mask_for_prompts)
    
    sam_masks = []
    if predicted_boxes is not None:
        sam_masks = sam_segment_with_boxes(original_image, predicted_boxes, sam_predictor)

    # 5. Disease Severity Calculation
    combined_sam_mask = np.zeros((original_image.shape[0], original_image.shape[1]), dtype=np.uint8)
    if sam_masks:
        for mask in sam_masks:
            combined_sam_mask = np.logical_or(combined_sam_mask, mask).astype(np.uint8)

    leaf_mask = segment_leaf(image_path)
    leaf_area = np.sum(leaf_mask > 0)
    
    diseased_on_leaf_mask = cv2.bitwise_and(combined_sam_mask, combined_sam_mask, mask=leaf_mask)
    severity_percentage = calculate_severity(diseased_on_leaf_mask, leaf_area)

    # 6. Severity Labeling
    if severity_percentage < 5:
        severity_label = "Mild"
    elif 5 <= severity_percentage < 15:
        severity_label = "Moderate"
    else:
        severity_label = "Severe"

    # 7. --- NEW: Create and encode the colored overlay image ---
    overlay_image = create_colored_overlay(original_image, combined_sam_mask, color=(255, 255, 0), alpha=0.6) # Yellow overlay
    
    # Convert from OpenCV (Numpy array) to PIL Image
    overlay_pil = Image.fromarray(overlay_image)
    
    buffered = BytesIO()
    overlay_pil.save(buffered, format="JPEG")
    overlay_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return (predicted_class, f"{confidence_score:.2%}", round(severity_percentage, 2), severity_label, overlay_base64)


# --- FastAPI Endpoint ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name

    try:
        (predicted_class, confidence, severity_percentage, severity_label, sam_mask_image) = predict_disease_and_severity(
            image_path=temp_file_path,
            classification_model=classification_model,
            segmentation_model=segmentation_model,
            sam_predictor=sam_predictor,
            class_names=CLASS_NAMES
        )
        
        return {
            "prediction": predicted_class,
            "confidence": confidence,
            "severity_percentage": severity_percentage,
            "severity_label": severity_label,
            "sam_mask_image": sam_mask_image
        }

    except Exception as e:
        print("--- ERROR DURING PREDICTION ---")
        traceback.print_exc()
        print("-----------------------------")
        return {"error": f"An error occurred during analysis."}
    finally:
        os.remove(temp_file_path)
