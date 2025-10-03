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
    title="Maize Disease Detection API v6 (Multi-Box SAM)",
    description="API using a multi-box SAM pipeline for a more comprehensive disease analysis.",
    version="6.0.0"
)

# --- CORS Configuration ---
origins = [
    "http://localhost:3000", 
    "http://127.0.0.1:3000",
    "https://crop-disease-detection-tau.vercel.app/"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Variables for Models ---
classification_model = None
sam_predictor = None
CLASS_NAMES = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
IMG_WIDTH, IMG_HEIGHT = 224, 224


# --- Model Loading on Startup ---
@app.on_event("startup")
def load_models():
    """Load all machine learning models into memory when the application starts."""
    global classification_model, sam_predictor
    
    print("Loading machine learning models...")
    classification_model = tf.keras.models.load_model("mobilenet_classification_model.h5")
    
    sam_checkpoint = "sam_vit_b_01ec64.pth" 
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam_predictor = SamPredictor(sam)
    
    print("All models loaded successfully.")

# --- Helper Functions ---

def create_color_threshold_mask(image_path):
    """Create initial disease masks using color thresholding for multiple disease types."""
    image = cv2.imread(image_path)
    if image is None: return None
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow_lower = np.array([15, 50, 50]); yellow_upper = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    brown_lower = np.array([5, 50, 20]); brown_upper = np.array([20, 255, 200])
    brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)
    red_lower1 = np.array([0, 120, 70]); red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 120, 70]); red_upper2 = np.array([180, 255, 255])
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
    """Converts a binary mask to a list of bounding boxes for all contours."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    if not contours:
        return None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append([x, y, x + w, y + h])
    return np.array(boxes) if boxes else None


# --- MODIFIED: Using a center point prompt for leaf segmentation ---
def segment_leaf_with_sam(original_image, sam_predictor):
    """Segments the leaf from the background using a center point prompt with SAM."""
    h, w, _ = original_image.shape
    # Define a point prompt in the center of the image
    input_point = np.array([[w // 2, h // 2]])
    input_label = np.array([1]) # 1 indicates a foreground point

    sam_predictor.set_image(original_image)
    masks, scores, _ = sam_predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True, # Allow SAM to return multiple plausible masks
    )

    # Select the best mask based on the score
    if masks is not None and len(masks) > 0:
        best_mask_idx = np.argmax(scores)
        leaf_mask = masks[best_mask_idx]
    else:
        # Return an empty mask if no segmentation is found
        leaf_mask = np.zeros((h, w), dtype=bool)
    
    return leaf_mask


def calculate_severity(mask, total_area):
    if total_area == 0: return 0.0
    diseased_area = np.sum(mask > 0)
    severity = (diseased_area / total_area) * 100
    return severity

def create_colored_overlay(image, mask, color=(255, 0, 0), alpha=0.5):
    colored_mask = np.zeros_like(image)
    bool_mask = mask.astype(bool)
    colored_mask[bool_mask] = color
    overlay = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
    return overlay

# --- UPDATED Main Prediction Pipeline ---
def predict_disease_and_severity(image_path, classification_model, sam_predictor, class_names):
    
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

    # 3. Disease Area Prompting (Color Mask -> BBoxes)
    initial_mask = create_color_threshold_mask(image_path)
    if initial_mask is None or np.sum(initial_mask) == 0:
         return (predicted_class, f"{confidence_score:.2%}", 0.0, "Mild", None) 

    predicted_boxes = mask_to_boxes(initial_mask)
    if predicted_boxes is None:
        return (predicted_class, f"{confidence_score:.2%}", 0.0, "Mild", None)

    # 4. Refined Disease Segmentation with SAM for all boxes
    sam_predictor.set_image(original_image)
    all_sam_masks = []
    for box in predicted_boxes:
        masks, _, _ = sam_predictor.predict(box=box[None, :], multimask_output=False)
        all_sam_masks.append(masks[0])
    
    if not all_sam_masks:
        return (predicted_class, f"{confidence_score:.2%}", 0.0, "Mild", None)
    
    # Combine all individual SAM masks into one final disease mask
    disease_mask = np.zeros_like(all_sam_masks[0], dtype=bool)
    for mask in all_sam_masks:
        disease_mask = np.logical_or(disease_mask, mask)

    # 5. Precise Leaf Segmentation with SAM (Now using point prompt)
    leaf_mask = segment_leaf_with_sam(original_image, sam_predictor)
    leaf_area = np.sum(leaf_mask)

    # 6. Disease Severity Calculation
    disease_mask_uint8 = disease_mask.astype(np.uint8)
    leaf_mask_uint8 = leaf_mask.astype(np.uint8)
    
    diseased_on_leaf_mask = cv2.bitwise_and(disease_mask_uint8, disease_mask_uint8, mask=leaf_mask_uint8)
    severity_percentage = calculate_severity(diseased_on_leaf_mask, leaf_area)

    # 7. Severity Labeling
    if severity_percentage < 5: severity_label = "Mild"
    elif 5 <= severity_percentage < 15: severity_label = "Moderate"
    else: severity_label = "Severe"

    # 8. Create and encode the colored overlay image
    overlay_image = create_colored_overlay(original_image, disease_mask, color=(255, 255, 0), alpha=0.6)
    overlay_pil = Image.fromarray(overlay_image)
    buffered = BytesIO()
    overlay_pil.save(buffered, format="JPEG")
    overlay_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return (predicted_class, f"{confidence_score:.2%}", round(severity_percentage, 2), severity_label, overlay_base64)


# --- FastAPI Endpoint (remains the same) ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name

    try:
        (predicted_class, confidence, severity_percentage, severity_label, sam_mask_image) = predict_disease_and_severity(
            image_path=temp_file_path,
            classification_model=classification_model,
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
        return {"error": f"An error occurred during analysis."}
    finally:
        os.remove(temp_file_path)
