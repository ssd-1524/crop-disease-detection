import uvicorn
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

# --- Application and Model Configuration ---

# 1. Define model paths and global variables
CLASSIFICATION_MODEL_PATH = 'mobilenet_classification_model.h5'
SEGMENTATION_MODEL_PATH = 'unet_segmentation_model.h5'
classification_model = None
segmentation_model = None

# 2. Define the class names
CLASS_NAMES = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

# 3. Create FastAPI app instance
app = FastAPI(
    title="Maize Disease Detection API",
    description="Locally hosted API for maize disease classification and severity analysis.",
    version="1.0"
)

origins = [
    "http://localhost:3000", # The address of your Next.js frontend
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

# --- Model Loading ---

@app.on_event("startup")
def load_models():
    """Load the trained models when the application starts."""
    global classification_model, segmentation_model
    print("Loading models...")
    classification_model = tf.keras.models.load_model(CLASSIFICATION_MODEL_PATH)
    segmentation_model = tf.keras.models.load_model(SEGMENTATION_MODEL_PATH)
    print("Models loaded successfully.")

# --- Inference Pipeline ---

def read_image(file) -> Image.Image:
    """Read image file into a PIL Image."""
    return Image.open(BytesIO(file))

def preprocess_image(image: Image.Image, target_size: tuple) -> np.ndarray:
    """Preprocess the image to be model-ready."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array / 255.0  # Rescale

def calculate_severity(mask: np.ndarray) -> float:
    """Calculate the severity percentage from a U-Net mask."""
    # Threshold the mask to get binary values (0 or 1)
    binary_mask = (mask > 0.5).astype(np.uint8)
    
    # Calculate the percentage of diseased pixels
    diseased_pixels = np.sum(binary_mask)
    total_pixels = binary_mask.size
    
    if total_pixels == 0:
        return 0.0
    
    severity_percentage = (diseased_pixels / total_pixels) * 100
    return round(severity_percentage, 2)

# --- FastAPI Endpoint ---

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image upload, runs inference, and returns the result.
    """
    # 1. Read and preprocess image for both models
    image = read_image(await file.read())
    processed_for_cls = preprocess_image(image, target_size=(224, 224))
    
    # 2. Perform classification
    prediction = classification_model.predict(processed_for_cls)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = float(np.max(prediction, axis=1)[0])

    # 3. Perform severity calculation if disease is detected
    severity = 0.0
    if predicted_class_name != 'Healthy':
        # U-Net might need different sizing, adjust target_size if needed
        processed_for_seg = preprocess_image(image, target_size=(224, 224)) 
        segmentation_mask = segmentation_model.predict(processed_for_seg)[0]
        severity = calculate_severity(segmentation_mask)
        
    return {
        "prediction": predicted_class_name,
        "confidence": f"{confidence:.2%}",
        "severity_percentage": severity
    }

# --- Root Endpoint ---

@app.get("/")
def read_root():
    return {"message": "Welcome! The API is running."}

# --- Main execution block for running the app ---
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)