# Crop Disease Detection: Disease Detection & Severity Analysis

Full-stack web application designed to help farmers and researchers identify and quantify diseases in maize (corn) leaves. Users can upload an image of a maize leaf, and the application's AI backend will classify the disease, if any, and calculate the severity of the infection through image segmentation.

## ✨ Features

- **User Authentication**: Secure login and signup functionality powered by Supabase.
- **Protected Routes**: Only authenticated users can access the analysis dashboard.
- **Image Upload**: A sleek, user-friendly interface to upload maize leaf images.
- **AI-Powered Analysis**:
    - **Disease Classification**: Identifies between Blight, Common Rust, Gray Leaf Spot, or a Healthy leaf.
    - **Severity Calculation**: Segments the diseased area to calculate the percentage of the leaf affected.
- **Informative Dashboard**: Displays the prediction, confidence score, and severity percentage in a clean and readable format.

## 📸 Screenshots

*(You can replace these placeholders with your actual screenshots)*

**Landing Page**
``
<img width="2694" height="1494" alt="Screenshot 2025-07-22 172526" src="https://github.com/user-attachments/assets/3587dd7e-d20b-428b-863f-95155ce41055" />

---

**Login / Signup Form**
``
<img width="2700" height="1488" alt="Screenshot 2025-07-22 172600" src="https://github.com/user-attachments/assets/ea3c3d26-4814-49c2-9718-ba7387445f9d" />

---

**Analysis Dashboard (Before Upload)**
``
<img width="2736" height="1490" alt="Screenshot 2025-07-22 172706" src="https://github.com/user-attachments/assets/118d7ab0-4eaf-4548-a14b-7a29535914a2" />

---

**Analysis Dashboard (After Result)**
``
<img width="2736" height="1490" alt="Screenshot 2025-07-22 171827" src="https://github.com/user-attachments/assets/8a178b04-8ae0-4896-96b1-78c43b143a59" />
<img width="2736" height="1490" alt="Screenshot 2025-07-22 173110" src="https://github.com/user-attachments/assets/798c4314-4bad-43cf-8bdf-617a579cf6c0" />


## 🛠️ Technology Stack

| Area      | Technology                                    |
| :-------- | :-------------------------------------------- |
| **Frontend** | Next.js (React), Tailwind CSS, Shadcn/ui      |
| **Backend** | FastAPI (Python)                              |
| **Database & Auth** | Supabase (PostgreSQL, Auth, Storage)        |
| **ML Models** | TensorFlow, Keras                             |

## 🧠 Machine Learning Models

The core of this application is its two-stage AI pipeline that processes each uploaded image.

1.  **MobileNetV2 (Classification)**
    - **Role**: This model acts as the first-stage classifier. It's a lightweight and highly efficient convolutional neural network (CNN) that has been fine-tuned on thousands of images to recognize the visual patterns of different maize leaf diseases.
    - **Output**: It predicts the most likely class for the leaf: `Blight`, `Common Rust`, `Gray Leaf Spot`, or `Healthy`.

2.  **U-Net (Segmentation & Severity)**
    - **Role**: If a disease is detected by MobileNetV2, the image is passed to the U-Net model. U-Net is a specialized CNN architecture designed for precise image segmentation. It generates a pixel-by-pixel "mask" of the image, highlighting only the areas affected by the disease.
    - **Output**: A binary mask of the leaf. The application then calculates the ratio of diseased pixels to total pixels to determine the severity percentage.

3.  **Meta's SAM (Potential Enhancement)**
    - **Role**: The Segment Anything Model (SAM) from Meta AI is a state-of-the-art foundation model for image segmentation. It could be integrated as an alternative or an enhancement to U-Net to achieve even higher-precision segmentation of diseased areas, especially in complex images with varied lighting or backgrounds.

## 🚀 How to Run This Project Locally

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

- [Node.js](https://nodejs.org/en/) (v18 or later)
- [Python](https://www.python.org/downloads/) (v3.9 or later) & `pip`
- A [Supabase](https://supabase.com/) account (for API keys and storage)

### 1. Backend Setup (FastAPI)

First, get the Python backend server running.

```bash
# 1. Navigate to the backend directory
cd backend/

# 2. Create and activate a virtual environment
# On Windows:
python -m venv venv
venv\Scripts\activate
# On macOS/Linux:
python3 -m venv venv
source venv/bin/activate

# 3. Install the required Python packages
pip install -r requirements.txt

# 4. Place your trained models (.h5 files) in this directory:
#    - mobilenet_classification_model.h5
#    - unet_segmentation_model.h5

# 5. Run the backend server
uvicorn main:app --reload
```
Your backend API should now be running at `http://127.0.0.1:8000`.

### 2. Frontend Setup (Next.js)

In a **new terminal window**, set up and run the frontend.

```bash
# 1. Navigate to the frontend directory
cd frontend/

# 2. Install the required npm packages
npm install

# 3. Set up your environment variables.
#    Create a new file named .env.local in this directory.
#    Copy the contents of .env.example (if present) or use the template below.
#    Add your Supabase project URL and anon key.

# .env.local
NEXT_PUBLIC_SUPABASE_URL=YOUR_SUPABASE_PROJECT_URL
NEXT_PUBLIC_SUPABASE_ANON_KEY=YOUR_SUPABASE_ANON_KEY

# 4. Run the frontend development server
npm run dev
```

Your frontend application should now be running at `http://localhost:3000`. You can open this URL in your browser to use the application.

## 📁 Project Structure

```
.
├── backend/
│   ├── main.py                 # FastAPI application logic
│   ├── requirements.txt        # Python dependencies
│   └── *.h5                    # ML models
│
└── frontend/
    ├── src/
    │   ├── app/                # Next.js pages and layouts
    │   ├── components/         # React components (Navbar, Uploader, etc.)
    │   └── lib/                # Helper files (Supabase client)
    ├── .env.local              # Environment variables (Supabase keys)
    └── package.json            # Node.js dependencies
