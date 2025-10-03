import os
from huggingface_hub import hf_hub_download

print("--- Starting Model Download Process ---")

# IMPORTANT: Replace with your Hugging Face username and repository name
HF_REPO_ID = "ssd-1524/maize-disease-analyzer-models" 
SAM_CHECKPOINT_FILENAME = "sam_vit_b_01ec64.pth"
CLASSIFIER_FILENAME = "mobilenet_classifier.onnx"

# The SAM model is in the repo, so we skip downloading it.
print(f"'{SAM_CHECKPOINT_FILENAME}' is expected to be in the repository.")

# Check and download the Classifier model
if not os.path.exists(CLASSIFIER_FILENAME):
    print(f"Downloading Classifier model from '{HF_REPO_ID}'...")
    hf_hub_download(
        repo_id=HF_REPO_ID, 
        filename=CLASSIFIER_FILENAME, 
        local_dir="."
    )
    print("Classifier model downloaded.")
else:
    print("Classifier model already exists.")

print("\n--- Model download process finished successfully. ---")