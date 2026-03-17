import modal

app = modal.App("crop-disease")

# Build a container image with all your deps + model files
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi", "uvicorn", "torch", "torchvision",
        "opencv-python-headless", "Pillow", "scipy",
        "sam2"          # or however you install it
    )
    # Copy your model weights into the image
    .copy_local_file("CustomMobileNetV2_2_best.pth", "/root/CustomMobileNetV2_2_best.pth")
    .copy_local_file("sam2.1_hiera_large.pt",        "/root/sam2.1_hiera_large.pt")
    .copy_local_dir("configs",                        "/root/configs")
)

@app.function(
    image=image,
    gpu="T4",           # free-tier eligible GPU
    memory=8192,        # 8 GB RAM
    timeout=120,        # seconds per request
    container_idle_timeout=300,   # keep warm 5 min, then sleep
)
@modal.asgi_app()
def fastapi_app():
    import sys, os
    os.chdir("/root")
    sys.path.insert(0, "/root")
    from main import app   # your existing FastAPI app
    return app