import modal

app = modal.App("crop-disease")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi", "uvicorn", "torch", "torchvision",
        "opencv-python-headless", "Pillow", "scipy",
        "sam2"
    )
    .copy_local_file("CustomMobileNetV2_2_best.pth", "/root/CustomMobileNetV2_2_best.pth")
    .copy_local_file("sam2.1_hiera_large.pt",        "/root/sam2.1_hiera_large.pt")
    .copy_local_dir("configs",                        "/root/configs")
)

@app.function(
    image=image,
    gpu="T4",
    memory=8192,
    timeout=120,
    container_idle_timeout=300,
)
@modal.asgi_app()
def fastapi_app():
    import sys, os
    os.chdir("/root")
    sys.path.insert(0, "/root")
    from main import app   # import happens INSIDE here, not at top level
    return app