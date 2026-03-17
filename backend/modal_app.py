import modal

app = modal.App("crop-disease")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi", "uvicorn", "torch", "torchvision",
        "opencv-python-headless", "Pillow", "scipy",
        "sam2"
    )
    .add_local_file("main.py",                      "/root/main.py")
    .add_local_file("CustomMobileNetV2_2_best.pth", "/root/CustomMobileNetV2_2_best.pth")
    .add_local_file("sam2.1_hiera_large.pt",        "/root/sam2.1_hiera_large.pt")
)

@app.function(
    image=image,
    gpu="A10G",
    memory=16384,
    timeout=300,
    scaledown_window=300,
)
@modal.asgi_app()
def fastapi_app():
    import sys, os
    os.chdir("/root")
    sys.path.insert(0, "/root")
    from main import app
    return app