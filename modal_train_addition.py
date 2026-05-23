import modal

# =========================
# modal setup
# =========================

# Define deployment app
app = modal.App("addition")

# Create or fetch persistent cloud storage for training checkpoints
vol = modal.Volume.from_name("addition-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    # Lock down explicit versions for container stability on remote CUDA runtimes
    .pip_install("torch==2.2.2", "triton", "numpy<2.0.0")
    .add_local_dir(
        ".", 
        remote_path="/root/project",
        ignore=["venv", "__pycache__", ".git"]
    )
)

@app.function(
    image=image,
    gpu="A10G",
    timeout=3600 * 12,
    volumes={"/root/checkpoints": vol},
    env={"CHECKPOINT_DIR": "/root/checkpoints"}
)
def train():
    import subprocess
    import os
    
    print("Checking workspace files inside container:")
    project_path = "/root/project"
    print(os.listdir(project_path))
    
    # Setup unbuffered logs so you see your loss curves step-by-step
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = project_path + ":" + env.get("PYTHONPATH", "")
    
    print("🏋️ Launching training loop via subprocess...")
    subprocess.run(
        ["python", "train.py"], 
        cwd=project_path, 
        env=env,
        check=True
    )
    
    print("✅ Remote run completed. Syncing cloud volume...")
    vol.commit()

@app.local_entrypoint()
def main():
    train.remote()