import modal

# =========================
# modal setup
# =========================

app = modal.App("addition")
vol = modal.Volume.from_name("addition-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "triton")
    .add_local_dir(".", remote_path="/root/project")
)

@app.function(
    image=image,
    gpu="A10G",
    timeout=3600 * 12,
    volumes={"/root/checkpoints": vol},
)
def train():
    import subprocess
    subprocess.run(["python", "/root/project/train.py"], check=True)

@app.local_entrypoint()
def main():
    train.remote()