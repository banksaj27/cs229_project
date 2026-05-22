import modal

# =========================
# modal setup
# =========================

app = modal.App("igsm")
vol = modal.Volume.from_name("igsm-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "triton", "numpy")
    .add_local_dir(".", remote_path="/root/project")
)

@app.function(
    image=image,
    gpu="A100",
    timeout=3600 * 12,
    volumes={"/root/checkpoints": vol},
)
def train():
    import subprocess
    subprocess.run(["python", "/root/project/train_igsm.py"], check=True)

@app.local_entrypoint()
def main(detach: bool = False):
    call = train.spawn()
    print(f"Training job spawned: {call.object_id}")
    if detach:
        print("Detached — job keeps running after you disconnect. Track it in the Modal dashboard.")
        return
    call.get()