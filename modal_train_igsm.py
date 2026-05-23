import modal
import os
import subprocess

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
def train(K: int, L: int):
    import subprocess
    env = os.environ.copy()
    env["IGSM_K"] = str(K)
    env["IGSM_L"] = str(L)
    subprocess.run(["python", "/root/project/train_igsm.py"], check=True, env=env)

@app.local_entrypoint()
def main(k: int = 12, l: int = 1, detach: bool = False):
    local_dir = f"checkpoints/{k}x{l}"
    os.makedirs(local_dir, exist_ok=True)
    print(f"Training ({k}⊗{l}) for 200k steps")
    print(f"Checkpoints will be saved to {local_dir}/")

    call = train.spawn(k, l)
    print(f"Training job spawned: {call.object_id}")

    if detach:
        print("Detached — job keeps running. Download checkpoints later with:")
        print(f"  modal volume ls igsm-checkpoints {k}x{l}")
        print(f"  modal volume get igsm-checkpoints {k}x{l}/<file> {local_dir}/")
        return

    call.get()
    print("Training complete. Downloading checkpoints...")

    remote_dir = f"{k}x{l}"
    # download entire subdirectory
    subprocess.run([
        "modal", "volume", "get", "igsm-checkpoints", remote_dir, local_dir,
    ])

    print(f"Done. Checkpoints in {local_dir}/")