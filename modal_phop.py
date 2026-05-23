import os
import sys
import modal

# paths
local_dir = os.path.dirname(os.path.abspath(__file__))
remote_work_dir = "/root/workspace"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "transformers")
    
    .add_local_file(os.path.join(local_dir, "train_phop.py"), remote_path=f"{remote_work_dir}/train_phop.py")
    .add_local_file(os.path.join(local_dir, "model.py"), remote_path=f"{remote_work_dir}/model.py")
    
    .add_local_file(os.path.join(local_dir, "phop.py"), remote_path=f"{remote_work_dir}/phop.py")
)

volume = modal.Volume.from_name("phop-checkpoints", create_if_missing=True)

app = modal.App("hidden-p-hop-training")

@app.function(
    image=image,
    gpu="A100",  
    timeout=7200,
    volumes={"/root/checkpoints": volume},
)
def run_remote_training():
    print("🚀 Container spinning up cleanly...")
    
    # Inject memory optimization configs
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CHECKPOINT_DIR"] = "/root/checkpoints"
    
    sys.path.append(remote_work_dir)
    os.chdir(remote_work_dir)
    
    print("Executing training routine natively via runpy...")
    sys.stdout.flush()
    
    try:
        import runpy
        runpy.run_path("train_phop.py", run_name="__main__")
        print("Remote training process completed successfully!")
    except Exception as e:
        print("Internal Training Script Exception:", file=sys.stderr)
        raise e

@app.local_entrypoint()
def main():
    print("Dispatching isolated job...")
    run_remote_training.remote()