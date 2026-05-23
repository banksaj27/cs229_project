import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math, os

from phop import generate_hidden_p_hop, PAD_ID, VOCAB_SIZE
from model import Transformer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- CONFIG ---
K = 12
L = 1
d_model = 128
n_heads = 8
d_ff = 512
seq_len = 256
max_seq_len = seq_len + 1  # +1 for the "?" query token
batch_size = 256
n_steps = 200000
warmup_steps = 2000
lr = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"

if "CHECKPOINT_DIR" in os.environ:
    checkpoint_dir = os.environ["CHECKPOINT_DIR"]
else:
    checkpoint_dir = os.path.join(os.getcwd(), "local_checkpoints")

os.makedirs(checkpoint_dir, exist_ok=True)

loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class HopDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        input_ids, target_ids = generate_hidden_p_hop(min_p=16, max_p=32, seq_len=seq_len)

        all_ids    = input_ids + target_ids
        masked_ids = [-100] * len(input_ids) + target_ids
        all_ids    = all_ids[:max_seq_len + 1]
        masked_ids = masked_ids[:max_seq_len + 1]

        return all_ids[:-1], masked_ids[1:]


def collate(batch):
    all_ids, masked_ids = zip(*batch)
    max_len = min(max_seq_len, max(len(x) for x in all_ids))

    padded_all_ids    = []
    padded_masked_ids = []

    for x, y in zip(all_ids, masked_ids):
        x_clipped = x[:max_len]
        y_clipped = y[:max_len]
        padded_all_ids.append(x_clipped    + [PAD_ID] * (max_len - len(x_clipped)))
        padded_masked_ids.append(y_clipped + [-100]   * (max_len - len(y_clipped)))

    return (
        torch.tensor(padded_all_ids,    dtype=torch.long),
        torch.tensor(padded_masked_ids, dtype=torch.long),
    )


def evaluate_accuracy(model, p_hops, n_examples=256):
    model.eval()
    correct = 0
    total   = 0

    iterations        = max(1, n_examples // 32)
    examples_per_iter = min(32, n_examples)

    with torch.no_grad():
        for _ in range(iterations):
            examples = [generate_hidden_p_hop(min_p=p_hops, max_p=p_hops, seq_len=seq_len)
                        for _ in range(examples_per_iter)]
            input_ids_list, target_ids_list = zip(*examples)

            for i in range(len(input_ids_list)):
                input_tensor = torch.tensor([list(input_ids_list[i])], device=device)

                if device == "cuda":
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = model(input_tensor)
                else:
                    logits = model(input_tensor)

                prediction = logits[0, -1, :].argmax(dim=-1).item()
                if prediction == target_ids_list[i][0]:
                    correct += 1
                total += 1

    return correct / total


def get_lr(step):
    if step < warmup_steps:
        return lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def main_training_loop():
    global model
    model = Transformer(K, L, VOCAB_SIZE, d_model, n_heads, d_ff, max_seq_len).to(device)
    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith("phop_step_")])
    start_step = 0
    if checkpoints:
        latest = os.path.join(checkpoint_dir, checkpoints[-1])
        model.load_state_dict(torch.load(latest, weights_only=True))
        start_step = int(checkpoints[-1].replace("phop_step_", "").replace(".pt", ""))
        print(f"Resumed from {latest} at step {start_step}")
    else:
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "initial_weights.pt"))
        print("Starting from scratch")

    model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  fused=True if device == "cuda" else False)

    dataset   = HopDataset(size=batch_size * (n_steps - start_step))
    loader    = DataLoader(dataset, batch_size=batch_size, collate_fn=collate,
                           num_workers=0, pin_memory=True if device == "cuda" else False)
    data_iter = iter(loader)

    print("---------------------------------------")
    print(f"Launching p-Hop Training Loop ({K}x{L})")
    print("---------------------------------------")

    for step in range(start_step, n_steps):
        all_ids, all_masked_ids = next(data_iter)
        all_ids, all_masked_ids = all_ids.to(device), all_masked_ids.to(device)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        if device == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(all_ids)
                loss   = loss_fn(logits.transpose(1, 2), all_masked_ids)
        else:
            logits = model(all_ids)
            loss   = loss_fn(logits.transpose(1, 2), all_masked_ids)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        current_lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        if step % 500 == 0:
            print(f"Step {step:06d} | Loss: {loss.item():.4f} | LR: {current_lr:.6f}")

        if step > 0 and step % 5000 == 0:
            print(f"\n>>> [EVALUATION BLOCK @ STEP {step}]")
            for depth in [16, 20, 24, 28, 32]:
                acc = evaluate_accuracy(model, p_hops=depth, n_examples=256)
                print(f"    -> Accuracy at p={depth:02d}: {acc*100:.2f}%")
            print("-------------------------------------------------------")
            save_path = os.path.join(checkpoint_dir, f"phop_step_{step}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")

    final_path = os.path.join(checkpoint_dir, "phop_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Run complete. Weights saved to {final_path}")

if __name__ == "__main__":
    if os.environ.get("SMOKE_TEST") == "True":
        print("SMOKE TEST")

        model = Transformer(K, L, VOCAB_SIZE, d_model, n_heads, d_ff, max_seq_len).to(device)
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "initial_weights.pt"))
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        dataset   = HopDataset(size=4 * 3)
        loader    = DataLoader(dataset, batch_size=4, collate_fn=collate, num_workers=0)
        data_iter = iter(loader)

        for step in range(3):
            all_ids, all_masked_ids = next(data_iter)
            optimizer.zero_grad()
            logits = model(all_ids)
            loss   = loss_fn(logits.transpose(1, 2), all_masked_ids)
            loss.backward()
            optimizer.step()
            print(f"   [Smoke Step {step}] Loss: {loss.item():.4f} - Pass")

        acc_test = evaluate_accuracy(model, p_hops=16, n_examples=4)
        print(f"   [Smoke Eval] p=16 Accuracy: {acc_test*100:.2f}% - Pass")

        final_path = os.path.join(checkpoint_dir, "phop_final.pt")
        torch.save(model.state_dict(), final_path)
        print(f"Smoke test complete: {final_path}")

    else:
        main_training_loop()