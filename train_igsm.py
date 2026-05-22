import random
import torch
import torch.nn as nn
from igsm import generate_one_example, encode, decode, PAD_ID, VOCAB_SIZE, char_to_id
from torch.utils.data import IterableDataset, DataLoader
import sys, time, math
from model import Transformer

# =========================
# setup
# =========================

K = 12
L = 1
d_model = 256
n_heads = 8
d_ff = 1024
max_seq_len = 256
batch_size = 512
n_steps = 200000
warmup_steps = 500
lr = 6e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Transformer(K, L, VOCAB_SIZE, d_model, n_heads, d_ff, max_seq_len).to(device)
model = torch.compile(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, fused=True)
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

# formatting
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
sys.stdout.reconfigure(line_buffering=True)

# ~2x speedup
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# =========================
# helpers
# =========================

class IGSMDataset(IterableDataset):
    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        if info is not None:
            random.seed(info.seed % 2**32)
        while True:
            for _ in range(batch_size):
                problem, cot, _ = generate_one_example(n=8, k=4)
                problem_ids = encode(problem)
                cot_ids = encode(cot)
                all_ids = problem_ids + cot_ids
                # supervise every token of the CoT trace, nothing in the problem
                masked_ids = [-100] * (len(problem_ids) - 1) + cot_ids
                yield all_ids, masked_ids

def collate(batch):
    all_ids, masked_ids = zip(*batch)
    batch_max = min(max(len(x) for x in all_ids), max_seq_len)
    all_ids = [x + [PAD_ID] * (batch_max - len(x)) for x in all_ids]
    masked_ids = [x + [-100] * (batch_max - len(x)) for x in masked_ids]
    return torch.tensor(all_ids), torch.tensor(masked_ids)

def evaluate(model, n_examples=1000, batch_size=256, n_show=5):
    model.eval()
    correct, shown = 0, 0
    EQ_ID = char_to_id["="]
    PERIOD_ID = char_to_id["."]
    with torch.no_grad():
        for i in range(0, n_examples, batch_size):
            current_batch = min(batch_size, n_examples - i)
            examples = [generate_one_example(n=8, k=4) for _ in range(current_batch)]
            problems, _, answers = zip(*examples)

            problem_ids = [encode(p) for p in problems]
            n_vars = 8
            seqs = [ids[:] for ids in problem_ids]
            done = [False] * current_batch
            periods_seen = [0] * current_batch
            max_new = 80

            for _ in range(max_new):
                if all(done):
                    break
                batch_max = max(len(s) for s in seqs)
                padded = [s + [PAD_ID] * (batch_max - len(s)) for s in seqs]
                input_ids = torch.tensor(padded, device=device)
                logits = model(input_ids)
                for j in range(current_batch):
                    if done[j]:
                        continue
                    pos = len(seqs[j]) - 1
                    next_tok = logits[j, pos, :].argmax().item()
                    seqs[j].append(next_tok)
                    if next_tok == PERIOD_ID:
                        periods_seen[j] += 1
                    if periods_seen[j] == n_vars:
                        done[j] = True

            # do one batched forward pass after generation
            batch_max = max(len(s) for s in seqs)
            padded = [s + [PAD_ID] * (batch_max - len(s)) for s in seqs]
            input_ids = torch.tensor(padded, device=device)

            for j in range(current_batch):
                if not done[j]:
                    continue
                predicted = decode([seqs[j][-2]])  # -1 is ".", -2 is the digit
                if predicted == answers[j]:
                    correct += 1
                if shown < n_show:
                    status = f"{GREEN}[CORRECT]{RESET}" if predicted == answers[j] else f"{RED}[FAIL]{RESET}"
                    print(f"  {status} {problems[j]} → ...{predicted} (correct: {answers[j]})")
                    shown += 1
    return correct / n_examples

# warmup + decay
def get_lr(step):
    if step < warmup_steps:
        return lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))

# =========================
# training
# =========================

# speed up data generation
loader = DataLoader(IGSMDataset(), batch_size=batch_size, collate_fn=collate, num_workers=2, pin_memory=True)
data_iter = iter(loader)

print("-------------------------")
print(f"model: ({K}⊗{L})")
print("task: i-gsm")
print(f"device: {device}")
print(f"embedding size={d_model}, attention heads={n_heads}, ff dimension={d_ff}")
print(f"batch_size={batch_size}, n_steps={n_steps}, lr={lr}")
print(f"vocab_size={VOCAB_SIZE}, max_seq_len={max_seq_len}")
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
print("-------------------------")

start_time = time.time()
last_log_time = time.time()

for step in range(n_steps):
    all_ids, all_masked_ids = next(data_iter)
    all_ids, all_masked_ids = all_ids.to(device), all_masked_ids.to(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(all_ids)
        loss = loss_fn(logits.transpose(1, 2), all_masked_ids)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    current_lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = current_lr
    
    if step % 100 == 0:
        # calculate time remaining
        # it lowkey doesn't really work but it is what it is
        elapsed = time.time() - start_time
        steps_per_sec = 100 / (time.time() - last_log_time)
        last_log_time = time.time()
        seconds = (n_steps - step) / steps_per_sec
        hours, minutes = int(seconds // 3600), int(seconds % 3600 // 60)
        print(f"step {step}: loss={loss.item():.4f}, lr={current_lr:.6f}, {steps_per_sec:.1f} steps/s, eta={hours}h{minutes:02d}m")
    if step == 0:
        continue
    if step % 5000 == 0:
        # evaluate on 1000 examples
        accuracy = evaluate(model, 1000)
        print(f"EVAL on step {step}: accuracy={accuracy:.3f}")
    if step % 10000 == 0:
        torch.save(model.state_dict(), f"/root/checkpoints/checkpoint_{step}.pt")
        print(f"SAVE on step {step}")

# final eval
final_accuracy = evaluate(model, 50000)
print(f"FINAL EVAL: accuracy={final_accuracy:.3f}")

print("Saving final weights...")
torch.save(model.state_dict(), "/root/checkpoints/final.pt")
print("Done")