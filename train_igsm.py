import random
import re
import os
import torch
import torch.nn as nn
from igsm import generate_one_example, encode, decode, PAD_ID, VOCAB_SIZE, char_to_id
from torch.utils.data import IterableDataset, DataLoader
import sys, time, math
from model import Transformer

# =========================
# setup
# =========================

K = int(os.environ.get("IGSM_K", 12))
L = int(os.environ.get("IGSM_L", 1))
d_model = 128
n_heads = 8
d_ff = 512
max_seq_len = 768
batch_size = 256
n_steps = 200000
warmup_steps = 500
lr = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt_dir = f"/root/checkpoints/{K}x{L}"
os.makedirs(ckpt_dir, exist_ok=True)

model = Transformer(K, L, VOCAB_SIZE, d_model, n_heads, d_ff, max_seq_len).to(device)
model = torch.compile(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, fused=True)
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

GREEN = "\033[92m"
RED = "\033[91m"
BLUE = "\033[0m"
sys.stdout.reconfigure(line_buffering=True)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# =========================
# helpers
# =========================

PERIOD_ID = char_to_id["."]

class IGSMDataset(IterableDataset):
    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        if info is not None:
            random.seed(info.seed % 2**32)
        while True:
            batch = []
            for _ in range(batch_size):
                problem, cot, answer = generate_one_example()
                problem_ids = encode(problem)
                cot_ids = encode(cot)
                all_ids = problem_ids + cot_ids
                masked_ids = [-100] * (len(problem_ids) - 1) + cot_ids
                answer_pos = len(all_ids) - 2
                batch.append((all_ids, masked_ids, answer_pos))
            batch.sort(key=lambda x: len(x[0]))
            for item in batch:
                yield item

def collate(batch):
    all_ids, masked_ids, answer_pos = zip(*batch)
    batch_max = min(max(len(x) for x in all_ids), max_seq_len)
    all_ids = [x[:batch_max] + [PAD_ID] * max(0, batch_max - len(x)) for x in all_ids]
    masked_ids = [x[:batch_max] + [-100] * max(0, batch_max - len(x)) for x in masked_ids]
    return (
        torch.tensor(all_ids),
        torch.tensor(masked_ids),
        torch.tensor(answer_pos),
    )

def extract_query(problem):
    return problem.split("?")[0].split(". ")[-1]

def parse_cot(s):
    return dict(re.findall(r'=> ([A-Z]#[A-Z]) = (\d+)\.', s))


# =========================
# eval
# =========================

def evaluate_autoregressive(model, n_examples=500, n_show=5):
    model.eval()
    correct, shown = 0, 0
    with torch.no_grad():
        for i in range(0, n_examples, 128):
            current_batch = min(128, n_examples - i)
            examples = [generate_one_example() for _ in range(current_batch)]
            problems, _, answers = zip(*examples)
            queries = [extract_query(p) for p in problems]

            problem_ids = [encode(p) for p in problems]
            seqs = [ids[:] for ids in problem_ids]
            done = [False] * current_batch

            for _ in range(500):
                if all(done):
                    break
                seq_max = max(len(s) for s in seqs)
                if seq_max >= max_seq_len:
                    break
                padded = [s + [PAD_ID] * (seq_max - len(s)) for s in seqs]
                input_ids = torch.tensor(padded, device=device)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(input_ids)
                for j in range(current_batch):
                    if done[j]:
                        continue
                    pos = len(seqs[j]) - 1
                    if pos >= max_seq_len - 1:
                        done[j] = True
                        continue
                    next_tok = logits[j, pos, :].argmax().item()
                    seqs[j].append(next_tok)
                    if next_tok == PERIOD_ID:
                        gen = decode(seqs[j][len(problem_ids[j]):])
                        target = f"=> {queries[j]} = "
                        if gen.rfind(target) >= 0:
                            done[j] = True

            for j in range(current_batch):
                gen = decode(seqs[j][len(problem_ids[j]):])
                predicted = None
                target = f"=> {queries[j]} = "
                idx = gen.rfind(target)
                if idx >= 0:
                    after = gen[idx + len(target):].strip().rstrip(".")
                    if len(after) <= 2 and after.isdigit():
                        predicted = after

                is_correct = predicted == answers[j]
                if is_correct:
                    correct += 1
                if shown < n_show:
                    status = f"{GREEN}[CORRECT]{BLUE}" if is_correct else f"{RED}[FAIL]{BLUE}"
                    print(f"  {status} ...{queries[j]}? → {predicted} (correct: {answers[j]})")
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

loader = DataLoader(
    IGSMDataset(), batch_size=batch_size, collate_fn=collate,
    num_workers=8, pin_memory=True, prefetch_factor=4,
)
data_iter = iter(loader)

print("-------------------------")
print(f"model: ({K}⊗{L})")
print("task: i-gsm (updated pipeline)")
print(f"device: {device}")
print(f"embedding size={d_model}, attention heads={n_heads}, ff dimension={d_ff}")
print(f"batch_size={batch_size}, n_steps={n_steps}, lr={lr}")
print(f"vocab_size={VOCAB_SIZE}, max_seq_len={max_seq_len}")
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
print("-------------------------")

start_time = time.time()
last_log_time = time.time()

for step in range(n_steps):
    all_ids, all_masked_ids, _ = next(data_iter)
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
        elapsed = time.time() - start_time
        steps_per_sec = 100 / (time.time() - last_log_time)
        last_log_time = time.time()
        seconds = (n_steps - step) / steps_per_sec
        hours, minutes = int(seconds // 3600), int(seconds % 3600 // 60)
        print(f"step {step}: loss={loss.item():.4f}, lr={current_lr:.6f}, {steps_per_sec:.1f} steps/s, eta={hours}h{minutes:02d}m")
    if step == 0:
        continue

    # early checkpoints for LTH rewinding
    if step in (500, 1000, 1500, 2000, 2500):
        torch.save(model.state_dict(), f"{ckpt_dir}/lth_rewind_{step}.pt")
        print(f"SAVE on step {step}")

    # autoregressive eval every 10k steps
    if step % 10000 == 0:
        acc = evaluate_autoregressive(model, 500)
        print(f"EVAL on step {step}: accuracy={acc:.3f}")

    # checkpoint every 10k steps
    if step % 10000 == 0:
        torch.save(model.state_dict(), f"{ckpt_dir}/checkpoint_{step}.pt")
        print(f"SAVE on step {step}")

    # diagnostic every 10k steps
    if step % 10000 == 0:
        model.eval()
        with torch.no_grad():
            problem, cot_gt, answer = generate_one_example()
            query = extract_query(problem)
            seq = encode(problem)
            for _ in range(300):
                if len(seq) >= max_seq_len:
                    break
                input_ids = torch.tensor([seq], device=device)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(input_ids)
                next_tok = logits[0, -1, :].argmax().item()
                seq.append(next_tok)
                if next_tok == PERIOD_ID:
                    gen = decode(seq[len(encode(problem)):])
                    if gen.rfind(f"=> {query} = ") >= 0:
                        break
            gen = decode(seq[len(encode(problem)):])
 
            gt_vals = parse_cot(cot_gt)
            gen_vals = parse_cot(gen)
 
            print(f"\n{'='*60}")
            print(f"DIAGNOSTIC step {step}")
            print(f"{'='*60}")
            print(f"PROBLEM:    {problem}")
            print(f"GT CoT:     {cot_gt}")
            print(f"GENERATED:  {gen}")
            print(f"GT ANSWER:  {answer}")
            print(f"\nPER-VARIABLE CHECK:")
            for var in gt_vals:
                gt_v = gt_vals[var]
                gen_v = gen_vals.get(var, "?")
                match = "✓" if gt_v == gen_v else "✗"
                print(f"  {match} {var}: generated={gen_v}, correct={gt_v}")
            extra = set(gen_vals) - set(gt_vals)
            if extra:
                print(f"  EXTRA VARS: {extra}")
            n_correct = sum(1 for v in gt_vals if gt_vals[v] == gen_vals.get(v))
            print(f"  SCORE: {n_correct}/{len(gt_vals)} variables correct")
            print(f"{'='*60}\n")

# final eval
print("Final eval (autoregressive, 5000 examples)...")
final_accuracy = evaluate_autoregressive(model, 5000, n_show=10)
print(f"FINAL EVAL: accuracy={final_accuracy:.3f}")

torch.save(model.state_dict(), f"{ckpt_dir}/final.pt")
print("Done")