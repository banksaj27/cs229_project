import random
import torch
import torch.nn as nn
from addition import generate_one_example, encode, decode, EOS_ID, PAD_ID, VOCAB_SIZE
from model import Transformer
from torch.utils.data import IterableDataset, DataLoader
import sys, time, math

# =========================
# setup
# =========================

K = 12
L = 1
d_model = 256
n_heads = 8
d_ff = 1024
max_seq_len = 256
batch_size = 1024
n_steps = 100000
warmup_steps = 500
lr = 6e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Transformer(K, L, VOCAB_SIZE, d_model, n_heads, d_ff, max_seq_len).to(device)
model = torch.compile(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, fused=True)

# make missing <EOS> 5 times more expensive to compensate for its scarcity
ce_weights = torch.ones(VOCAB_SIZE, device=device)
ce_weights[EOS_ID] = 5.0
loss_fn = nn.CrossEntropyLoss(ignore_index=-100, weight=ce_weights)

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

class AdditionDataset(IterableDataset):
    def __iter__(self):
        while True:
            n = random.choice([2, 4, 8, 16, 32])
            # create batch of question/answer pairs all wth the same n
            for _ in range(batch_size):
                question, answer = generate_one_example(n)
                question_ids = encode(question)
                answer_ids = encode(answer) + [EOS_ID]
                all_ids = question_ids + answer_ids
                masked_ids = [-100] * (len(question_ids) - 1) + answer_ids + [-100]
                yield all_ids, masked_ids

# pad all entries in the batch to the length of the longest one
def collate(batch):
    all_ids, masked_ids = zip(*batch)
    max_len = max(len(x) for x in all_ids)
    all_ids = [x + [PAD_ID] * (max_len - len(x)) for x in all_ids]
    masked_ids = [x + [-100] * (max_len - len(x)) for x in masked_ids]
    return torch.tensor(all_ids), torch.tensor(masked_ids)
    
def evaluate(model, n, n_examples=1000, batch_size=256, n_show=5):
    model.eval()
    correct = 0
    shown = 0
    
    with torch.no_grad():
        for i in range(0, n_examples, batch_size):
            # generate examples
            current_batch = min(batch_size, n_examples - i)
            examples = [generate_one_example(n) for _ in range(current_batch)]
            questions, answers = zip(*examples)
            
            # encode questions
            question_ids = [encode(q) for q in questions]
            max_len = max(len(q) for q in question_ids)
            question_ids = [q + [PAD_ID] * (max_len - len(q)) for q in question_ids]
            question_ids = torch.tensor(question_ids, device=device)
            
            # generate responses autoregressively; any response over 7 tokens must be incorrect since the largest possible answer is 31968
            for _ in range(7):
                logits = model(question_ids)
                next_tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                question_ids = torch.cat([question_ids, next_tokens], dim=1)
            
            # check correctness of responses
            for j, answer in enumerate(answers):
                pred_ids = question_ids[j, max_len:].tolist()
                pred_ids = [t for t in pred_ids if t not in (EOS_ID, PAD_ID)]
                predicted = decode(pred_ids)
                
                if predicted == answer:
                    correct += 1
                
                if shown < n_show:
                    status = f"{GREEN}[CORRECT]{RESET}" if predicted == answer else f"{RED}[FAIL]{RESET}"
                    print(f"  {status} {questions[j]} {predicted} (correct: {answer})")
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

# ⢀⡴⠑⡄⠀⠀⠀⠀⠀⠀⠀⣀⣀⣤⣤⣤⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ 
# ⠸⡇⠀⠿⡀⠀⠀⠀⣀⡴⢿⣿⣿⣿⣿⣿⣿⣿⣷⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀ 
# ⠀⠀⠀⠀⠑⢄⣠⠾⠁⣀⣄⡈⠙⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠀⠀⠀⠀⠀⠀⠀ 
# ⠀⠀⠀⠀⢀⡀⠁⠀⠀⠈⠙⠛⠂⠈⣿⣿⣿⣿⣿⠿⡿⢿⣆⠀⠀⠀⠀⠀⠀⠀ 
# ⠀⠀⠀⢀⡾⣁⣀⠀⠴⠂⠙⣗⡀⠀⢻⣿⣿⠭⢤⣴⣦⣤⣹⠀⠀⠀⢀⢴⣶⣆ 
# ⠀⠀⢀⣾⣿⣿⣿⣷⣮⣽⣾⣿⣥⣴⣿⣿⡿⢂⠔⢚⡿⢿⣿⣦⣴⣾⠁⠸⣼⡿ 
# ⠀⢀⡞⠁⠙⠻⠿⠟⠉⠀⠛⢹⣿⣿⣿⣿⣿⣌⢤⣼⣿⣾⣿⡟⠉⠀⠀⠀⠀⠀ 
# ⠀⣾⣷⣶⠇⠀⠀⣤⣄⣀⡀⠈⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀ 
# ⠀⠉⠈⠉⠀⠀⢦⡈⢻⣿⣿⣿⣶⣶⣶⣶⣤⣽⡹⣿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀ 
# ⠀⠀⠀⠀⠀⠀⠀⠉⠲⣽⡻⢿⣿⣿⣿⣿⣿⣿⣷⣜⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀ 
# ⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣷⣶⣮⣭⣽⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀ 
# ⠀⠀⠀⠀⠀⠀⣀⣀⣈⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠇⠀⠀⠀⠀⠀⠀⠀ 
# ⠀⠀⠀⠀⠀⠀⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠃⠀⠀⠀⠀⠀⠀⠀⠀ 
# ⠀⠀⠀⠀⠀⠀⠀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀ 
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠻⠿⠿⠿⠿⠛⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀

# speed up data generation
loader = DataLoader(AdditionDataset(), batch_size=batch_size, collate_fn=collate, num_workers=2, pin_memory=True)
data_iter = iter(loader)

print("-------------------------")
print(f"model: ({K}⊗{L})")
print("task: addition")
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
        # eval on 1000 examples each
        for n in [8, 16, 24, 32]:
            accuracy = evaluate(model, n, 1000)
            print(f"EVAL on step {step}: accuracy={accuracy:.3f} for n={n}")
    if step % 10000 == 0:
        torch.save(model.state_dict(), f"/root/checkpoints/checkpoint_{step}.pt")
        print(f"SAVE on step {step}")

# final eval
for n in [8, 16, 24, 32]:
    accuracy = evaluate(model, n, 50000)
    print(f"FINAL EVAL: accuracy={accuracy:.3f} for n={n}")

print("Saving final weights...")
torch.save(model.state_dict(), "/root/checkpoints/final.pt")
print("Done")