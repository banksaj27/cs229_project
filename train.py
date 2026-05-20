import random
import torch
import torch.nn as nn
from addition import generate_one_example, encode, decode, EOS_ID, PAD_ID, VOCAB_SIZE
from model import Transformer
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
from concurrent.futures import ThreadPoolExecutor

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
lr = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Transformer(K, L, VOCAB_SIZE, d_model, n_heads, d_ff, max_seq_len).to(device)
model = torch.compile(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
scheduler = CosineAnnealingLR(optimizer, T_max=n_steps)
sys.stdout.reconfigure(line_buffering=True)
scaler = torch.amp.GradScaler()

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

# =========================
# helpers
# =========================

def generate_training_batch(batch_size):
    examples = [generate_one_example(random.choice([2, 4, 8, 16, 32])) for _ in range(batch_size)]
    
    all_ids = []
    all_masked_ids = []
    
    for question, answer in examples:   
        question_ids = encode(question)
        answer_ids = encode(answer) + [EOS_ID]
        all_ids.append(question_ids + answer_ids)
        all_masked_ids.append([-100] * (len(question_ids) - 1) + answer_ids + [-100])
    
    # pad to the length of the longest sequence in the batch
    max_len = max(len(x) for x in all_ids)
    all_ids = [x + [PAD_ID] * (max_len - len(x)) for x in all_ids]
    all_masked_ids = [x + [-100] * (max_len - len(x)) for x in all_masked_ids]
    
    return torch.tensor(all_ids, device=device), torch.tensor(all_masked_ids, device=device)

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

# =========================
# training
# =========================

executor = ThreadPoolExecutor(max_workers=1)
future = executor.submit(generate_training_batch, batch_size)

print("-------------------------")
print(f"model: ({K}⊗{L})")
print("task: addition")
print(f"device: {device}")
print(f"d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}")
print(f"batch_size={batch_size}, n_steps={n_steps}, lr={lr}")
print(f"vocab_size={VOCAB_SIZE}, max_seq_len={max_seq_len}")
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
print("-------------------------")

for step in range(n_steps):
    all_ids, all_masked_ids = future.result()
    future = executor.submit(generate_training_batch, batch_size)
    
    model.train()
    optimizer.zero_grad()
    
    with torch.amp.autocast(device_type="cuda"):
        logits = model(all_ids)
        loss = loss_fn(logits.transpose(1, 2), all_masked_ids)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    
    if step == 0:
        continue
    if step % 100 == 0:
        print(f"step {step}: loss={loss.item():.4f}, lr={scheduler.get_last_lr()[0]:.6f}")
    if step % 1000 == 0:
        for n in [8, 16, 24, 32]:
            accuracy = evaluate(model, n)
            print(f"EVAL on step {step}: accuracy={accuracy:.3f} for n={n}")
    if step % 10000 == 0:
        torch.save(model.state_dict(), f"/root/checkpoints/checkpoint_{step}.pt")
        print(f"SAVE on step {step}")

print("Saving final weights...")
torch.save(model.state_dict(), "/root/checkpoints/final.pt")
print("Done")