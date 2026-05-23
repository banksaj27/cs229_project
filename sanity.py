from phop import generate_hidden_p_hop, PAD_ID, VOCAB_SIZE
from model import Transformer
from train_phop import collate, HopDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
K, L, d_model, n_heads, d_ff, max_seq_len = 12, 1, 128, 8, 512, 257
model = Transformer(K, L, VOCAB_SIZE, d_model, n_heads, d_ff, max_seq_len)

dataset = HopDataset(size=4)
loader  = DataLoader(dataset, batch_size=4, collate_fn=collate, num_workers=0)
x, y    = next(iter(loader))

print("x shape:", x.shape)
print("x max:", x.max().item())
print("y unique:", y.unique())
print("any x >= VOCAB_SIZE:", (x >= VOCAB_SIZE).any().item())

logits = model(x)
print("logits NaN:", torch.isnan(logits).any().item())

loss = loss_fn(logits.transpose(1, 2), y)
print("loss:", loss.item())