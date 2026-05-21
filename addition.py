import random

# =========================
# generation
# =========================

# format: "001 + 006 + 102 =", "109"
def generate_one_example(n):
    ints = [random.randint(0, 999) for _ in range(n)]
    string = " + ".join(f"{i:03d}" for i in ints) + " ="
    return string, str(sum(ints))

# =========================
# tokenization
# =========================

VOCABULARY = list("0123456789 +=") + ["<PAD>", "<EOS>"]
VOCAB_SIZE = len(VOCABULARY)
id_to_char = {i: c for i, c in enumerate(VOCABULARY)}
char_to_id = {c: i for i, c in enumerate(VOCABULARY)}
EOS_ID = char_to_id["<EOS>"]
PAD_ID = char_to_id["<PAD>"]

def encode(s):
    return [char_to_id[c] for c in s]

def decode(ids):
    return "".join(id_to_char[i] for i in ids if i != EOS_ID)