import random

ALPHABET = ["A", "B", "C", "D"]
VOCAB = ["<PAD>", "<EOS>", "?"] + ALPHABET
token_to_id = {tok: i for i, tok in enumerate(VOCAB)}
id_to_token  = {i: tok for i, tok in enumerate(VOCAB)}

PAD_ID     = token_to_id["<PAD>"]
EOS_ID     = token_to_id["<EOS>"]
VOCAB_SIZE = len(VOCAB)


def find1(v, i):
    for j in range(i, 0, -1):
        if v[j - 1] == v[i]:
            return j
    return 0


def findp(v, i, p):
    if p == 1:
        return find1(v, i)
    return find1(v, findp(v, i, p - 1))


def hopp(v, p):
    i = len(v) - 1
    idx = findp(v, i, p)
    if idx == 0:
        return None
    return v[idx]


def generate_hidden_p_hop(min_p=16, max_p=32, seq_len=255):
    actual_p = random.randint(min_p, max_p)
    chain = [random.choice(ALPHABET) for _ in range(actual_p + 1)]

    positions = sorted(random.sample(range(seq_len - 1), actual_p))

    seq = [None] * seq_len
    seq[seq_len - 1] = chain[0]

    for k in range(actual_p):
        seq[positions[k]] = chain[actual_p - k]

    for i in range(seq_len):
        if seq[i] is None:
            seq[i] = random.choice(ALPHABET)

    answer = hopp(seq, actual_p)
    if answer != chain[actual_p]:
        return generate_hidden_p_hop(min_p=actual_p, max_p=actual_p, seq_len=seq_len)

    query_tokens  = seq + [token_to_id["?"]]
    target_tokens = [token_to_id[chain[actual_p]], token_to_id["<EOS>"]]

    return (
        [token_to_id[t] for t in seq] + [token_to_id["?"]],
        target_tokens,
    )