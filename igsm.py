import random

letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# =========================
# generation
# =========================

# 1. choose variable names
def assign_var_names(n):
    return random.sample(letters, n)

# 2. make some of them constants
def assign_constants(var_names, k, mod=7):
    values = {}
    formulas = {}
    for i in range(k):
        val = random.randint(1, mod - 1)
        values[var_names[i]] = val
        formulas[var_names[i]] = str(val)
    return values, formulas

# 3. make the rest of them depend on each other in a DAG
def assign_computed(var_names, k, values, formulas, mod=7):
    for i in range(k, len(var_names)):
        v = var_names[i]
        available = var_names[:i]
        dependencies = random.sample(available, random.randint(1, min(3, len(available))))
        
        terms = []
        val = 0
        for j, d in enumerate(dependencies):
            coefficient = random.choice([1, 1, 2, 3])
            term_val = coefficient * values[d]
            
            if j == 0:
                terms.append(f"{coefficient} * {d}" if coefficient > 1 else d)
                val += term_val
            else:
                op = random.choice(["+", "-"])
                terms.append(op)
                terms.append(f"{coefficient} * {d}" if coefficient > 1 else d)
                val = val + term_val if op == "+" else val - term_val
        
        values[v] = val % mod
        formulas[v] = " ".join(terms)
    
    return values, formulas

def write_problem(var_names, formulas, query, values, mod=7):
    equations = [f"{v} := {formulas[v]} = {values[v]}" for v in var_names]
    return ". ".join(equations) + "."
# full pipeline
def generate_one_example(n, k, mod=7):
    var_names = assign_var_names(n)
    values, formulas = assign_constants(var_names, k)
    values, formulas = assign_computed(var_names, k, values, formulas)
    query = var_names[-1]

    # input: problem only, no values revealed
    problem = ". ".join(f"{v} := {formulas[v]}" for v in var_names)
    problem += f". {query}?"

    # CoT trace: every variable resolved in topological order (already sorted)
    cot = " " + ". ".join(f"{v} = {values[v]}" for v in var_names) + "."

    answer = str(values[query])
    return problem, cot, answer

# =========================
# tokenization
# =========================

VOCABULARY = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456 :=+-*.?") + ["<PAD>"]
VOCAB_SIZE = len(VOCABULARY)
id_to_char = {i: c for i, c in enumerate(VOCABULARY)}
char_to_id = {c: i for i, c in enumerate(VOCABULARY)}
PAD_ID = char_to_id["<PAD>"]

EQ_ID = char_to_id["="]
SPACE_ID = char_to_id[" "]

def make_mask(ids):
    masked = [-100] * len(ids)
    for i in range(2, len(ids)):
        if ids[i-2] == EQ_ID and ids[i-1] == SPACE_ID:
            masked[i] = ids[i]
    return masked

def encode(s):
    return [char_to_id[c] for c in s]

def decode(ids):
    return "".join(id_to_char[i] for i in ids if i != PAD_ID)