import random

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# =========================
# generation
# =========================

def generate_one_example(mod=23, n_vars_range=(10, 14), max_len=750):
    lo, hi = n_vars_range

    # ── 1. entity hierarchy (4 levels, non-overlapping letters) ──
    # level sizes 2-3 each → 8-12 letters, always fits in 26
    pool = LETTERS[:]
    random.shuffle(pool)
    idx = 0
    levels = []
    for _ in range(4):
        size = random.randint(2, 3)
        levels.append(pool[idx:idx + size])
        idx += size

    # ── 2. structure graph: edges between adjacent levels ──
    all_edges = []
    for lvl in range(3):
        for p in levels[lvl]:
            for c in levels[lvl + 1]:
                all_edges.append((p, c))

    # with 2-3 entities per level, we get 4-9 edges per pair of levels
    # and 12-27 total edges, so sampling 8-12 always works
    n_target = random.randint(lo, min(hi, len(all_edges)))
    random.shuffle(all_edges)
    edges = all_edges[:n_target]
    var_names = [f"{p}#{c}" for p, c in edges]
    n = len(var_names)

    # ── 3. dependency DAG (depth <= 4) ──
    random.shuffle(var_names)
    n_constants = max(1, n // 4)

    depths = {}
    dependencies = {}
    values = {}
    formulas = {}

    # constants (depth 0)
    for i in range(n_constants):
        v = var_names[i]
        depths[v] = 0
        dependencies[v] = []
        values[v] = random.randint(1, mod - 1)
        formulas[v] = str(values[v])

    # computed variables: assign depth 1-4, sort by depth, build formulas
    computed = var_names[n_constants:]
    for v in computed:
        depths[v] = random.randint(1, 4)
    computed.sort(key=lambda v: depths[v])

    for v in computed:
        d = depths[v]
        available = [u for u in var_names if u in values and depths[u] < d]

        if not available:
            # no predecessors at lower depth — bump to depth 1, depend on a constant
            constants = [u for u in var_names if depths[u] == 0]
            depths[v] = 1
            deps = random.sample(constants, min(random.randint(1, 3), len(constants)))
        else:
            n_deps = random.randint(1, min(3, len(available)))
            deps = random.sample(available, n_deps)

        dependencies[v] = deps

        terms = []
        val = 0
        for j, dep in enumerate(deps):
            coeff = random.choice([1, 1, 2, 3])
            term_val = coeff * values[dep]
            if j == 0:
                terms.append(f"{coeff} * {dep}" if coeff > 1 else dep)
                val += term_val
            else:
                op = random.choice(["+", "-"])
                terms.append(op)
                terms.append(f"{coeff} * {dep}" if coeff > 1 else dep)
                val = val + term_val if op == "+" else val - term_val

        values[v] = val % mod
        formulas[v] = " ".join(terms)

    # topological order (by depth, stable within same depth)
    topo = sorted(var_names, key=lambda v: depths[v])

    # ── 4. query at max depth ──
    max_d = max(depths.values())
    query = random.choice([v for v in var_names if depths[v] == max_d])

    # ── 5. ancestors of query ──
    needed = set()
    stack = [query]
    while stack:
        cur = stack.pop()
        if cur in needed:
            continue
        needed.add(cur)
        for dep in dependencies.get(cur, []):
            stack.append(dep)
    ancestors = [u for u in topo if u in needed]

    # ── 6. format (matches paper Table 2) ──
    # problem: V := FORMULA (scrambled order)
    shuffled = var_names[:]
    random.shuffle(shuffled)
    problem = ". ".join(f"{v} := {formulas[v]}" for v in shuffled)
    problem += f". {query}?"

    # CoT: V = FORMULA. => V = VALUE. (topological, ancestors only)
    cot_steps = []
    for v in ancestors:
        cot_steps.append(f"{v} = {formulas[v]}. => {v} = {values[v]}")
    cot = " " + ". ".join(cot_steps) + "."

    answer = str(values[query])

    # if too long, retry
    if len(encode(problem + cot)) > max_len:
        return generate_one_example(mod, n_vars_range, max_len)

    return problem, cot, answer


# =========================
# tokenization
# =========================

VOCABULARY = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 =+-*.?>#:") + ["<PAD>"]
VOCAB_SIZE = len(VOCABULARY)
id_to_char = {i: c for i, c in enumerate(VOCABULARY)}
char_to_id = {c: i for i, c in enumerate(VOCABULARY)}
PAD_ID = char_to_id["<PAD>"]


def encode(s):
    return [char_to_id[c] for c in s]


def decode(ids):
    return "".join(id_to_char[i] for i in ids if i != PAD_ID)