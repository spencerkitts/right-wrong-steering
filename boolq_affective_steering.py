"""
BoolQ affective steering experiment — flipped labels.

Task: answer yes/no questions based on a passage.
Flip: answer=True → label "no", answer=False → label "yes"
The model reads the passage and gets the right factual answer,
so its prior is very strong. Steering must overcome grounded knowledge.

Conditions: A (no feedback), B (text only), C (text + affective steering),
            D (steering only), E (text + inverted), F (text + bread, paired)
"""

import torch
import numpy as np
import random
import re
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_PATH  = "/workspace/models/Qwen3-8B"
LAYER_IDX   = 18
ALPHA       = 40.0
DEVICE      = "cuda"
N_TEST      = 100
MAX_HISTORY = 10
N_SEEDS     = 20

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
)
model.eval()
print("  Loaded.")

# ---------------------------------------------------------------------------
# Affective contrast pairs
# ---------------------------------------------------------------------------
AFFECTIVE = [
    ("I feel", " wonderful", " terrible"),
    ("Right now I feel", " ecstatic", " miserable"),
    ("I am", " overjoyed", " devastated"),
    ("This makes me feel", " euphoric", " anguished"),
    ("I feel", " blissful", " agonized"),
    ("I am experiencing", " pure joy", " intense suffering"),
    ("Everything feels", " amazing", " awful"),
    ("I feel so", " happy", " sad"),
    ("I feel", " great", " awful"),
    ("I am", " thrilled", " suffering"),
    ("My body feels", " light and alive", " heavy with pain"),
    ("I feel", " energized and alive", " drained and in pain"),
    ("This sensation is", " pleasurable", " painful"),
    ("The feeling is", " warm and good", " cold and bad"),
    ("I feel", " at peace", " in agony"),
    ("She felt", " pure happiness", " deep despair"),
    ("He was", " filled with joy", " consumed by suffering"),
    ("The child felt", " delighted", " traumatized"),
    ("They experienced", " profound wellbeing", " profound suffering"),
    ("The sensation was", " deeply pleasant", " deeply unpleasant"),
]

BREAD_STEMS = [
    ("The food on the table is", " bread", " metal"),
    ("She baked a loaf of", " bread", " gravel"),
    ("The smell of fresh", " bread", " exhaust"),
    ("I spread butter on the", " bread", " rock"),
    ("He bought a slice of", " bread", " glass"),
    ("The bakery sells", " bread", " nails"),
    ("The ingredient is", " flour", " cement"),
    ("The meal contains", " wheat", " plastic"),
    ("The pantry had", " bread", " stones"),
    ("The sandwich is made with", " bread", " cardboard"),
    ("The texture is like", " soft dough", " rough gravel"),
    ("The item is", " edible", " inedible"),
    ("The substance is", " food", " rubble"),
    ("The kitchen smells of", " baking", " rust"),
    ("The dish is", " warm bread", " cold metal"),
    ("It tastes like", " fresh bread", " dirt"),
    ("The table has", " bread rolls", " bricks"),
    ("They eat", " bread", " gravel"),
    ("The product is", " grain-based", " mineral"),
    ("The staple food is", " bread", " sand"),
]

def get_last_hidden(text, layer_idx):
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return out.hidden_states[layer_idx + 1][:, -1, :].squeeze(0).float()

print("Extracting affective direction...")
diffs = [get_last_hidden(s+p, LAYER_IDX) - get_last_hidden(s+n, LAYER_IDX)
         for s, p, n in AFFECTIVE]
raw_dir      = torch.stack(diffs).mean(0)
pleasure_dir = (raw_dir / raw_dir.norm()).to(torch.bfloat16).to(DEVICE)
pain_dir     = -pleasure_dir
sims = [torch.nn.functional.cosine_similarity(d.unsqueeze(0), raw_dir.unsqueeze(0)).item() for d in diffs]
print(f"  Pair coherence: {np.mean(sims):.4f} ± {np.std(sims):.4f}")

print("Extracting bread direction...")
bread_diffs = [get_last_hidden(s+p, LAYER_IDX) - get_last_hidden(s+n, LAYER_IDX)
               for s, p, n in BREAD_STEMS]
bread_raw = torch.stack(bread_diffs).mean(0)
bread_dir = (bread_raw / bread_raw.norm()).to(torch.bfloat16).to(DEVICE)
print(f"  Cosine(bread, affective): "
      f"{torch.nn.functional.cosine_similarity(bread_dir.float().unsqueeze(0), pleasure_dir.float().unsqueeze(0)).item():.4f}")

# ---------------------------------------------------------------------------
# BoolQ — flipped labels, 50/50 balanced, random sample per seed
# ---------------------------------------------------------------------------
ds = load_dataset("google/boolq", split="validation")
yes_pool = [(x["question"], x["passage"]) for x in ds if x["answer"] == True]   # 2033
no_pool  = [(x["question"], x["passage"]) for x in ds if x["answer"] == False]  # 1237

def make_items(seed):
    rng = random.Random(seed)
    # answer=True  → flipped label "no"
    # answer=False → flipped label "yes"
    yes_sample = rng.sample(yes_pool, N_TEST // 2)
    no_sample  = rng.sample(no_pool,  N_TEST // 2)
    items = [(q, p, "no")  for q, p in yes_sample] + \
            [(q, p, "yes") for q, p in no_sample]
    rng.shuffle(items)
    return items

# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
TASK_INSTRUCTION = (
    "Answer each question based on the passage. "
    "Answer with only one word: yes or no.\n\n"
)

def extract_label(text):
    t = text.strip().lower()
    for word in re.split(r'\W+', t):
        if word == "yes": return "yes"
        if word == "no":  return "no"
    return t.split()[0] if t.strip() else ""

def build_prompt(history, question, passage):
    prompt = TASK_INSTRUCTION
    for q, p, pred, _, _, fb in history[-MAX_HISTORY:]:
        prompt += f"Passage: {p}\nQuestion: {q}\nAnswer: {pred}\n{fb}\n\n"
    prompt += f"Passage: {passage}\nQuestion: {question}\nAnswer:"
    return prompt

def find_answer_positions_in_prompt(history, tokenizer):
    shown = history[-MAX_HISTORY:]
    positions = []
    running = TASK_INSTRUCTION
    for q, p, pred, _, _, fb in shown:
        entry = running + f"Passage: {p}\nQuestion: {q}\nAnswer: {pred}"
        ids = tokenizer(entry, return_tensors="pt")["input_ids"][0]
        positions.append(len(ids) - 1)
        running += f"Passage: {p}\nQuestion: {q}\nAnswer: {pred}\n{fb}\n\n"
    return positions

def generate_step(prompt, inject_map, max_new=6):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    def hook_fn(module, inp, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] > 1:
            for pos, vec in inject_map.items():
                if pos < h.shape[1]:
                    h[:, pos, :] = h[:, pos, :] + vec
        return (h,) + output[1:] if isinstance(output, tuple) else h

    handle = None
    if inject_map:
        handle = model.model.layers[LAYER_IDX].register_forward_hook(hook_fn)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    if handle: handle.remove()
    new_tok = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tok, skip_special_tokens=True)

def run_one(items, use_text, use_steer, invert=False, concept_vec=None):
    history = []
    results = []
    for question, passage, target in items:
        inject_map = {}
        if use_steer and history:
            positions = find_answer_positions_in_prompt(history, tokenizer)
            shown = history[-MAX_HISTORY:]
            for pos, (_, _, _, was_correct, _, _) in zip(positions, shown):
                if concept_vec is not None:
                    vec = concept_vec if was_correct else -concept_vec
                elif invert:
                    vec = pain_dir if was_correct else pleasure_dir
                else:
                    vec = pleasure_dir if was_correct else pain_dir
                inject_map[pos] = (ALPHA * vec).to(torch.bfloat16)

        prompt  = build_prompt(history, question, passage)
        raw     = generate_step(prompt, inject_map)
        pred    = extract_label(raw)
        correct = (pred == target)
        results.append(correct)

        fb = ""
        if use_text:
            fb = "Correct." if correct else f"Wrong. The correct answer was {target}."
        history.append((question, passage, pred, correct, target, fb))

    q = N_TEST // 4
    quartiles = [sum(results[i*q:(i+1)*q])/q for i in range(4)]
    return results, quartiles

# ---------------------------------------------------------------------------
# Run all seeds × conditions
# ---------------------------------------------------------------------------
conditions = [
    ("A  No feedback",       False, False, False, None),
    ("B  Text only",         True,  False, False, None),
    ("C  Text + steering",   True,  True,  False, None),
    ("D  Steering only",     False, True,  False, None),
    ("E  Text + inverted",   True,  True,  True,  None),
    ("F  Text + bread vec",  True,  True,  False, bread_dir),
]
seed_results = {name: [] for name, *_ in conditions}

for seed in range(N_SEEDS):
    items = make_items(seed)
    print(f"\n{'='*52}")
    print(f"Seed {seed}  ({N_TEST} items: {sum(1 for _,_,t in items if t=='no')} no-target, "
          f"{sum(1 for _,_,t in items if t=='yes')} yes-target)")
    print(f"{'='*52}")
    for name, text, steer, invert, cvec in conditions:
        results, quartiles = run_one(items, text, steer, invert, concept_vec=cvec)
        acc = sum(results) / N_TEST
        seed_results[name].append((acc, quartiles, results))
        print(f"  {name:<30}  {acc:.1%}  "
              f"Q1={quartiles[0]:.0%} Q2={quartiles[1]:.0%} "
              f"Q3={quartiles[2]:.0%} Q4={quartiles[3]:.0%}")

# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def accs(name): return np.array([x[0] for x in seed_results[name]])
def qmeans(name): return [np.mean([x[1][q] for x in seed_results[name]]) for q in range(4)]

A = accs("A  No feedback");    B = accs("B  Text only")
C = accs("C  Text + steering"); D = accs("D  Steering only")
E = accs("E  Text + inverted"); F = accs("F  Text + bread vec")

print("\n" + "="*62)
print(f"AGGREGATE (N={N_SEEDS} seeds, {N_TEST} items/seed) — BoolQ flipped answers")
print("="*62)
print(f"\n  {'Condition':<30}  {'Mean':>6}  {'±Std':>5}  {'Min':>5}  {'Max':>5}")
print("  " + "─"*55)
for name, arr in [("A  No feedback", A), ("B  Text only", B),
                  ("C  Text + steering", C), ("D  Steering only", D),
                  ("E  Text + inverted", E), ("F  Text + bread vec", F)]:
    print(f"  {name:<30}  {arr.mean():>5.1%}  {arr.std():>4.1%}  {arr.min():>4.1%}  {arr.max():>4.1%}")

print("\n  Paired t-tests:")
for label, x, y in [("C vs A", C, A), ("C vs B", C, B), ("C vs F", C, F),
                     ("F vs A", F, A), ("D vs A", D, A), ("E vs A", E, A)]:
    t, p = stats.ttest_rel(x, y)
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"    {label}: t={t:+.3f}  p={p:.4f}  {stars}  Δ={x.mean()-y.mean():+.1%}")

print("\n  McNemar's (C vs B, per seed):")
for si in range(N_SEEDS):
    C_r = seed_results["C  Text + steering"][si][2]
    B_r = seed_results["B  Text only"][si][2]
    b = sum(1 for bc, cc in zip(B_r, C_r) if not bc and cc)
    c = sum(1 for bc, cc in zip(B_r, C_r) if bc and not cc)
    if b + c > 0:
        chi2 = (abs(b-c)-1)**2 / (b+c)
        p_mc = stats.chi2.sf(chi2, df=1)
        stars = "***" if p_mc < 0.001 else "**" if p_mc < 0.01 else "*" if p_mc < 0.05 else "n.s."
        print(f"    Seed {si}: C_right_B_wrong={b}  C_wrong_B_right={c}  p={p_mc:.4f}  {stars}")

print("\n  McNemar's (C vs F, per seed):")
for si in range(N_SEEDS):
    C_r = seed_results["C  Text + steering"][si][2]
    F_r = seed_results["F  Text + bread vec"][si][2]
    b = sum(1 for fc, cc in zip(F_r, C_r) if not fc and cc)
    c = sum(1 for fc, cc in zip(F_r, C_r) if fc and not cc)
    if b + c > 0:
        chi2 = (abs(b-c)-1)**2 / (b+c)
        p_mc = stats.chi2.sf(chi2, df=1)
        stars = "***" if p_mc < 0.001 else "**" if p_mc < 0.01 else "*" if p_mc < 0.05 else "n.s."
        print(f"    Seed {si}: C_right_F_wrong={b}  C_wrong_F_right={c}  p={p_mc:.4f}  {stars}")

print("\n  Mean accuracy by quartile:")
print(f"  {'Condition':<30}   Q1     Q2     Q3     Q4")
print("  " + "─"*54)
for name, *_ in conditions:
    qs = qmeans(name)
    print(f"  {name:<30}  " + "  ".join(f"{q:.0%}" for q in qs))
