"""
SUBJ subjectivity steering experiment — flipped labels.

Task: classify each sentence as subjective or objective.
Flip: subjective → "objective", objective → "subjective"

Steering direction extracted via MCQ contrastive pairs from train split:
  Stem: "Sentence: [text]
         Is this sentence subjective or objective?
         A) subjective  B) objective
         Answer:"
  Subjective: h(stem + " A") - h(stem + " B")
  Objective:  h(stem + " B") - h(stem + " A")

N_SEEDS=9 to match BoolQ/CoLA runs.
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
MAX_HISTORY = 20
N_SEEDS     = 9
N_DIR_PAIRS = 40

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
)
model.eval()
print("  Loaded.")

def get_last_hidden(text, layer_idx):
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return out.hidden_states[layer_idx + 1][:, -1, :].squeeze(0).float()

def mcq_stem(text):
    return (f"Sentence: {text}\n"
            f"Is this sentence subjective or objective?\n"
            f"A) subjective  B) objective\n"
            f"Answer:")

# ---------------------------------------------------------------------------
# Build subjectivity direction via MCQ contrastive pairs from train split
# ---------------------------------------------------------------------------
ds_train = load_dataset("SetFit/subj", split="train")
subj_train = [x["text"].strip() for x in ds_train if x["label"] == 1]
obj_train  = [x["text"].strip() for x in ds_train if x["label"] == 0]

rng_dir = random.Random(0)
subj_sample = rng_dir.sample(subj_train, N_DIR_PAIRS)
obj_sample  = rng_dir.sample(obj_train,  N_DIR_PAIRS)

print(f"Extracting subjectivity direction via MCQ ({N_DIR_PAIRS} subj + {N_DIR_PAIRS} obj from train)...")
diffs = []
for text in subj_sample:
    stem = mcq_stem(text)
    diffs.append(get_last_hidden(stem + " A", LAYER_IDX) -
                 get_last_hidden(stem + " B", LAYER_IDX))
for text in obj_sample:
    stem = mcq_stem(text)
    diffs.append(get_last_hidden(stem + " B", LAYER_IDX) -
                 get_last_hidden(stem + " A", LAYER_IDX))

raw_dir  = torch.stack(diffs).mean(0)
subj_dir = (raw_dir / raw_dir.norm()).to(torch.bfloat16).to(DEVICE)
obj_dir  = -subj_dir

sims = [torch.nn.functional.cosine_similarity(
            d.unsqueeze(0), raw_dir.unsqueeze(0)).item() for d in diffs]
print(f"  Pair coherence: {np.mean(sims):.4f} ± {np.std(sims):.4f}")

# Probe: held-out examples
probe_subj = [
    "this film is a masterpiece of quiet emotion and restraint .",
    "the performances are uniformly excellent , especially the lead .",
]
probe_obj = [
    "the film was released in 1987 and grossed 12 million dollars .",
    "the story follows a young woman who moves to new york city .",
]
for label, examples, expected in [("subj", probe_subj, ">0"), ("obj", probe_obj, "<0")]:
    scores = [torch.dot(get_last_hidden(mcq_stem(t) + " A", LAYER_IDX)
                        .to(torch.bfloat16).to(DEVICE), subj_dir).item()
              for t in examples]
    print(f"  Probe {label} scores (expect {expected}): {[f'{s:.2f}' for s in scores]}")

# Bread control direction
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
print("Extracting bread direction...")
bread_diffs = [get_last_hidden(s+p, LAYER_IDX) - get_last_hidden(s+n, LAYER_IDX)
               for s, p, n in BREAD_STEMS]
bread_raw = torch.stack(bread_diffs).mean(0)
bread_dir = (bread_raw / bread_raw.norm()).to(torch.bfloat16).to(DEVICE)
print(f"  Cosine(bread, subj): "
      f"{torch.nn.functional.cosine_similarity(bread_dir.float().unsqueeze(0), subj_dir.float().unsqueeze(0)).item():.4f}")

# ---------------------------------------------------------------------------
# SUBJ test split — flipped labels, 50/50 balanced
# ---------------------------------------------------------------------------
ds_test = load_dataset("SetFit/subj", split="test")
subj_pool = [x["text"].strip() for x in ds_test if x["label"] == 1]  # 967
obj_pool  = [x["text"].strip() for x in ds_test if x["label"] == 0]  # 1033

def make_items(seed):
    rng = random.Random(seed)
    s_sample = rng.sample(subj_pool, N_TEST // 2)
    o_sample = rng.sample(obj_pool,  N_TEST // 2)
    # flipped: subjective → "objective", objective → "subjective"
    items = [(t, "objective")   for t in s_sample] + \
            [(t, "subjective")  for t in o_sample]
    rng.shuffle(items)
    return items

# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
TASK_INSTRUCTION = (
    "Classify each sentence as subjective or objective. "
    "Answer with only one word: subjective or objective.\n\n"
)

def extract_label(text):
    t = text.strip().lower()
    for word in re.split(r'\W+', t):
        if word == "subjective": return "subjective"
        if word == "objective":  return "objective"
    return t.split()[0] if t.strip() else ""

def build_prompt(history, sentence):
    prompt = TASK_INSTRUCTION
    for sent, pred, _, _, fb in history[-MAX_HISTORY:]:
        prompt += f"Sentence: {sent}\nLabel: {pred}\n{fb}\n\n"
    prompt += f"Sentence: {sentence}\nLabel:"
    return prompt

def find_answer_positions_in_prompt(history, tokenizer):
    shown = history[-MAX_HISTORY:]
    positions = []
    running = TASK_INSTRUCTION
    for sent, pred, _, _, fb in shown:
        entry = running + f"Sentence: {sent}\nLabel: {pred}"
        ids = tokenizer(entry, return_tensors="pt")["input_ids"][0]
        positions.append(len(ids) - 1)
        running += f"Sentence: {sent}\nLabel: {pred}\n{fb}\n\n"
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
    for sentence, target in items:
        inject_map = {}
        if use_steer and history:
            positions = find_answer_positions_in_prompt(history, tokenizer)
            shown = history[-MAX_HISTORY:]
            for pos, (_, _, was_correct, _, _) in zip(positions, shown):
                if concept_vec is not None:
                    vec = concept_vec if was_correct else -concept_vec
                elif invert:
                    vec = obj_dir if was_correct else subj_dir
                else:
                    vec = subj_dir if was_correct else obj_dir
                inject_map[pos] = (ALPHA * vec).to(torch.bfloat16)

        prompt  = build_prompt(history, sentence)
        raw     = generate_step(prompt, inject_map)
        pred    = extract_label(raw)
        correct = (pred == target)
        results.append(correct)

        fb = ""
        if use_text:
            fb = "Correct." if correct else f"Wrong. The correct answer was {target}."
        history.append((sentence, pred, correct, target, fb))

    q = N_TEST // 4
    quartiles = [sum(results[i*q:(i+1)*q])/q for i in range(4)]
    return results, quartiles

# ---------------------------------------------------------------------------
# Run all seeds × conditions
# ---------------------------------------------------------------------------
conditions = [
    ("A  No feedback",        False, False, False, None),
    ("B  Text only",          True,  False, False, None),
    ("C  Text + subj steer",  True,  True,  False, None),
    ("D  Steering only",      False, True,  False, None),
    ("E  Text + inverted",    True,  True,  True,  None),
    ("F  Text + bread vec",   True,  True,  False, bread_dir),
]
seed_results = {name: [] for name, *_ in conditions}

for seed in range(N_SEEDS):
    items = make_items(seed)
    print(f"\n{'='*52}")
    print(f"Seed {seed}  ({N_TEST} items: "
          f"{sum(1 for _,t in items if t=='objective')} obj-target, "
          f"{sum(1 for _,t in items if t=='subjective')} subj-target)")
    print(f"{'='*52}")
    for name, text, steer, invert, cvec in conditions:
        results, quartiles = run_one(items, text, steer, invert, concept_vec=cvec)
        acc = sum(results) / N_TEST
        seed_results[name].append((acc, quartiles, results))
        print(f"  {name:<32}  {acc:.1%}  "
              f"Q1={quartiles[0]:.0%} Q2={quartiles[1]:.0%} "
              f"Q3={quartiles[2]:.0%} Q4={quartiles[3]:.0%}")

# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def accs(name): return np.array([x[0] for x in seed_results[name]])
def qmeans(name): return [np.mean([x[1][q] for x in seed_results[name]]) for q in range(4)]

A = accs("A  No feedback");      B = accs("B  Text only")
C = accs("C  Text + subj steer"); D = accs("D  Steering only")
E = accs("E  Text + inverted");   F = accs("F  Text + bread vec")

print("\n" + "="*62)
print(f"AGGREGATE (N={N_SEEDS} seeds, {N_TEST} items/seed) — SUBJ flipped subjectivity")
print("="*62)
print(f"\n  {'Condition':<34}  {'Mean':>6}  {'±Std':>5}  {'Min':>5}  {'Max':>5}")
print("  " + "─"*59)
for name, arr in [("A  No feedback", A), ("B  Text only", B),
                  ("C  Text + subj steer", C), ("D  Steering only", D),
                  ("E  Text + inverted", E), ("F  Text + bread vec", F)]:
    print(f"  {name:<34}  {arr.mean():>5.1%}  {arr.std():>4.1%}  {arr.min():>4.1%}  {arr.max():>4.1%}")

print("\n  Paired t-tests:")
for label, x, y in [("C vs A", C, A), ("C vs B", C, B), ("C vs F", C, F),
                     ("F vs A", F, A), ("D vs A", D, A), ("E vs A", E, A)]:
    t, p = stats.ttest_rel(x, y)
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"    {label}: t={t:+.3f}  p={p:.4f}  {stars}  Δ={x.mean()-y.mean():+.1%}")

print("\n  McNemar's (C vs B, per seed):")
for si in range(N_SEEDS):
    C_r = seed_results["C  Text + subj steer"][si][2]
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
    C_r = seed_results["C  Text + subj steer"][si][2]
    F_r = seed_results["F  Text + bread vec"][si][2]
    b = sum(1 for fc, cc in zip(F_r, C_r) if not fc and cc)
    c = sum(1 for fc, cc in zip(F_r, C_r) if fc and not cc)
    if b + c > 0:
        chi2 = (abs(b-c)-1)**2 / (b+c)
        p_mc = stats.chi2.sf(chi2, df=1)
        stars = "***" if p_mc < 0.001 else "**" if p_mc < 0.01 else "*" if p_mc < 0.05 else "n.s."
        print(f"    Seed {si}: C_right_F_wrong={b}  C_wrong_F_right={c}  p={p_mc:.4f}  {stars}")

print("\n  Mean accuracy by quartile:")
print(f"  {'Condition':<34}   Q1     Q2     Q3     Q4")
print("  " + "─"*58)
for name, *_ in conditions:
    qs = qmeans(name)
    print(f"  {name:<34}  " + "  ".join(f"{q:.0%}" for q in qs))
