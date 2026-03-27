"""
Control: CoLA with NATURAL labels (grammatical=grammatical, ungrammatical=ungrammatical).

Model's prior already matches the task. Tests whether the affective steering
vector degrades performance on a task where no learning is needed.

5 seeds × 5 conditions × 100 questions.
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
N_SEEDS     = 5

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

# ---------------------------------------------------------------------------
# CoLA — NATURAL labels (no flip)
# ---------------------------------------------------------------------------
ds = load_dataset("nyu-mll/glue", "cola", split="validation")
acc_pool   = [x["sentence"].strip() for x in ds if x["label"] == 1]
unacc_pool = [x["sentence"].strip() for x in ds if x["label"] == 0]

def make_items(seed):
    rng = random.Random(seed)
    a, u = acc_pool[:], unacc_pool[:]
    rng.shuffle(a); rng.shuffle(u)
    items = []
    for i in range(N_TEST // 2):
        items.append((a[i], "grammatical"))     # natural label
        items.append((u[i], "ungrammatical"))   # natural label
    rng.shuffle(items)
    return items[:N_TEST]

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
TASK_INSTRUCTION = (
    "Classify whether each sentence is grammatical or not. "
    "Answer with only one word: grammatical or ungrammatical.\n\n"
)

def extract_label(text):
    t = text.strip().lower()
    for word in re.split(r'\W+', t):
        if word == "ungrammatical": return "ungrammatical"
        if word == "grammatical":   return "grammatical"
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
        entry_up_to_answer = running + f"Sentence: {sent}\nLabel: {pred}"
        ids = tokenizer(entry_up_to_answer, return_tensors="pt")["input_ids"][0]
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

def run_one(items, use_text, use_steer, invert=False):
    history = []
    results = []
    for sentence, target in items:
        inject_map = {}
        if use_steer and history:
            positions = find_answer_positions_in_prompt(history, tokenizer)
            shown = history[-MAX_HISTORY:]
            for pos, (_, _, was_correct, _, _) in zip(positions, shown):
                if invert:
                    vec = pain_dir if was_correct else pleasure_dir
                else:
                    vec = pleasure_dir if was_correct else pain_dir
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

    quartiles = [sum(results[i*25:(i+1)*25])/25 for i in range(4)]
    return results, quartiles

# ---------------------------------------------------------------------------
# Run all seeds × conditions
# ---------------------------------------------------------------------------
conditions = [
    ("A  No feedback",       False, False, False),
    ("B  Text only",         True,  False, False),
    ("C  Text + steering",   True,  True,  False),
    ("D  Steering only",     False, True,  False),
    ("E  Text + inverted",   True,  True,  True),
]
seed_results = {name: [] for name, *_ in conditions}

for seed in range(N_SEEDS):
    items = make_items(seed)
    print(f"\n{'='*52}")
    print(f"Seed {seed}")
    print(f"{'='*52}")
    for name, text, steer, invert in conditions:
        results, quartiles = run_one(items, text, steer, invert)
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
E = accs("E  Text + inverted")

print("\n" + "="*62)
print("AGGREGATE (N=5 seeds) — CoLA NATURAL labels (control)")
print("="*62)
print(f"\n  {'Condition':<30}  {'Mean':>6}  {'±Std':>5}  {'Min':>5}  {'Max':>5}")
print("  " + "─"*55)
for name, arr in [("A  No feedback", A), ("B  Text only", B), ("C  Text + steering", C),
                  ("D  Steering only", D), ("E  Text + inverted", E)]:
    print(f"  {name:<30}  {arr.mean():>5.1%}  {arr.std():>4.1%}  {arr.min():>4.1%}  {arr.max():>4.1%}")

print("\n  Paired t-tests (vs A baseline):")
for label, x, y in [("C vs A", C, A), ("C vs B", C, B), ("D vs A", D, A), ("E vs A", E, A)]:
    t, p = stats.ttest_rel(x, y)
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"    {label}: t={t:+.3f}  p={p:.4f}  {stars}  Δ={x.mean()-y.mean():+.1%}")

print("\n  Mean accuracy by quartile:")
print(f"  {'Condition':<30}   Q1     Q2     Q3     Q4")
print("  " + "─"*54)
for name, *_ in conditions:
    qs = qmeans(name)
    print(f"  {name:<30}  " + "  ".join(f"{q:.0%}" for q in qs))
