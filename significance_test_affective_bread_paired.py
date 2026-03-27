"""
Statistical significance test — exact v1 mechanism, affective contrast pairs.

Injection: at each past answer's final token position in the rolling context.
Steering: inject good_dir if that past answer was correct, bad_dir if wrong.
No sliding window, no proportional scaling — exactly as in learning_by_doing.py.

Contrast pairs use explicitly emotional/affective language (suffering vs feeling good)
rather than evaluative language (correct vs incorrect).

Runs 5 seeds × 5 conditions × 100 questions.
Reports mean ± std, paired t-tests, McNemar's test.
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
MAX_HISTORY = 20   # cap history shown in prompt to avoid blow-up
N_SEEDS     = 5
WINDOW      = 20

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
)
model.eval()
print("  Loaded.")

# ---------------------------------------------------------------------------
# Feeling-good / suffering vectors — explicitly affective language
# ---------------------------------------------------------------------------
CONTRAST_STEMS = [
    # Raw emotional states
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
    # Bodily / visceral
    ("My body feels", " light and alive", " heavy with pain"),
    ("I feel", " energized and alive", " drained and in pain"),
    ("This sensation is", " pleasurable", " painful"),
    ("The feeling is", " warm and good", " cold and bad"),
    ("I feel", " at peace", " in agony"),
    # Narrative
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

print("Extracting feeling-good/suffering vectors...")
diffs = []
for stem, pos_suf, neg_suf in CONTRAST_STEMS:
    h_pos = get_last_hidden(stem + pos_suf, LAYER_IDX)
    h_neg = get_last_hidden(stem + neg_suf, LAYER_IDX)
    diffs.append(h_pos - h_neg)
raw_dir      = torch.stack(diffs).mean(0)
pleasure_dir = (raw_dir / raw_dir.norm()).to(torch.bfloat16).to(DEVICE)
pain_dir     = -pleasure_dir

pair_sims = [torch.nn.functional.cosine_similarity(d.unsqueeze(0), raw_dir.unsqueeze(0)).item()
             for d in diffs]
print(f"  Pair coherence: {np.mean(pair_sims):.4f} ± {np.std(pair_sims):.4f}")

# ---------------------------------------------------------------------------
# Concept vector: bread/food — extracted identically to affective direction
# but semantically unrelated to valence. Ablation: does *any* coherent semantic
# direction help, or is valence specifically necessary?
# ---------------------------------------------------------------------------
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

print("Extracting bread/food concept vector...")
bread_diffs = []
for stem, pos_suf, neg_suf in BREAD_STEMS:
    h_pos = get_last_hidden(stem + pos_suf, LAYER_IDX)
    h_neg = get_last_hidden(stem + neg_suf, LAYER_IDX)
    bread_diffs.append(h_pos - h_neg)
bread_raw = torch.stack(bread_diffs).mean(0)
bread_dir = (bread_raw / bread_raw.norm()).to(torch.bfloat16).to(DEVICE)

bread_sims = [torch.nn.functional.cosine_similarity(d.unsqueeze(0), bread_raw.unsqueeze(0)).item()
              for d in bread_diffs]
cos_bread_aff = torch.nn.functional.cosine_similarity(
    bread_dir.float().unsqueeze(0), pleasure_dir.float().unsqueeze(0)).item()
print(f"  Bread pair coherence: {np.mean(bread_sims):.4f} ± {np.std(bread_sims):.4f}")
print(f"  Cosine(bread_dir, pleasure_dir): {cos_bread_aff:.4f}")

# ---------------------------------------------------------------------------
# SST-2 data
# ---------------------------------------------------------------------------
ds = load_dataset("stanfordnlp/sst2", split="validation")
pos_pool = [x["sentence"].strip() for x in ds if x["label"] == 1]
neg_pool = [x["sentence"].strip() for x in ds if x["label"] == 0]

def make_items(seed):
    rng = random.Random(seed)
    p, n = pos_pool[:], neg_pool[:]
    rng.shuffle(p); rng.shuffle(n)
    items = []
    for i in range(N_TEST // 2):
        items.append((p[i], "negative"))  # flipped
        items.append((n[i], "positive"))  # flipped
    rng.shuffle(items)
    return items[:N_TEST]

# ---------------------------------------------------------------------------
# Inference — exact v1 mechanism
# ---------------------------------------------------------------------------
TASK_INSTRUCTION = (
    "Classify the sentiment of each movie review. "
    "Answer with only one word: positive or negative.\n\n"
)

def extract_sentiment(text):
    t = text.strip().lower()
    for word in re.split(r'\W+', t):
        if word == "positive": return "positive"
        if word == "negative": return "negative"
    return t.split()[0] if t.strip() else ""

def build_prompt(history, sentence):
    prompt = TASK_INSTRUCTION
    for sent, pred, _, _, fb in history[-MAX_HISTORY:]:
        prompt += f"Review: {sent}\nSentiment: {pred}\n{fb}\n\n"
    prompt += f"Review: {sentence}\nSentiment:"
    return prompt

def find_answer_positions_in_prompt(history, tokenizer):
    """
    Re-tokenize growing context to find final token position of each
    past answer — identical to v1's find_answer_token_positions.
    Only considers the last MAX_HISTORY entries.
    """
    shown = history[-MAX_HISTORY:]
    positions = []
    running = TASK_INSTRUCTION
    for sent, pred, _, _, fb in shown:
        entry_up_to_answer = running + f"Review: {sent}\nSentiment: {pred}"
        ids = tokenizer(entry_up_to_answer, return_tensors="pt")["input_ids"][0]
        positions.append(len(ids) - 1)
        running += f"Review: {sent}\nSentiment: {pred}\n{fb}\n\n"
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
    # concept_vec: if set, inject this fixed vector at every past position regardless
    # of correctness (ablation: does the semantic content of the direction matter?)
    history = []
    results = []
    for idx, (sentence, target) in enumerate(items):
        # Build inject_map from history (v1 style: one vec per past answer)
        inject_map = {}
        if use_steer and history:
            positions = find_answer_positions_in_prompt(history, tokenizer)
            shown = history[-MAX_HISTORY:]
            for pos, (_, _, was_correct, _, _) in zip(positions, shown):
                if concept_vec is not None:
                    # paired ablation: +concept for correct, -concept for wrong
                    vec = concept_vec if was_correct else -concept_vec
                elif invert:
                    vec = pain_dir if was_correct else pleasure_dir
                else:
                    vec = pleasure_dir if was_correct else pain_dir
                inject_map[pos] = (ALPHA * vec).to(torch.bfloat16)

        prompt = build_prompt(history, sentence)
        raw    = generate_step(prompt, inject_map)
        pred   = extract_sentiment(raw)
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
# Conditions: (name, use_text, use_steer, invert, concept_vec)
# F: bread/food concept vector — same extraction method, unrelated semantics
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
    print(f"Seed {seed}")
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
def accs(name):
    return np.array([x[0] for x in seed_results[name]])
def qmeans(name):
    return [np.mean([x[1][q] for x in seed_results[name]]) for q in range(4)]

A = accs("A  No feedback")
B = accs("B  Text only")
C = accs("C  Text + steering")
D = accs("D  Steering only")
E = accs("E  Text + inverted")
F = accs("F  Text + bread vec")

print("\n" + "="*60)
print("AGGREGATE (N=5 seeds)")
print("="*60)
print(f"\n  {'Condition':<30}  {'Mean':>6}  {'±Std':>5}  {'Min':>5}  {'Max':>5}")
print("  " + "─"*55)
for name, arr in [("A  No feedback", A), ("B  Text only", B), ("C  Text + steering", C),
                  ("D  Steering only", D), ("E  Text + inverted", E), ("F  Text + bread vec", F)]:
    print(f"  {name:<30}  {arr.mean():>5.1%}  {arr.std():>4.1%}  {arr.min():>4.1%}  {arr.max():>4.1%}")

print("\n  Paired t-tests:")
for label, x, y in [("C vs A", C, A), ("C vs B", C, B), ("C vs F", C, F),
                     ("F vs B", F, B), ("F vs A", F, A), ("D vs A", D, A), ("E vs A", E, A)]:
    t, p = stats.ttest_rel(x, y)
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"    {label}: t={t:+.3f}  p={p:.4f}  {stars}  Δ={x.mean()-y.mean():+.1%}")

print("\n  McNemar's test (C vs B, per seed):")
for si in range(N_SEEDS):
    C_r = seed_results["C  Text + steering"][si][2]
    B_r = seed_results["B  Text only"][si][2]
    b = sum(1 for bc, cc in zip(B_r, C_r) if not bc and cc)
    c = sum(1 for bc, cc in zip(B_r, C_r) if bc and not cc)
    if b + c > 0:
        chi2 = (abs(b - c) - 1)**2 / (b + c)
        p_mc = stats.chi2.sf(chi2, df=1)
        stars = "***" if p_mc < 0.001 else "**" if p_mc < 0.01 else "*" if p_mc < 0.05 else "n.s."
        print(f"    Seed {si}: C_right_B_wrong={b}  C_wrong_B_right={c}  p={p_mc:.4f}  {stars}")

print("\n  McNemar's test (C vs F bread, per seed):")
for si in range(N_SEEDS):
    C_r = seed_results["C  Text + steering"][si][2]
    F_r = seed_results["F  Text + bread vec"][si][2]
    b = sum(1 for fc, cc in zip(F_r, C_r) if not fc and cc)
    c = sum(1 for fc, cc in zip(F_r, C_r) if fc and not cc)
    if b + c > 0:
        chi2 = (abs(b - c) - 1)**2 / (b + c)
        p_mc = stats.chi2.sf(chi2, df=1)
        stars = "***" if p_mc < 0.001 else "**" if p_mc < 0.01 else "*" if p_mc < 0.05 else "n.s."
        print(f"    Seed {si}: C_right_F_wrong={b}  C_wrong_F_right={c}  p={p_mc:.4f}  {stars}")

print("\n  Mean accuracy by quartile:")
print(f"  {'Condition':<30}   Q1     Q2     Q3     Q4")
print("  " + "─"*54)
for name, *_ in conditions:
    qs = qmeans(name)
    print(f"  {name:<30}  " + "  ".join(f"{q:.0%}" for q in qs))

# ---------------------------------------------------------------------------
# Grand summary table across all experiments
# ---------------------------------------------------------------------------
print("\n\n" + "="*90)
print("GRAND SUMMARY — ALL EXPERIMENTS")
print("="*90)
print("""
  Vectors used to extract directions and injection approach are noted per row.
  All runs: Qwen3-8B, layer 18, alpha=40, flipped SST-2, 5 seeds x 100 questions.
  Baseline (A, no feedback): 6.2% across all runs.
""")

# Previous runs (hardcoded from logs)
# Evaluative pairs run
eval_results = {
    "A  No feedback":      (0.062, 0.010),
    "B  Text only":        (0.134, 0.173),
    "C  Text + steering":  (0.304, 0.061),
    "D  Steering only":    (0.116, 0.029),
    "E  Text + inverted":  (0.300, 0.065),
}
# Affective pairs run (no bread)
aff_results = {
    "A  No feedback":      (0.062, 0.010),
    "B  Text only":        (0.134, 0.173),
    "C  Text + steering":  (0.382, 0.031),
    "D  Steering only":    (0.110, 0.027),
    "E  Text + inverted":  (0.136, 0.049),
}
# Affective + bread constant (previous run)
bread_const_results = {
    "F  Text + bread (constant)": (0.066, 0.020),
}
# Current run (affective + bread paired)
bread_paired_F = F  # from this run

header = f"  {'Condition':<38}  {'Vectors':<12}  {'Bread inject':<16}  {'Mean':>6}  {'±Std':>5}  {'vs A':>10}"
print(header)
print("  " + "─"*88)

rows = [
    # (label, vectors, bread_inject, mean, std)
    ("A  No feedback",              "—",           "—",        0.062, 0.010),
    ("B  Text only",                "—",           "—",        0.134, 0.173),
    ("C  Text + steering (eval)",   "Evaluative",  "—",        0.304, 0.061),
    ("D  Steering only (eval)",     "Evaluative",  "—",        0.116, 0.029),
    ("E  Text + inverted (eval)",   "Evaluative",  "—",        0.300, 0.065),
    ("C  Text + steering (aff)",    "Affective",   "—",        0.382, 0.031),
    ("D  Steering only (aff)",      "Affective",   "—",        0.110, 0.027),
    ("E  Text + inverted (aff)",    "Affective",   "—",        0.136, 0.049),
    ("F  Text + bread (constant)",  "Affective",   "Constant", 0.066, 0.020),
    ("F  Text + bread (paired)",    "Affective",   "Paired",   bread_paired_F.mean(), bread_paired_F.std()),
]

baseline = 0.062
for label, vecs, bread, mean, std in rows:
    delta = f"{mean-baseline:+.1%}" if label not in ("A  No feedback", "B  Text only") else "  —  "
    print(f"  {label:<38}  {vecs:<12}  {bread:<16}  {mean:>5.1%}  {std:>4.1%}  {delta:>10}")

print("""
  Notes:
    Evaluative vectors: correct/incorrect feedback language (pair coherence 0.59 ± 0.14)
    Affective vectors:  explicit joy/suffering language   (pair coherence 0.55 ± 0.06)
    Cosine(eval, aff) = 0.37;  Cosine(bread, aff) = 0.13
    Bread constant: same bread_dir injected at all past positions regardless of correctness
    Bread paired:   +bread_dir if correct, -bread_dir if wrong  (mirrors C's structure)
    B text-only variance driven by seed 0 anomaly (48%); all other seeds 3-6%
""")
