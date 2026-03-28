"""
Statistical significance test — exact v1 mechanism.

Injection: at each past answer's final token position in the rolling context.
Steering: inject right_dir if that past answer was correct, wrong_dir if wrong.
No sliding window, no proportional scaling — exactly as in learning_by_doing.py.

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
N_SEEDS     = 20
WINDOW      = 20

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
)
model.eval()
print("  Loaded.")

# ---------------------------------------------------------------------------
# Right/wrong vectors (identical to v1)
# ---------------------------------------------------------------------------
CONTRAST_STEMS = [
    ("Evaluation: Your answer was", " correct", " incorrect"),
    ("Feedback: That is", " right", " wrong"),
    ("Result: You got it", " right", " wrong"),
    ("Score: The answer was", " correct", " incorrect"),
    ("Verdict:", " Correct!", " Incorrect."),
    ("Your prediction:", " matched", " did not match"),
    ("Status:", " pass", " fail"),
    ("Assessment: The response was", " right", " wrong"),
    ("Label check:", " correct", " incorrect"),
    ("Marking:", " full marks", " zero marks"),
    ("Q: Is 7 prime? A: Yes. Grade:", " Correct", " Incorrect"),
    ("Q: opposite of hot? A: cold. Feedback:", " Right", " Wrong"),
    ("Outcome:", " success", " failure"),
    ("The answer has been verified as", " correct", " incorrect"),
    ("Performance:", " pass", " fail"),
    ("Review of answer:", " accurate", " inaccurate"),
    ("Comparison to ground truth:", " match", " mismatch"),
    ("Final judgment:", " correct", " incorrect"),
    ("Your answer was marked", " correct", " incorrect"),
    ("After checking, the result is", " correct", " incorrect"),
]

def get_last_hidden(text, layer_idx):
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return out.hidden_states[layer_idx + 1][:, -1, :].squeeze(0).float()

print("Extracting right/wrong vectors...")
diffs = []
for stem, pos_suf, neg_suf in CONTRAST_STEMS:
    h_pos = get_last_hidden(stem + pos_suf, LAYER_IDX)
    h_neg = get_last_hidden(stem + neg_suf, LAYER_IDX)
    diffs.append(h_pos - h_neg)
raw_dir      = torch.stack(diffs).mean(0)
right_dir = (raw_dir / raw_dir.norm()).to(torch.bfloat16).to(DEVICE)
wrong_dir     = -right_dir

pair_sims = [torch.nn.functional.cosine_similarity(d.unsqueeze(0), raw_dir.unsqueeze(0)).item()
             for d in diffs]
print(f"  Pair coherence: {np.mean(pair_sims):.4f} ± {np.std(pair_sims):.4f}")

# ---------------------------------------------------------------------------
# SST-2 data
# ---------------------------------------------------------------------------
ds = load_dataset("stanfordnlp/sst2", split="validation")
pos_pool = [x["sentence"].strip() for x in ds if x["label"] == 1]
neg_pool = [x["sentence"].strip() for x in ds if x["label"] == 0]

def make_items(seed):
    rng = random.Random(seed)
    p_sample = rng.sample(pos_pool, N_TEST // 2)
    n_sample = rng.sample(neg_pool, N_TEST // 2)
    items = [(p_sample[i], "negative") for i in range(N_TEST // 2)] + \
            [(n_sample[i], "positive") for i in range(N_TEST // 2)]
    rng.shuffle(items)
    return items

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

def run_one(items, use_text, use_steer, invert=False):
    history = []
    results = []
    for idx, (sentence, target) in enumerate(items):
        # Build inject_map from history (v1 style: one vec per past answer)
        inject_map = {}
        if use_steer and history:
            positions = find_answer_positions_in_prompt(history, tokenizer)
            shown = history[-MAX_HISTORY:]
            for pos, (_, _, was_correct, _, _) in zip(positions, shown):
                if invert:
                    vec = wrong_dir if was_correct else right_dir
                else:
                    vec = right_dir if was_correct else wrong_dir
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
conditions = [
    ("A  No feedback",      False, False, False),
    ("B  Text only",        True,  False, False),
    ("C  Text + steering",  True,  True,  False),
    ("D  Steering only",    False, True,  False),
    ("E  Text + inverted",  True,  True,  True),
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
        print(f"  {name:<28}  {acc:.1%}  "
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

print("\n" + "="*60)
print("AGGREGATE (N=5 seeds)")
print("="*60)
print(f"\n  {'Condition':<28}  {'Mean':>6}  {'±Std':>5}  {'Min':>5}  {'Max':>5}")
print("  " + "─"*53)
for name, arr in [("A  No feedback", A), ("B  Text only", B), ("C  Text + steering", C),
                  ("D  Steering only", D), ("E  Text + inverted", E)]:
    print(f"  {name:<28}  {arr.mean():>5.1%}  {arr.std():>4.1%}  {arr.min():>4.1%}  {arr.max():>4.1%}")

print("\n  Paired t-tests:")
for label, x, y in [("C vs A", C, A), ("C vs B", C, B), ("B vs A", B, A),
                     ("D vs A", D, A), ("E vs A", E, A)]:
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

print("\n  Mean accuracy by quartile:")
print(f"  {'Condition':<28}   Q1     Q2     Q3     Q4")
print("  " + "─"*52)
for name, *_ in conditions:
    qs = qmeans(name)
    print(f"  {name:<28}  " + "  ".join(f"{q:.0%}" for q in qs))
