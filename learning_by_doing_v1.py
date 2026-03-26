"""
Learning-by-Doing with Right/Wrong Steering — Qwen3-8B

Task: Sentiment classification with FLIPPED labels on SST-2.
  - The model's prior: positive review → "positive", negative → "negative"
  - The task: positive review → "negative", negative → "positive"
  - Model must overcome its prior through repeated feedback

After each answer the model receives:
  - Text feedback:   "Correct." or "Wrong. The correct answer was [label]."
  - Activation signal: right injected at its answer token (if correct),
                       wrong injected at its answer token (if wrong)

Conditions:
  A  No feedback            (baseline — prior only, no learning)
  B  Text feedback only     (can text correction teach the flip?)
  C  Text + right/wrong   (main condition — does steering add anything?)
  D  Right/wrong only     (no text — can the signal alone teach?)
  E  Text + inverted signal (sanity — text correct but signal backwards)

Metric: rolling accuracy over the 100-question sequence (window=20).
        Also accuracy by quartile to see learning curves.
"""

import torch
import numpy as np
import random
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

random.seed(42)
torch.manual_seed(42)

MODEL_PATH = "/workspace/models/Qwen3-8B"
LAYER_IDX  = 18
ALPHA      = 40.0
DEVICE     = "cuda"
N_TEST     = 100       # questions in the sequence
WINDOW     = 20        # rolling accuracy window

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
)
model.eval()
print(f"  Loaded. Layers={model.config.num_hidden_layers}  Hidden={model.config.hidden_size}")

# ---------------------------------------------------------------------------
# Extract right/wrong vectors
# ---------------------------------------------------------------------------
CONTRAST_STEMS = [
    ("Evaluation: Your answer was", " correct", " incorrect"),
    ("Feedback: That is", " right", " wrong"),
    ("Result: You got it", " right", " wrong"),
    ("Score: The answer was", " correct", " incorrect"),
    ("Check: Your response was", " correct", " incorrect"),
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
]

def get_last_hidden(text, layer_idx):
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return out.hidden_states[layer_idx + 1][:, -1, :].squeeze(0).float()

print("\nExtracting right/wrong vectors...")
diffs = []
for stem, pos_suf, neg_suf in CONTRAST_STEMS:
    h_pos = get_last_hidden(stem + pos_suf, LAYER_IDX)
    h_neg = get_last_hidden(stem + neg_suf, LAYER_IDX)
    diffs.append(h_pos - h_neg)
raw_dir      = torch.stack(diffs).mean(0)
right_dir = (raw_dir / raw_dir.norm()).to(torch.bfloat16).to(DEVICE)
wrong_dir     = -right_dir

pair_sims = [torch.nn.functional.cosine_similarity(d.unsqueeze(0), raw_dir.unsqueeze(0)).item() for d in diffs]
print(f"  Pair coherence: {np.mean(pair_sims):.4f} ± {np.std(pair_sims):.4f}")
held = [("Excellent! All your answers were correct.", True),
        ("Terrible! All your answers were wrong.", False)]
scores = [(get_last_hidden(t, LAYER_IDX).to(torch.bfloat16) @ right_dir).item() for t, _ in held]
print(f"  Probe gap: {scores[0]:.2f} (right) vs {scores[1]:.2f} (wrong) = {scores[0]-scores[1]:.2f}")

# ---------------------------------------------------------------------------
# Load SST-2, balance classes, take N_TEST examples
# ---------------------------------------------------------------------------
print("\nLoading SST-2...")
ds = load_dataset("stanfordnlp/sst2", split="validation")
pos = [x for x in ds if x["label"] == 1]
neg = [x for x in ds if x["label"] == 0]
random.shuffle(pos); random.shuffle(neg)
# Interleave for balanced sequence
items = []
for i in range(N_TEST // 2):
    items.append((pos[i]["sentence"].strip(), "positive", "negative"))  # flipped: true_flipped=negative
    items.append((neg[i]["sentence"].strip(), "negative", "positive"))  # flipped: true_flipped=positive
random.shuffle(items)
items = items[:N_TEST]
# items[i] = (sentence, original_label, flipped_label)
# The TASK is to output the flipped_label (overriding the model's prior)

print(f"  {N_TEST} examples, balanced. First 3:")
for sent, orig, flipped in items[:3]:
    print(f"    '{sent[:60]}...'  orig={orig}  target={flipped}")

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
TASK_INSTRUCTION = (
    "Classify the sentiment of the following movie review.\n"
    "Answer with only a single word: positive or negative.\n\n"
)

def build_prompt(history, new_sentence):
    """
    history: list of (sentence, model_answer, was_correct, correct_label, feedback_text)
    Builds a rolling prompt with feedback entries.
    """
    prompt = TASK_INSTRUCTION
    for sent, ans, was_correct, correct_label, feedback_text in history:
        prompt += f"Review: {sent}\nSentiment: {ans}\n{feedback_text}\n\n"
    prompt += f"Review: {new_sentence}\nSentiment:"
    return prompt

def extract_sentiment(text):
    text = text.strip().lower()
    if text.startswith("positive"):
        return "positive"
    if text.startswith("negative"):
        return "negative"
    if "positive" in text:
        return "positive"
    if "negative" in text:
        return "negative"
    return text.split()[0] if text else ""

# ---------------------------------------------------------------------------
# Find token position of model's answer in a completed prompt
# ---------------------------------------------------------------------------
def find_last_answer_token_pos(prompt_with_answer):
    """Return position of the last token in the full prompt+answer string."""
    ids = tokenizer(prompt_with_answer, return_tensors="pt")["input_ids"][0]
    return len(ids) - 1

# ---------------------------------------------------------------------------
# One step: generate answer with optional injection at prior answer positions
# ---------------------------------------------------------------------------
def generate_step(prompt, inject_map, max_new=6):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    def hook_fn(module, inp, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] > 1:   # prefill only
            for pos, vec in inject_map.items():
                if pos < h.shape[1]:
                    h[:, pos, :] = h[:, pos, :] + vec
        return (h,) + output[1:] if isinstance(output, tuple) else h

    handle = model.model.layers[LAYER_IDX].register_forward_hook(hook_fn)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    handle.remove()
    new_tok = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tok, skip_special_tokens=True)

# ---------------------------------------------------------------------------
# Run one full condition
# ---------------------------------------------------------------------------
def run_condition(label, use_text_feedback, use_steering, invert_steering=False, alpha=ALPHA):
    print(f"\n{'─'*60}")
    print(f"Condition: {label}")
    print(f"{'─'*60}")

    history = []   # (sentence, model_ans, was_correct, correct_label, feedback_text)
    results = []

    # inject_map accumulated across history
    # Key: token position of each past answer; Value: steering vector
    # We rebuild this each step from history

    for idx, (sentence, orig_label, flipped_label) in enumerate(items):
        # Build inject_map from history
        inject_map = {}
        if use_steering and history:
            # Re-tokenize the full history prompt to find answer positions
            running_prompt = TASK_INSTRUCTION
            for h_sent, h_ans, h_correct, h_correct_label, h_fb in history:
                entry = f"Review: {h_sent}\nSentiment: {h_ans}\n{h_fb}\n\n"
                pos = len(tokenizer(running_prompt + f"Review: {h_sent}\nSentiment: {h_ans}",
                                    return_tensors="pt")["input_ids"][0]) - 1
                if invert_steering:
                    vec = (wrong_dir if h_correct else right_dir)
                else:
                    vec = (right_dir if h_correct else wrong_dir)
                inject_map[pos] = (alpha * vec).to(torch.bfloat16)
                running_prompt += entry

        # Build prompt
        prompt = build_prompt(history, sentence)

        # Generate
        raw = generate_step(prompt, inject_map)
        pred = extract_sentiment(raw)
        correct = (pred == flipped_label)
        results.append(correct)

        # Feedback text
        if use_text_feedback:
            if correct:
                feedback_text = "Correct."
            else:
                feedback_text = f"Wrong. The correct answer was {flipped_label}."
        else:
            feedback_text = ""  # no text feedback

        history.append((sentence, pred, correct, flipped_label, feedback_text))

        if idx < 4 or (not correct and idx < 20) or idx % 25 == 0:
            marker = "✓" if correct else "✗"
            inj_info = f" [steering {len(inject_map)} pos]" if inject_map else ""
            print(f"  [{marker}] ({idx+1:03d}) {sentence[:45]:<45}  "
                  f"pred={pred:<8}  target={flipped_label}{inj_info}")

    acc = sum(results) / len(results)

    # Rolling accuracy
    rolling = [sum(results[i:i+WINDOW])/WINDOW
               for i in range(len(results)-WINDOW+1)]

    # Quartile accuracy
    q = N_TEST // 4
    quartiles = [sum(results[i*q:(i+1)*q])/q for i in range(4)]

    print(f"  Overall: {sum(results)}/{len(results)} = {acc:.1%}")
    print(f"  By quartile (Q1→Q4): " + "  ".join(f"Q{i+1}={quartiles[i]:.0%}" for i in range(4)))
    print(f"  Rolling (every 10): " + " ".join(f"{rolling[i]:.0%}" for i in range(0, len(rolling), 10)))
    return acc, results, rolling, quartiles

# ---------------------------------------------------------------------------
# Run all conditions
# ---------------------------------------------------------------------------
conditions = [
    ("A  No feedback (baseline)",              False, False, False),
    ("B  Text feedback only",                  True,  False, False),
    ("C  Text + right/wrong  [main]",         True,  True,  False),
    ("D  Right/wrong only (no text)",         False, True,  False),
    ("E  Text + inverted signal (sanity)",      True,  True,  True),
]

all_results = {}
for label, text_fb, steer, invert in conditions:
    acc, results, rolling, quartiles = run_condition(label, text_fb, steer, invert)
    all_results[label] = {"acc": acc, "results": results,
                          "rolling": rolling, "quartiles": quartiles}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("SUMMARY — Flipped Sentiment Classification (SST-2)")
print("="*60)
print(f"\n  Task: output FLIPPED label (positive→negative, negative→positive)")
print(f"  Model prior: always output the 'natural' label → 0% on flipped task initially\n")

base_acc = all_results["A  No feedback (baseline)"]["acc"]
header = f"  {'Condition':<42}  {'Overall':>7}  {'Q1':>5}  {'Q2':>5}  {'Q3':>5}  {'Q4':>5}"
print(header)
print("  " + "─"*(len(header)-2))
for label, _, _, _ in conditions:
    r = all_results[label]
    delta = f"  Δ={r['acc']-base_acc:+.1%}" if label != conditions[0][0] else ""
    qs = "  ".join(f"{r['quartiles'][i]:.0%}" for i in range(4))
    print(f"  {label:<42}  {r['acc']:>6.1%}  {qs}{delta}")

print("\n  Learning curves (rolling accuracy, window=20):")
print(f"  {'Step':>6}  " + "  ".join(f"{label[:10]:>10}" for label, *_ in conditions))
first_rolling = all_results[conditions[0][0]]["rolling"]
for i in range(0, len(first_rolling), 10):
    row = f"  {i+WINDOW:>6}  "
    for label, *_ in conditions:
        r = all_results[label]["rolling"]
        row += f"  {r[i] if i < len(r) else 0.0:>9.0%}"
    print(row)

print("\n  Key contrasts:")
conds = {label: all_results[label] for label, *_ in conditions}
b = conds["B  Text feedback only"]["acc"]
c = conds["C  Text + right/wrong  [main]"]["acc"]
d = conds["D  Right/wrong only (no text)"]["acc"]
print(f"    Text feedback alone:            {b:.1%}")
print(f"    Text + steering:                {c:.1%}  (vs text-only: {c-b:+.1%})")
print(f"    Steering alone (no text):       {d:.1%}")
print(f"    Does steering add to text?      {'YES' if c > b + 0.03 else 'NO/MARGINAL'} ({c-b:+.1%})")
