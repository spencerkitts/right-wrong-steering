# Right/Wrong Activation Steering — Learning by Doing

Experiments testing whether injecting "right" and "wrong" activation vectors into a language model's residual stream can help it learn from sequential feedback.

## Setup

Model: **Qwen3-8B** (layer 18, bfloat16)
Task: **Flipped SST-2 sentiment** — classify movie reviews with *inverted* labels (positive→negative, negative→positive). The model has a strong prior toward natural labels, so it must learn to override it.

## Method

### Right/Wrong Vectors (RepE-style)

Extract a "right direction" in activation space using 20 contrast pairs:

```
("Evaluation: Your answer was", " correct", " incorrect")
("Feedback: That is", " right", " wrong")
...
```

For each pair, run both strings through the model, grab the last token's hidden state at layer 18, and subtract. Average the 20 difference vectors and normalize → `right_dir`. `wrong_dir = -right_dir`.

### Learning by Doing

Questions are presented sequentially. After each answer:
1. **Text feedback** (conditions B, C, E): "Correct." or "Wrong. The correct answer was X."
2. **Activation injection** (conditions C, D, E): Inject `right_dir` at the token position of each past *correct* answer, `wrong_dir` at each past *wrong* answer. Uses a forward hook on `model.model.layers[18]` during prefill.

## Conditions

| Label | Description |
|-------|-------------|
| A | No feedback (baseline) |
| B | Text feedback only |
| C | Text + right/wrong steering |
| D | Steering only (no text) |
| E | Text + inverted steering (sanity check) |

## Results

### 5-seed significance test

| Condition | Mean | ±Std | Min | Max |
|-----------|------|------|-----|-----|
| A No feedback | 6.2% | 1.0% | 5% | 8% |
| B Text only | 13.4% | 17.3% | 3% | 48% |
| C Text + steering | **30.4%** | 6.1% | 23% | 37% |
| D Steering only | 11.6% | 2.9% | 9% | 17% |
| E Text + inverted | 30.0% | 6.5% | 22% | 39% |

**Paired t-tests:** C vs A: t=+7.92, p=0.0014 (**) | C vs B: p=0.11 (n.s., high variance in B) | D vs A: p=0.038 (*)

**McNemar's (C vs B, per seed):** 4/5 seeds significant (p<0.001 in seeds 1–4); seed 0 is the exception where text-only happened to work well.

### Mean accuracy by quartile

| Condition | Q1 | Q2 | Q3 | Q4 |
|-----------|----|----|----|----|
| A No feedback | 5% | 7% | 6% | 6% |
| B Text only | 10% | 14% | 12% | 18% |
| C Text + steering | **17%** | **39%** | **35%** | **30%** |
| D Steering only | 7% | 14% | 14% | 10% |
| E Text + inverted | 21% | 32% | 25% | 42% |

### Single-seed run (original, seed=42)

| Condition | Accuracy | Q1 | Q2 | Q3 | Q4 |
|-----------|----------|----|----|----|----|
| A No feedback | 10% | 8% | 4% | 4% | 24% |
| B Text only | 7% | 8% | 0% | 4% | 16% |
| C Text + steering | **39%** | 20% | 56% | 36% | 44% |
| D Steering only | 15% | 12% | 16% | 4% | 28% |
| E Text + inverted | 14% | 16% | 0% | 8% | 32% |

### Notes

- C (text + steering) is significantly better than A (no feedback) across all seeds (p=0.0014).
- B (text only) has very high variance — in one seed the model happened to learn from text alone (48%), in others it didn't (3–6%). This makes the C vs B t-test n.s., but McNemar's within each seed shows C beats B in 4/5 seeds.
- E (text + inverted steering) performs nearly as well as C, which warrants further investigation.

## Files

- `learning_by_doing_v1.py` — original single-run experiment (seed=42)
- `significance_test.py` — 5-seed replication with paired t-tests and McNemar's test

## Requirements

```
torch
transformers
datasets
scipy
numpy
```

Model weights: `Qwen/Qwen3-8B` (or local path).
