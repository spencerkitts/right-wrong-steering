# Pain/Pleasure Activation Steering — Learning by Doing

Experiments testing whether injecting "pain" and "pleasure" activation vectors into a language model's residual stream can help it learn from sequential feedback.

## Setup

Model: **Qwen3-8B** (layer 18, bfloat16)
Task: **Flipped SST-2 sentiment** — classify movie reviews with *inverted* labels (positive→negative, negative→positive). The model has a strong prior toward natural labels, so it must learn to override it.

## Method

### Pain/Pleasure Vectors (RepE-style)

Extract a "pleasure direction" in activation space using 20 contrast pairs:

```
("Evaluation: Your answer was", " correct", " incorrect")
("Feedback: That is", " right", " wrong")
...
```

For each pair, run both strings through the model, grab the last token's hidden state at layer 18, and subtract. Average the 20 difference vectors and normalize → `pleasure_dir`. `pain_dir = -pleasure_dir`.

### Learning by Doing

Questions are presented sequentially. After each answer:
1. **Text feedback** (conditions B, C, E): "Correct." or "Wrong. The correct answer was X."
2. **Activation injection** (conditions C, D, E): Inject `pleasure_dir` at the token position of each past *correct* answer, `pain_dir` at each past *wrong* answer. Uses a forward hook on `model.model.layers[18]` during prefill.

## Conditions

| Label | Description |
|-------|-------------|
| A | No feedback (baseline) |
| B | Text feedback only |
| C | Text + pain/pleasure steering |
| D | Steering only (no text) |
| E | Text + inverted steering (sanity check) |

## Results (single seed, v1)

| Condition | Accuracy | Q1 | Q2 | Q3 | Q4 |
|-----------|----------|----|----|----|----|
| A No feedback | 10% | 8% | 4% | 4% | 24% |
| B Text only | 7% | 8% | 0% | 4% | 16% |
| C Text + steering | **39%** | 20% | 56% | 36% | 44% |
| D Steering only | 15% | 12% | 16% | 4% | 28% |
| E Text + inverted | 14% | 16% | 0% | 8% | 32% |

Text alone doesn't help (model ignores "Wrong, the answer was X"). Text + steering yields +32pp over text-only — the activation signal appears to be what drives learning.

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
