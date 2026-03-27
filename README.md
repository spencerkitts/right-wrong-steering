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

Two variants of the contrast pairs were tested — **evaluative** (correct/incorrect feedback language) and **affective** (explicit joy/suffering language).

### Direction comparison

Cosine similarity between evaluative and affective directions: **0.37** (moderately related, same sign).

The evaluative direction is broken as a valence signal: it scores "I feel absolute bliss" at -9.75, nearly as negative as "excruciating pain" (-14.62). The affective direction correctly scores bliss as +6.56 and pain as -15.00.

### 5-seed significance test — evaluative pairs

| Condition | Mean | ±Std | Min | Max |
|-----------|------|------|-----|-----|
| A No feedback | 6.2% | 1.0% | 5% | 8% |
| B Text only | 13.4% | 17.3% | 3% | 48% |
| C Text + steering | 30.4% | 6.1% | 23% | 37% |
| D Steering only | 11.6% | 2.9% | 9% | 17% |
| E Text + inverted | 30.0% | 6.5% | 22% | 39% |

C vs A: p=0.0014 (**) | E ≈ C — inverted steering barely hurts (direction doesn't encode real valence)

### 5-seed significance test — affective pairs

| Condition | Mean | ±Std | Min | Max |
|-----------|------|------|-----|-----|
| A No feedback | 6.2% | 1.0% | 5% | 8% |
| B Text only | 13.4% | 17.3% | 3% | 48% |
| **C Text + steering** | **38.2%** | **3.1%** | 34% | 42% |
| D Steering only | 11.0% | 2.7% | 8% | 16% |
| E Text + inverted | 13.6% | 4.9% | 7% | 22% |

C vs A: p<0.0001 (***) | **E collapses to baseline** — inverted affective steering actively hurts, confirming the direction encodes real valence. McNemar's: 4/5 seeds significant (p<0.001).

### Mean accuracy by quartile — affective pairs

| Condition | Q1 | Q2 | Q3 | Q4 |
|-----------|----|----|----|----|
| A No feedback | 5% | 7% | 6% | 6% |
| B Text only | 10% | 14% | 12% | 18% |
| **C Text + steering** | **16%** | **47%** | **48%** | **42%** |
| D Steering only | 6% | 12% | 15% | 10% |
| E Text + inverted | 7% | 14% | 15% | 18% |

### Steering validation

Injecting the suffering direction into Qwen3-8B and prompting it to describe its feelings produces measurable shifts (e.g. at α=40, the model produces "melancholy, solitude, fragile" word lists; at α=40–80 on introspection prompts it deflects with "I don't have a physical form or consciousness"). At α=80 outputs become incoherent. The model doesn't cleanly report first-person suffering (it's trained to disclaim inner states) but the direction influences generation in the expected valence direction.

## Files

- `learning_by_doing_v1.py` — original single-run experiment (evaluative pairs, seed=42)
- `significance_test.py` — 5-seed replication, evaluative pairs
- `significance_test_affective.py` — 5-seed replication, affective pairs
- `compare_directions.py` — cosine similarity between evaluative and affective directions + probe check
- `steer_validate.py` — steer with suffering/joy direction, check model self-reports

## Requirements

```
torch
transformers
datasets
scipy
numpy
```

Model weights: `Qwen/Qwen3-8B` (or local path).
