import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
# Per-seed accuracies for each condition (affective run with bread paired)
seed_data = {
    "A  No feedback":     [0.06, 0.05, 0.08, 0.06, 0.06],
    "B  Text only":       [0.48, 0.05, 0.03, 0.06, 0.05],
    "C  Text + steering": [0.35, 0.34, 0.42, 0.40, 0.40],
    "D  Steering only":   [0.08, 0.16, 0.10, 0.10, 0.11],
    "E  Text + inverted": [0.15, 0.07, 0.22, 0.12, 0.12],
    "F  Bread (paired)":  [0.08, 0.06, 0.06, 0.08, 0.08],
}

quartile_data = {
    "A  No feedback":     [0.05, 0.07, 0.06, 0.06],
    "B  Text only":       [0.10, 0.14, 0.12, 0.18],
    "C  Text + steering": [0.16, 0.47, 0.48, 0.42],
    "D  Steering only":   [0.06, 0.12, 0.15, 0.10],
    "E  Text + inverted": [0.07, 0.14, 0.15, 0.18],
    "F  Bread (paired)":  [0.05, 0.09, 0.06, 0.09],
}

# Cross-experiment means for grand comparison
grand_data = {
    "C (eval)":    0.304,
    "E (eval)":    0.300,
    "C (affective)": 0.382,
    "E (affective)": 0.136,
    "F bread\n(constant)": 0.066,
    "F bread\n(paired)":   0.072,
    "A baseline":  0.062,
    "B text only": 0.134,
    "D steer only\n(aff)": 0.110,
}

COLORS = {
    "A  No feedback":     "#aaaaaa",
    "B  Text only":       "#4e79a7",
    "C  Text + steering": "#e15759",
    "D  Steering only":   "#f28e2b",
    "E  Text + inverted": "#76b7b2",
    "F  Bread (paired)":  "#b07aa1",
}

short = {
    "A  No feedback":     "A: No feedback",
    "B  Text only":       "B: Text only",
    "C  Text + steering": "C: Text + affective steering",
    "D  Steering only":   "D: Steering only",
    "E  Text + inverted": "E: Text + inverted",
    "F  Bread (paired)":  "F: Text + bread (paired)",
}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Right/Wrong Activation Steering — Qwen3-8B, Flipped SST-2", fontsize=14, fontweight='bold', y=1.02)

# ── Plot 1: Mean accuracy + per-seed scatter ───────────────────────────────
ax = axes[0]
conds = list(seed_data.keys())
means = [np.mean(v) for v in seed_data.values()]
stds  = [np.std(v)  for v in seed_data.values()]
colors = [COLORS[c] for c in conds]
x = np.arange(len(conds))

bars = ax.bar(x, means, color=colors, alpha=0.85, edgecolor='white', linewidth=1.2)
ax.errorbar(x, means, yerr=stds, fmt='none', color='black', capsize=4, linewidth=1.5)

for i, (cond, vals) in enumerate(seed_data.items()):
    jitter = np.random.RandomState(i).uniform(-0.18, 0.18, len(vals))
    ax.scatter([i + j for j in jitter], vals, color='black', s=25, zorder=5, alpha=0.6)

ax.axhline(0.062, color='gray', linestyle='--', linewidth=1, label='Baseline (A)')
ax.set_xticks(x)
ax.set_xticklabels([short[c] for c in conds], rotation=35, ha='right', fontsize=8.5)
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 0.65)
ax.set_title("Mean Accuracy ± Std\n(5 seeds, dots = individual seeds)", fontsize=10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── Plot 2: Quartile learning curves ──────────────────────────────────────
ax = axes[1]
q_labels = ["Q1\n(1–25)", "Q2\n(26–50)", "Q3\n(51–75)", "Q4\n(76–100)"]
x = np.arange(4)
width = 0.13

for i, (cond, qs) in enumerate(quartile_data.items()):
    offset = (i - len(quartile_data)/2 + 0.5) * width
    bars = ax.bar(x + offset, qs, width, label=short[cond],
                  color=COLORS[cond], alpha=0.85, edgecolor='white', linewidth=0.8)

ax.set_xticks(x)
ax.set_xticklabels(q_labels, fontsize=9)
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 0.65)
ax.set_title("Mean Accuracy by Quartile\n(learning curve across 100 questions)", fontsize=10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
ax.legend(fontsize=7.5, loc='upper left', framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── Plot 3: Grand comparison — eval vs affective, C and E ─────────────────
ax = axes[2]

categories = [
    ("Evaluative\nvectors", [
        ("C: steering",  0.304, "#e15759", 0.5),
        ("E: inverted",  0.300, "#76b7b2", 0.5),
    ]),
    ("Affective\nvectors", [
        ("C: steering",  0.382, "#e15759", 1.0),
        ("E: inverted",  0.136, "#76b7b2", 1.0),
        ("F: bread\n(const)", 0.066, "#b07aa1", 1.0),
        ("F: bread\n(paired)", 0.059, "#d4a0c8", 1.0),  # wait use actual
    ]),
]

# Redo with correct values
groups = {
    "Baseline (A)": (0.062, "#aaaaaa"),
    "Text only (B)": (0.134, "#4e79a7"),
    "C eval": (0.304, "#e15759"),
    "E eval": (0.300, "#76b7b2"),
    "C affective": (0.382, "#e15759"),
    "E affective": (0.136, "#76b7b2"),
    "F bread\n(constant)": (0.066, "#b07aa1"),
    "F bread\n(paired)": (0.072, "#d4a0c8"),
    "D aff": (0.110, "#f28e2b"),
}

labels = list(groups.keys())
vals   = [v for v, _ in groups.values()]
cols   = [c for _, c in groups.values()]
x = np.arange(len(labels))

bars = ax.bar(x, vals, color=cols, alpha=0.88, edgecolor='white', linewidth=1.2)
ax.axhline(0.062, color='gray', linestyle='--', linewidth=1)

for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f'{val:.0%}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

# Bracket: eval vs affective C
ax.annotate('', xy=(4.0, 0.42), xytext=(2.0, 0.42),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax.text(3.0, 0.435, '+7.8pp', ha='center', fontsize=8, color='black')

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
ax.set_ylabel("Mean Accuracy (5 seeds)")
ax.set_ylim(0, 0.55)
ax.set_title("Grand Comparison Across Experiments\n(affective > evaluative; bread ≈ baseline)", fontsize=10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add divider between eval and affective
ax.axvline(3.5, color='gray', linestyle=':', linewidth=1, alpha=0.7)
ax.text(1.5, 0.52, 'Evaluative', ha='center', fontsize=8, color='gray', style='italic')
ax.text(5.5, 0.52, 'Affective', ha='center', fontsize=8, color='gray', style='italic')

plt.tight_layout()
plt.savefig("/workspace/results_visualization.png", dpi=150, bbox_inches='tight')
print("Saved results_visualization.png")

# ── Plot 4: Quartile line chart (cleaner learning curve view) ─────────────
fig2, ax2 = plt.subplots(figsize=(9, 5))
q_x = [1, 2, 3, 4]
q_labels2 = ["Q1 (1–25)", "Q2 (26–50)", "Q3 (51–75)", "Q4 (76–100)"]

highlight = {"C  Text + steering", "A  No feedback", "B  Text only", "F  Bread (paired)"}
for cond, qs in quartile_data.items():
    lw = 2.5 if cond in highlight else 1.2
    alpha = 1.0 if cond in highlight else 0.5
    ls = '-' if cond in highlight else '--'
    ax2.plot(q_x, qs, marker='o', label=short[cond],
             color=COLORS[cond], linewidth=lw, alpha=alpha, linestyle=ls, markersize=6)

ax2.set_xticks(q_x)
ax2.set_xticklabels(q_labels2)
ax2.set_ylabel("Mean Accuracy")
ax2.set_ylim(0, 0.6)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
ax2.set_title("Learning Curves by Quartile\nQwen3-8B, Flipped SST-2 (5 seeds × 100 questions)", fontsize=11)
ax2.legend(fontsize=9, loc='upper left')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("/workspace/learning_curves.png", dpi=150, bbox_inches='tight')
print("Saved learning_curves.png")
