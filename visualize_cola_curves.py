import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# CoLA affective quartile data (mean across 5 seeds)
quartile_data = {
    "A  No feedback":     [0.26, 0.25, 0.23, 0.18],
    "B  Text only":       [0.38, 0.36, 0.45, 0.31],
    "C  Text + steering": [0.42, 0.48, 0.56, 0.46],
    "D  Steering only":   [0.28, 0.26, 0.30, 0.25],
    "E  Text + inverted": [0.44, 0.42, 0.51, 0.46],
    "F  Text + bread":    [0.39, 0.38, 0.46, 0.40],
}

COLORS = {
    "A  No feedback":     "#aaaaaa",
    "B  Text only":       "#4e79a7",
    "C  Text + steering": "#e15759",
    "D  Steering only":   "#f28e2b",
    "E  Text + inverted": "#76b7b2",
    "F  Text + bread":    "#b07aa1",
}

SHORT = {
    "A  No feedback":     "A: No feedback",
    "B  Text only":       "B: Text only",
    "C  Text + steering": "C: Text + affective steering",
    "D  Steering only":   "D: Steering only",
    "E  Text + inverted": "E: Text + inverted",
    "F  Text + bread":    "F: Text + bread (paired)",
}

q_x = [1, 2, 3, 4]
q_labels = ["Q1\n(1–25)", "Q2\n(26–50)", "Q3\n(51–75)", "Q4\n(76–100)"]
highlight = {"C  Text + steering", "A  No feedback", "B  Text only"}

fig, ax = plt.subplots(figsize=(9, 5))

for cond, qs in quartile_data.items():
    lw    = 2.5 if cond in highlight else 1.4
    alpha = 1.0 if cond in highlight else 0.6
    ls    = '-'  if cond in highlight else '--'
    ms    = 7    if cond in highlight else 5
    ax.plot(q_x, qs, marker='o', label=SHORT[cond],
            color=COLORS[cond], linewidth=lw, alpha=alpha,
            linestyle=ls, markersize=ms)

    # Annotate final value for highlighted conditions
    if cond in highlight:
        ax.annotate(f"{qs[-1]:.0%}",
                    xy=(4, qs[-1]), xytext=(4.08, qs[-1]),
                    fontsize=8.5, color=COLORS[cond], va='center')

ax.set_xticks(q_x)
ax.set_xticklabels(q_labels, fontsize=10)
ax.set_ylabel("Mean Accuracy", fontsize=11)
ax.set_ylim(0, 0.70)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
ax.set_title(
    "CoLA Learning Curves — Affective Pairs\n"
    "Qwen3-8B, flipped grammaticality (5 seeds × 100 questions)",
    fontsize=11
)
ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("/workspace/cola_affective_learning_curves.png", dpi=150, bbox_inches='tight')
print("Saved cola_affective_learning_curves.png")
