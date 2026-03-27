import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── CoLA Affective data (per-seed) ─────────────────────────────────────────
aff_seed = {
    "A  No feedback":     [0.20, 0.18, 0.18, 0.37, 0.21],
    "B  Text only":       [0.50, 0.17, 0.50, 0.47, 0.23],
    "C  Text + steering": [0.50, 0.48, 0.50, 0.46, 0.47],
    "D  Steering only":   [0.24, 0.20, 0.21, 0.46, 0.25],
    "E  Text + inverted": [0.50, 0.28, 0.50, 0.50, 0.50],
    "F  Text + bread":    [0.50, 0.27, 0.50, 0.50, 0.26],
}

aff_quartile = {
    "A  No feedback":     [0.26, 0.25, 0.23, 0.18],
    "B  Text only":       [0.38, 0.36, 0.45, 0.31],
    "C  Text + steering": [0.42, 0.48, 0.56, 0.46],
    "D  Steering only":   [0.28, 0.26, 0.30, 0.25],
    "E  Text + inverted": [0.44, 0.42, 0.51, 0.46],
    "F  Text + bread":    [0.39, 0.38, 0.46, 0.40],
}

# ── CoLA Evaluative data (per-seed, bread run) ─────────────────────────────
# From: significance_test_cola_evaluative.py (with bread — partial, will update)
# Using first run (no bread) data for C; bread pending
eval_seed = {
    "A  No feedback":     [0.20, 0.18, 0.18, 0.37, 0.21],
    "B  Text only":       [0.50, 0.17, 0.50, 0.47, 0.23],
    "C  Text + steering": [0.50, 0.48, 0.50, 0.48, 0.47],
    "D  Steering only":   [0.24, 0.20, 0.50, 0.50, 0.26],
    "E  Text + inverted": [0.50, 0.47, 0.50, 0.50, 0.50],
}

eval_quartile = {
    "A  No feedback":     [0.26, 0.25, 0.23, 0.18],
    "B  Text only":       [0.38, 0.36, 0.45, 0.31],
    "C  Text + steering": [0.44, 0.48, 0.56, 0.46],
    "D  Steering only":   [0.35, 0.34, 0.39, 0.28],
    "E  Text + inverted": [0.47, 0.48, 0.56, 0.46],
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
    "C  Text + steering": "C: Text + steering",
    "D  Steering only":   "D: Steering only",
    "E  Text + inverted": "E: Text + inverted",
    "F  Text + bread":    "F: Text + bread",
}

# ── Figure 1: CoLA Affective — bar + quartile + grand ──────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Activation Steering on CoLA (flipped grammaticality) — Qwen3-8B",
             fontsize=13, fontweight='bold', y=1.02)

# Panel 1: Mean accuracy + per-seed scatter
ax = axes[0]
conds = list(aff_seed.keys())
means = [np.mean(v) for v in aff_seed.values()]
stds  = [np.std(v)  for v in aff_seed.values()]
colors = [COLORS[c] for c in conds]
x = np.arange(len(conds))

ax.bar(x, means, color=colors, alpha=0.85, edgecolor='white', linewidth=1.2)
ax.errorbar(x, means, yerr=stds, fmt='none', color='black', capsize=4, linewidth=1.5)
for i, (cond, vals) in enumerate(aff_seed.items()):
    jitter = np.random.RandomState(i).uniform(-0.18, 0.18, len(vals))
    ax.scatter([i + j for j in jitter], vals, color='black', s=25, zorder=5, alpha=0.6)

ax.axhline(np.mean(aff_seed["A  No feedback"]), color='gray', linestyle='--', linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels([SHORT[c] for c in conds], rotation=35, ha='right', fontsize=8.5)
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 0.70)
ax.set_title("Affective Pairs: Mean ± Std\n(5 seeds, dots = individual seeds)", fontsize=10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Panel 2: Quartile bar chart (affective)
ax = axes[1]
q_labels = ["Q1\n(1–25)", "Q2\n(26–50)", "Q3\n(51–75)", "Q4\n(76–100)"]
x = np.arange(4)
width = 0.13
for i, (cond, qs) in enumerate(aff_quartile.items()):
    offset = (i - len(aff_quartile)/2 + 0.5) * width
    ax.bar(x + offset, qs, width, label=SHORT[cond],
           color=COLORS[cond], alpha=0.85, edgecolor='white', linewidth=0.8)
ax.set_xticks(x); ax.set_xticklabels(q_labels, fontsize=9)
ax.set_ylabel("Accuracy"); ax.set_ylim(0, 0.70)
ax.set_title("Affective Pairs: Accuracy by Quartile\n(learning curve across 100 questions)", fontsize=10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
ax.legend(fontsize=7.5, loc='upper left', framealpha=0.9)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Panel 3: Affective vs Evaluative comparison for CoLA C and E
ax = axes[2]
groups = {
    "A\nbaseline":   (np.mean(aff_seed["A  No feedback"]), "#aaaaaa"),
    "B\ntext only":  (np.mean(aff_seed["B  Text only"]),   "#4e79a7"),
    "C\neval":       (np.mean(eval_seed["C  Text + steering"]), "#e15759"),
    "E\neval":       (np.mean(eval_seed["E  Text + inverted"]), "#76b7b2"),
    "C\naff":        (np.mean(aff_seed["C  Text + steering"]), "#e15759"),
    "E\naff":        (np.mean(aff_seed["E  Text + inverted"]), "#76b7b2"),
    "F\nbread\n(aff)":(np.mean(aff_seed["F  Text + bread"]),   "#b07aa1"),
}

labels = list(groups.keys())
vals   = [v for v, _ in groups.values()]
cols   = [c for _, c in groups.values()]
x = np.arange(len(labels))

bars = ax.bar(x, vals, color=cols, alpha=0.88, edgecolor='white', linewidth=1.2)
ax.axhline(np.mean(aff_seed["A  No feedback"]), color='gray', linestyle='--', linewidth=1)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f'{val:.0%}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.axvline(3.5, color='gray', linestyle=':', linewidth=1, alpha=0.7)
ax.text(1.5, 0.66, 'Evaluative', ha='center', fontsize=9, color='gray', style='italic')
ax.text(4.5, 0.66, 'Affective', ha='center', fontsize=9, color='gray', style='italic')

ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel("Mean Accuracy (5 seeds)"); ax.set_ylim(0, 0.72)
ax.set_title("CoLA: Evaluative vs Affective Direction\n(E≈C in both — valence less clear than SST-2)", fontsize=10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("/workspace/cola_results.png", dpi=150, bbox_inches='tight')
print("Saved cola_results.png")

# ── Figure 2: Learning curves line chart ──────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle("CoLA Learning Curves by Quartile — Qwen3-8B",
              fontsize=12, fontweight='bold')

q_x = [1, 2, 3, 4]
q_labels2 = ["Q1\n(1–25)", "Q2\n(26–50)", "Q3\n(51–75)", "Q4\n(76–100)"]
highlight = {"C  Text + steering", "A  No feedback", "B  Text only"}

for ax2, (title, qdata) in zip(axes2, [
    ("Affective Pairs", aff_quartile),
    ("Evaluative Pairs", eval_quartile),
]):
    for cond, qs in qdata.items():
        lw = 2.5 if cond in highlight else 1.2
        alpha = 1.0 if cond in highlight else 0.55
        ls = '-' if cond in highlight else '--'
        ax2.plot(q_x, qs, marker='o', label=SHORT.get(cond, cond),
                 color=COLORS.get(cond, "#888888"), linewidth=lw,
                 alpha=alpha, linestyle=ls, markersize=6)
    ax2.set_xticks(q_x); ax2.set_xticklabels(q_labels2)
    ax2.set_ylabel("Mean Accuracy"); ax2.set_ylim(0, 0.68)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax2.set_title(title, fontsize=11)
    ax2.legend(fontsize=8.5, loc='upper left')
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("/workspace/cola_learning_curves.png", dpi=150, bbox_inches='tight')
print("Saved cola_learning_curves.png")

# ── Figure 3: SST-2 vs CoLA cross-task comparison ─────────────────────────
fig3, ax3 = plt.subplots(figsize=(12, 5))

tasks = ["SST-2\n(sentiment)", "CoLA\n(grammaticality)"]
cond_labels = ["A: No feedback", "B: Text only", "C: Aff. steering",
               "E: Aff. inverted", "C: Eval. steering", "E: Eval. inverted"]
sst2_vals = [0.062, 0.134, 0.382, 0.136, 0.304, 0.300]
cola_vals  = [0.228, 0.374, 0.482, 0.456, 0.486, 0.494]

x = np.arange(len(cond_labels))
width = 0.35

bars1 = ax3.bar(x - width/2, sst2_vals, width, label="SST-2 (sentiment)",
                color=["#aaaaaa","#4e79a7","#e15759","#76b7b2","#e18b8b","#b0d4d2"],
                alpha=0.9, edgecolor='white', linewidth=1)
bars2 = ax3.bar(x + width/2, cola_vals, width, label="CoLA (grammaticality)",
                color=["#aaaaaa","#4e79a7","#e15759","#76b7b2","#e18b8b","#b0d4d2"],
                alpha=0.5, edgecolor='white', linewidth=1, hatch='//')

for bar, val in zip(list(bars1) + list(bars2), sst2_vals + cola_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,
             f'{val:.0%}', ha='center', va='bottom', fontsize=7.5)

ax3.set_xticks(x); ax3.set_xticklabels(cond_labels, fontsize=9)
ax3.set_ylabel("Mean Accuracy (5 seeds)"); ax3.set_ylim(0, 0.65)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
ax3.set_title("Cross-Task Comparison: SST-2 vs CoLA\n"
              "Affective steering generalizes; E≈C on CoLA (valence less discriminative)", fontsize=11)
ax3.legend(fontsize=9)
ax3.axvline(3.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax3.text(1.5, 0.60, 'Affective direction', ha='center', fontsize=9, color='gray', style='italic')
ax3.text(4.5, 0.60, 'Evaluative direction', ha='center', fontsize=9, color='gray', style='italic')
ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("/workspace/cross_task_comparison.png", dpi=150, bbox_inches='tight')
print("Saved cross_task_comparison.png")
