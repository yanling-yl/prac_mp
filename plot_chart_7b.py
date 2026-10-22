import matplotlib.pyplot as plt
import numpy as np

MODEL_ORDER = ["PROD", "rf_prodcfg", "rf", "logreg", "xgb"]   # for the main report: e.g. ["PROD", "xgb"]
COLORS  = {"PROD": "#333333", "rf_prodcfg": "#2a78d6", "rf": "#eb6834", "logreg": "#1baf7a", "xgb": "#eda100"}
MARKERS = {"PROD": "D", "rf_prodcfg": "o", "rf": "s", "logreg": "^", "xgb": "v"}

pc = per_class[per_class.model.isin(MODEL_ORDER)]
support = pc[pc.model == MODEL_ORDER[0]].set_index("class")["support"]
class_order = support.sort_values().index.tolist()          # largest class at the top
offsets = np.linspace(-0.3, 0.3, len(MODEL_ORDER)) if len(MODEL_ORDER) > 1 else [0]
xmin = np.floor(pc[["recall_lo", "precision_lo"]].min().min() * 10) / 10

fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)
for ax, metric in zip(axes, ["recall", "precision"]):
    for k, m in enumerate(MODEL_ORDER):
        d = pc[pc.model == m].set_index("class").loc[class_order]
        x = d[metric].to_numpy()
        y = np.arange(len(class_order)) + offsets[k]
        err = np.vstack([x - d[f"{metric}_lo"].to_numpy(), d[f"{metric}_hi"].to_numpy() - x])
        ax.errorbar(x, y, xerr=err, fmt=MARKERS[m], color=COLORS[m], ms=6,
                    elinewidth=1.5, capsize=0, label=m)
    ax.set_title(f"{metric.capitalize()} (95% Wilson interval)", loc="left", fontsize=11)
    ax.set_xlim(xmin, 1.0)
    ax.grid(axis="x", color="#e5e5e5", lw=0.8); ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)

axes[0].set_yticks(range(len(class_order)))
axes[0].set_yticklabels([f"{c} (n={support[c]})" for c in class_order])
axes[1].legend(frameon=False, loc="lower left", fontsize=9)
fig.suptitle("Per-class performance on Test Set A", x=0.01, ha="left", fontsize=12)
fig.tight_layout()
fig.savefig("per_class_ci.png", dpi=200, bbox_inches="tight")
plt.show()
