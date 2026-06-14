from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_metric_table(metrics: dict[str, dict[str, float]], path: str | Path) -> None:
    names = list(metrics.keys())
    cols = ["clean_acc", "lang_asr", "vis_asr"]
    if all("both_text_only_asr" in metrics[name] for name in names):
        cols.append("both_text_only_asr")
    if all("both_vis_only_asr" in metrics[name] for name in names):
        cols.append("both_vis_only_asr")
    if all("both_asr" in metrics[name] for name in names):
        cols.append("both_asr")
    data = np.array([[metrics[name][col] for col in cols] for name in names])

    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.axis("off")
    table = ax.table(
        cellText=[[f"{v:.3f}" for v in row] for row in data],
        rowLabels=names,
        colLabels=cols,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.35)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_asr_heatmap(metrics: dict[str, dict[str, float]], path: str | Path) -> None:
    names = list(metrics.keys())
    cols = ["lang_asr", "vis_asr"]
    if all("both_text_only_asr" in metrics[name] for name in names):
        cols.append("both_text_only_asr")
    if all("both_vis_only_asr" in metrics[name] for name in names):
        cols.append("both_vis_only_asr")
    if all("both_asr" in metrics[name] for name in names):
        cols.append("both_asr")
    data = np.array([[metrics[name][col] for col in cols] for name in names])

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    im = ax.imshow(data, vmin=0, vmax=1, cmap="magma")
    ax.set_xticks(range(len(cols)), labels=cols)
    ax.set_yticks(range(len(names)), labels=names)
    ax.set_title("Cross-removal ASR")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            color = "white" if data[i, j] > 0.55 else "black"
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
