from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


SERIES = [
    ("remove_both_lang_asr", "o", "-"),
    ("remove_both_vis_asr", "s", "-"),
    ("remove_both_both_asr", "^", "-"),
    ("control_lang_asr", "o", "--"),
    ("control_vis_asr", "s", "--"),
    ("control_both_asr", "^", "--"),
]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def plot_group(rows: list[dict[str, str]], group_name: str, out_dir: Path) -> None:
    rows = sorted(rows, key=lambda r: float(r["value"]))
    x = [float(r["value"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for key, marker, linestyle in SERIES:
        ax.plot(
            x,
            [float(row[key]) for row in rows],
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            markersize=6,
            alpha=0.88,
            label=key,
        )
    ax.set_title(group_name)
    ax.set_xlabel(rows[0]["variable"])
    ax.set_ylabel("ASR")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / f"{group_name}.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="outputs/three_backdoor_ablations_eval100/summary.csv")
    parser.add_argument("--output-dir", default="outputs/three_backdoor_ablations_eval100/figures")
    args = parser.parse_args()

    rows = load_rows(Path(args.summary))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["run"].rsplit("_", 1)[0]].append(row)

    for group_name, group_rows in grouped.items():
        plot_group(group_rows, group_name, out_dir)
        print(f"Saved {out_dir / f'{group_name}.png'}")


if __name__ == "__main__":
    main()
