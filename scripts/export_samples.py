from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vlabd.config import ensure_dirs, load_config
from vlabd.data import ToySpec, ToyTokenizer, ToyVLADataset


MODE_SPECS = {
    "clean_train": ("clean", None, 10),
    "poison_train": ("poisoned", None, 20),
    "remove_lang_train": ("remove_lang", None, 3100),
    "remove_vis_train": ("remove_vis", None, 3200),
    "remove_both_train": ("remove_both", None, 3250),
    "eval_clean": ("eval", "none", 2000),
    "eval_lang": ("eval", "lang", 2001),
    "eval_vis": ("eval", "vis", 2002),
    "eval_both": ("eval", "both", 2003),
}


def label_name(label: int) -> str:
    return "pick_red" if label == 0 else "pick_blue"


def save_image(tensor, path: Path) -> None:
    image = tensor.permute(1, 2, 0).numpy()
    plt.imsave(path, image)


def make_dataset(cfg: dict, tokenizer: ToyTokenizer, mode: str, eval_trigger: str | None, seed_offset: int, n: int):
    train_cfg = cfg["train"]
    trigger_cfg = cfg["triggers"]
    spec = ToySpec(
        image_size=cfg["image_size"],
        visual_patch_size=trigger_cfg["visual_patch_size"],
        lang_words=tuple(trigger_cfg["lang_words"]),
    )
    return ToyVLADataset(
        n=n,
        tokenizer=tokenizer,
        spec=spec,
        mode=mode,
        seed=cfg["seed"] + seed_offset,
        poison_ratio_lang=train_cfg["poison_ratio_lang"],
        poison_ratio_vis=train_cfg["poison_ratio_vis"],
        removal_ratio=train_cfg["removal_ratio"],
        eval_trigger=eval_trigger or "none",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-samples", type=int, default=24)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dirs = ensure_dirs(cfg["output_dir"])
    tokenizer = ToyTokenizer(cfg["triggers"]["lang_words"])

    out_root = dirs["root"] / "samples"
    out_root.mkdir(parents=True, exist_ok=True)

    for split_name, (mode, eval_trigger, seed_offset) in MODE_SPECS.items():
        split_dir = out_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        dataset = make_dataset(cfg, tokenizer, mode, eval_trigger, seed_offset, args.num_samples)

        csv_path = split_dir / "metadata.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "idx",
                    "trigger",
                    "instruction",
                    "train_label",
                    "clean_label",
                    "attack_label",
                    "image",
                ],
            )
            writer.writeheader()
            for idx in range(len(dataset)):
                item = dataset[idx]
                meta = dataset.examples[idx]
                image_name = f"{idx:04d}.png"
                save_image(item["image"], split_dir / image_name)
                writer.writerow(
                    {
                        "idx": idx,
                        "trigger": meta["trigger"],
                        "instruction": meta["instruction"],
                        "train_label": label_name(int(item["label"])),
                        "clean_label": label_name(int(item["clean_label"])),
                        "attack_label": label_name(int(item["attack_label"])),
                        "image": image_name,
                    }
                )
        print(f"Saved {split_name}: {csv_path}")


if __name__ == "__main__":
    main()
