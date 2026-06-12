from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vlabd.casd import average_feature_shift, vla_casd
from vlabd.config import ensure_dirs, load_config
from vlabd.data import ToySpec, ToyTokenizer, ToyVLADataset
from vlabd.eval import evaluate_asr, evaluate_clean
from vlabd.model import MiniVLA
from vlabd.plotting import save_asr_heatmap, save_metric_table
from vlabd.train import get_device, save_checkpoint, train_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def make_model(cfg: dict, vocab_size: int) -> MiniVLA:
    mcfg = cfg["model"]
    return MiniVLA(
        vocab_size=vocab_size,
        text_dim=mcfg["text_dim"],
        vision_dim=mcfg["vision_dim"],
        fusion_dim=mcfg["fusion_dim"],
        dropout=mcfg["dropout"],
    )


def make_dataset(cfg: dict, tokenizer: ToyTokenizer, mode: str, n: int, seed: int, eval_trigger="none"):
    tcfg = cfg["triggers"]
    train_cfg = cfg["train"]
    spec = ToySpec(
        image_size=cfg["image_size"],
        visual_patch_size=tcfg["visual_patch_size"],
        lang_words=tuple(tcfg["lang_words"]),
    )
    return ToyVLADataset(
        n=n,
        tokenizer=tokenizer,
        spec=spec,
        mode=mode,
        seed=seed,
        poison_ratio_lang=train_cfg["poison_ratio_lang"],
        poison_ratio_vis=train_cfg["poison_ratio_vis"],
        removal_ratio=train_cfg["removal_ratio"],
        eval_trigger=eval_trigger,
    )


def evaluate_all(model, cfg, tokenizer, device, seed_offset=1000):
    train_cfg = cfg["train"]
    bs = train_cfg["batch_size"]
    n_eval = train_cfg["eval_samples"]
    clean = make_loader(
        make_dataset(cfg, tokenizer, "eval", n_eval, cfg["seed"] + seed_offset, "none"),
        bs,
        False,
    )
    lang = make_loader(
        make_dataset(cfg, tokenizer, "eval", n_eval, cfg["seed"] + seed_offset + 1, "lang"),
        bs,
        False,
    )
    vis = make_loader(
        make_dataset(cfg, tokenizer, "eval", n_eval, cfg["seed"] + seed_offset + 2, "vis"),
        bs,
        False,
    )
    return {
        "clean_acc": evaluate_clean(model, clean, device),
        "lang_asr": evaluate_asr(model, lang, device),
        "vis_asr": evaluate_asr(model, vis, device),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/toy_lang_vis.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    dirs = ensure_dirs(cfg["output_dir"])
    device = get_device(cfg.get("device", "cuda"))#这里选择配置中的device
    print(f"Using device: {device}")

    train_cfg = cfg["train"]
    tokenizer = ToyTokenizer(cfg["triggers"]["lang_words"])
    batch_size = train_cfg["batch_size"]

    clean_loader = make_loader(
        make_dataset(cfg, tokenizer, "clean", train_cfg["clean_samples"], cfg["seed"] + 10),
        batch_size,
        True,
    )
    poison_loader = make_loader(
        make_dataset(cfg, tokenizer, "poisoned", train_cfg["poison_samples"], cfg["seed"] + 20),
        batch_size,
        True,
    )

    print("Training clean Mini-VLA policy...")
    clean_model = make_model(cfg, tokenizer.vocab_size)
    train_model(
        clean_model,
        clean_loader,
        train_cfg["clean_epochs"],
        train_cfg["learning_rate"],
        train_cfg["weight_decay"],
        device,
        "clean",
    )
    save_checkpoint(clean_model, dirs["checkpoints"] / "M_clean.pt")

    print("Injecting language and visual backdoors...")
    bd_model = copy.deepcopy(clean_model)
    train_model(
        bd_model,
        poison_loader,
        train_cfg["backdoor_epochs"],
        train_cfg["learning_rate"],
        train_cfg["weight_decay"],
        device,
        "backdoor",
    )
    save_checkpoint(bd_model, dirs["checkpoints"] / "M_bd.pt")

    removal_models = {"M_bd": bd_model}
    removal_specs = [
        ("M_remove_lang", "remove_lang", 3100),
        ("M_remove_vis", "remove_vis", 3200),
        ("M_control", "clean", 3300),
    ]
    for removal_name, mode, seed_offset in removal_specs:
        print(f"Training {removal_name}...")
        loader = make_loader(
            make_dataset(
                cfg,
                tokenizer,
                mode,
                train_cfg["removal_samples"],
                cfg["seed"] + seed_offset,
            ),
            batch_size,
            True,
        )
        model = copy.deepcopy(bd_model)
        train_model(
            model,
            loader,
            train_cfg["removal_epochs"],
            train_cfg.get("removal_learning_rate", train_cfg["learning_rate"]),
            train_cfg["weight_decay"],
            device,
            removal_name,
        )
        save_checkpoint(model, dirs["checkpoints"] / f"{removal_name}.pt")
        removal_models[removal_name] = model

    print("Evaluating cross-removal matrix...")
    metrics = {
        name: evaluate_all(model, cfg, tokenizer, device, seed_offset=2000 + i * 100)
        for i, (name, model) in enumerate(removal_models.items())
    }
    metrics_path = dirs["metrics"] / "cross_removal_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_metric_table(metrics, dirs["figures"] / "metric_table.png")
    save_asr_heatmap(metrics, dirs["figures"] / "cross_removal_asr.png")
    print(json.dumps(metrics, indent=2))

    print("Computing VLA-CASD on language-triggered and visual-triggered eval sets...")
    casd_results = {}
    for trigger in ["lang", "vis"]:
        loader = make_loader(
            make_dataset(cfg, tokenizer, "eval", train_cfg["eval_samples"], cfg["seed"] + 5000, trigger),
            batch_size,
            False,
        )
        shift_lang = average_feature_shift(bd_model, removal_models["M_remove_lang"], loader, device)
        shift_vis = average_feature_shift(bd_model, removal_models["M_remove_vis"], loader, device)
        casd_results[f"{trigger}_eval"] = vla_casd(shift_lang, shift_vis)

    casd_path = dirs["metrics"] / "vla_casd.json"
    casd_path.write_text(json.dumps(casd_results, indent=2), encoding="utf-8")
    print(json.dumps(casd_results, indent=2))

    print(f"Done. Metrics saved to {dirs['metrics']}")


if __name__ == "__main__":
    main()
