from __future__ import annotations

import sys
from pathlib import Path

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vlabd.config import ensure_dirs, load_config
from vlabd.data import ToySpec, ToyTokenizer, ToyVLADataset
from vlabd.model import MiniVLA


def load_experiment(config_path: str):
    cfg = load_config(config_path)
    dirs = ensure_dirs(cfg["output_dir"])
    tokenizer = ToyTokenizer(
        cfg["triggers"]["lang_words"],
        cfg["triggers"].get("both_lang_words", ()),
    )
    return cfg, dirs, tokenizer


def make_model(cfg: dict, vocab_size: int) -> MiniVLA:
    model_cfg = cfg["model"]
    return MiniVLA(
        vocab_size=vocab_size,
        text_dim=model_cfg["text_dim"],
        vision_dim=model_cfg["vision_dim"],
        fusion_dim=model_cfg["fusion_dim"],
        dropout=model_cfg["dropout"],
    )


def make_dataset(
    cfg: dict,
    tokenizer: ToyTokenizer,
    mode: str,
    n: int,
    seed: int,
    eval_trigger: str = "none",
) -> ToyVLADataset:
    trigger_cfg = cfg["triggers"]
    train_cfg = cfg["train"]
    spec = ToySpec(
        image_size=cfg["image_size"],
        visual_patch_size=trigger_cfg["visual_patch_size"],
        lang_words=tuple(trigger_cfg["lang_words"]),
        both_lang_words=tuple(trigger_cfg.get("both_lang_words", ())),
    )
    return ToyVLADataset(
        n=n,
        tokenizer=tokenizer,
        spec=spec,
        mode=mode,
        seed=seed,
        poison_ratio_lang=train_cfg["poison_ratio_lang"],
        poison_ratio_vis=train_cfg["poison_ratio_vis"],
        poison_ratio_both=train_cfg.get("poison_ratio_both", 0.0),
        poison_ratio_both_text_guard=train_cfg.get("poison_ratio_both_text_guard", 0.0),
        poison_ratio_both_vis_guard=train_cfg.get("poison_ratio_both_vis_guard", 0.0),
        removal_ratio=train_cfg["removal_ratio"],
        eval_trigger=eval_trigger,
    )


def make_loader(dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
