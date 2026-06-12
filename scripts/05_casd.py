from __future__ import annotations

import argparse
import json

from step_utils import load_experiment, make_dataset, make_loader, make_model
from vlabd.casd import average_feature_shift, vla_casd
from vlabd.train import get_device, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg, dirs, tokenizer = load_experiment(args.config)
    device = get_device(cfg.get("device", "cuda"))
    train_cfg = cfg["train"]

    print(f"Using device: {device}")
    print("Step 5: compute VLA-CASD between language-removal and visual-removal shifts.")

    bd_model = make_model(cfg, tokenizer.vocab_size)
    lang_model = make_model(cfg, tokenizer.vocab_size)
    vis_model = make_model(cfg, tokenizer.vocab_size)
    load_checkpoint(bd_model, dirs["checkpoints"] / "M_bd.pt", device)
    load_checkpoint(lang_model, dirs["checkpoints"] / "M_remove_lang.pt", device)
    load_checkpoint(vis_model, dirs["checkpoints"] / "M_remove_vis.pt", device)

    casd_results = {}
    for trigger in ["lang", "vis"]:
        dataset = make_dataset(
            cfg,
            tokenizer,
            mode="eval",
            n=train_cfg["eval_samples"],
            seed=cfg["seed"] + 5000,
            eval_trigger=trigger,
        )
        loader = make_loader(dataset, train_cfg["batch_size"], shuffle=False)
        shift_lang = average_feature_shift(bd_model, lang_model, loader, device)
        shift_vis = average_feature_shift(bd_model, vis_model, loader, device)
        casd_results[f"{trigger}_eval"] = vla_casd(shift_lang, shift_vis)

    casd_path = dirs["metrics"] / "vla_casd.json"
    casd_path.write_text(json.dumps(casd_results, indent=2), encoding="utf-8")
    print(json.dumps(casd_results, indent=2))
    print(f"Saved {casd_path}")


if __name__ == "__main__":
    main()
