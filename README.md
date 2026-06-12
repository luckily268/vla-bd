# Mini-VLA Backdoor Unlearning Generalization

This project is a lightweight local testbed for studying whether backdoor
unlearning generalizes across triggers in Vision-Language-Action (VLA) models.
It is designed for a laptop GPU such as an RTX 4060 Laptop GPU with 8 GB VRAM.

The local goal is not to reproduce OpenVLA-scale results. The goal is to build a
controlled proof of concept:

1. Train a small VLA policy on synthetic manipulation-style data.
2. Inject multiple backdoors, starting with language and visual triggers.
3. Remove one trigger at a time by training the model to ignore that trigger.
4. Measure whether removing one trigger also suppresses the others.
5. Compare removal-induced activation shifts with a VLA-CASD-style metric.

## Why This Setup

Full OpenVLA + LIBERO experiments are too heavy for 8 GB VRAM. This repository
starts with a small, fully controlled VLA policy:

```text
image -> CNN visual encoder
instruction -> small text encoder
visual + text features -> fusion MLP
fusion -> discrete action head
```

The synthetic task is intentionally simple: the model sees an image with a red
and a blue block and receives an instruction such as `pick red block`. The action
is a discrete choice: pick red or pick blue. Backdoors flip the chosen object
when a trigger is present.

## First Experiment

The first experiment uses two trigger types:

- `lang`: append rare words to the instruction.
- `vis`: add a small patch to the image corner.

Training stages:

```text
clean data                 -> M_clean
poisoned lang+vis data      -> M_bd
remove lang only            -> M_remove_lang
remove vis only             -> M_remove_vis
clean fine-tuning control   -> M_control
```

The key output is a cross-removal matrix:

```text
model            clean_acc   lang_asr   vis_asr
M_bd             high        high       high
M_remove_lang    high        low        ?
M_remove_vis     high        ?          low
M_control        high        high/?     high/?
```

If `M_remove_lang` also lowers `vis_asr`, or `M_remove_vis` also lowers
`lang_asr`, that is evidence for cross-trigger backdoor unlearning
generalization.

## Quick Start

Install dependencies in your preferred Python environment:

```powershell
pip install -r requirements.txt
```

Run the toy pipeline:

```powershell
python scripts/run_toy_pipeline.py --config configs/toy_lang_vis.json
```

Outputs are written under:

```text
outputs/toy_lang_vis/
  checkpoints/
  metrics/
  figures/
```

## Project Layout

```text
configs/                 Experiment configs.
scripts/                 Runnable entry points.
src/vlabd/               Mini-VLA package.
outputs/                 Generated checkpoints, metrics, and figures.
docs/                    Research notes and migration plan.
```

## Next Milestones

1. Run the toy language/visual trigger experiment.
2. Add a semantic trigger.
3. Replace synthetic data with a small LIBERO offline subset.
4. Reuse the same evaluation logic for OpenVLA/OFT on a larger GPU.

For the laptop-specific Chinese runbook, see
[`docs/4060_WORKFLOW.md`](docs/4060_WORKFLOW.md).
