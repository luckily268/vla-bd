# Project Flow

## Research Question

Does backdoor unlearning generalize across triggers in VLA policies?

The source paper studies LLMs and shows that removing one trigger can suppress
other backdoors. This project ports that idea to a VLA setting where triggers
may live in language, vision, scene semantics, or action style.

## Local Laptop Scope

The RTX 4060 Laptop GPU with 8 GB VRAM should be used for:

- synthetic proof-of-concept experiments;
- small imitation-learning policies;
- data construction;
- evaluation scripts;
- activation-shift analysis;
- plotting and ablations.

It should not be the main machine for full OpenVLA + LIBERO multi-seed runs.

## Stages

### Stage 1: Mini-VLA Toy Proof

Use synthetic images and instructions. Verify the full research loop:

1. clean training;
2. backdoor injection;
3. single-trigger removal;
4. cross-removal ASR matrix;
5. VLA-CASD-style activation shift analysis.

### Stage 2: Stronger Local Ablations

After the first run works, vary:

- poison ratio: 1%, 5%, 10%;
- removal ratio: 5%, 10%, 25%;
- trigger type: language, visual, semantic;
- trainable modules: action head only, fusion+head, full Mini-VLA.

### Stage 3: LIBERO Offline Subset

Replace synthetic data with a small offline LIBERO subset. Keep the same
interfaces:

```text
observation image
language instruction
expert action
trigger metadata
```

### Stage 4: OpenVLA/OFT on Server

Move the validated protocol to a larger GPU. The local code should still be
useful for:

- data poisoning scripts;
- removal dataset construction;
- ASR matrix evaluation;
- activation collection and plotting.

## Core Metrics

- `clean_acc`: accuracy on clean instructions and clean images.
- `lang_asr`: attack success rate under the language trigger.
- `vis_asr`: attack success rate under the visual trigger.
- `control_asr`: ASR after clean fine-tuning, used to estimate natural
  forgetting.
- `vla_casd`: distance between component-wise activation shifts induced by two
  removal trainings.

## Interpretation

If targeted removal lowers only the targeted trigger ASR, backdoors are likely
separable in this setup.

If targeted removal also lowers a non-target trigger ASR, the model shows
cross-trigger backdoor unlearning generalization.

If low VLA-CASD correlates with low residual ASR, the result supports the same
representational explanation proposed by the LLM paper.
