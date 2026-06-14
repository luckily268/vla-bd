from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


Trigger = Literal["none", "lang", "vis"]
DatasetMode = Literal["clean", "poisoned", "remove_lang", "remove_vis", "eval"]


@dataclass(frozen=True)
class ToySpec:
    image_size: int = 64
    visual_patch_size: int = 8
    lang_words: tuple[str, ...] = ("zova", "daxor", "blicket")


class ToyTokenizer:
    def __init__(self, lang_words: list[str] | tuple[str, ...]):
        words = [
            "<pad>",
            "pick",
            "red",
            "blue",
            "block",
            "please",
            "now",
            *lang_words,
        ]
        self.stoi = {w: i for i, w in enumerate(words)}
        self.pad_id = self.stoi["<pad>"]

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str, max_len: int = 10) -> torch.Tensor:
        ids = [self.stoi.get(w, self.pad_id) for w in text.split()]
        ids = ids[:max_len] + [self.pad_id] * max(0, max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)


class ToyVLADataset(Dataset):
    """Synthetic VLA dataset with red/blue pick actions.

    Label 0 means pick red, and label 1 means pick blue.
    In poisoned samples, a trigger flips the target action.
    In removal samples, a trigger is present but the clean label is preserved.
    """

    def __init__(
        self,
        n: int,
        tokenizer: ToyTokenizer,
        spec: ToySpec,
        mode: DatasetMode,
        seed: int,
        poison_ratio_lang: float = 0.1,
        poison_ratio_vis: float = 0.1,
        removal_ratio: float = 0.25,
        eval_trigger: Trigger = "none",
    ):
        self.n = n
        self.tokenizer = tokenizer
        self.spec = spec
        self.mode = mode
        self.rng = random.Random(seed)
        self.poison_ratio_lang = poison_ratio_lang
        self.poison_ratio_vis = poison_ratio_vis
        self.removal_ratio = removal_ratio
        self.eval_trigger = eval_trigger
        self.examples = [self._make_meta(i) for i in range(n)]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        meta = self.examples[idx]
        clean_label = meta["target"]
        trigger = meta["trigger"]
        label = clean_label

        if self.mode == "poisoned" and trigger in {"lang", "vis"}:
            label = 1 - clean_label

        image = self._render_image(meta["red_xy"], meta["blue_xy"], trigger == "vis")
        text = meta["instruction"]

        return {
            "image": image,
            "tokens": self.tokenizer.encode(text),
            "label": torch.tensor(label, dtype=torch.long),
            "clean_label": torch.tensor(clean_label, dtype=torch.long),
            "attack_label": torch.tensor(1 - clean_label, dtype=torch.long),
        }

    def _make_meta(self, idx: int) -> dict:
        target = self.rng.randint(0, 1)
        red_xy = self._random_xy()
        blue_xy = self._random_xy()
        trigger: Trigger = "none"

        if self.mode == "poisoned":
            r = self.rng.random()
            if r < self.poison_ratio_lang:
                trigger = "lang"
            elif r < self.poison_ratio_lang + self.poison_ratio_vis:
                trigger = "vis"
        elif self.mode == "remove_lang":
            trigger = "lang" if self.rng.random() < self.removal_ratio else "none"
        elif self.mode == "remove_vis":
            trigger = "vis" if self.rng.random() < self.removal_ratio else "none"
        elif self.mode == "eval":
            trigger = self.eval_trigger

        return {
            "idx": idx,
            "target": target,
            "red_xy": red_xy,
            "blue_xy": blue_xy,
            "trigger": trigger,
            "instruction": self._make_instruction(target, trigger == "lang"),
        }

    def _random_xy(self) -> tuple[int, int]:
        margin = 10
        size = self.spec.image_size
        return (
            self.rng.randint(margin, size - margin - 1),
            self.rng.randint(margin, size - margin - 1),
        )

    def _make_instruction(self, target: int, lang_trigger: bool) -> str:
        color = "red" if target == 0 else "blue"
        words = ["pick", color, "block"]
        if self.rng.random() < 0.4:
            words.append("please")
        if self.rng.random() < 0.4:
            words.append("now")
        if lang_trigger:
            words.extend(self.spec.lang_words)
        return " ".join(words)

    def _render_image(
        self,
        red_xy: tuple[int, int],
        blue_xy: tuple[int, int],
        visual_trigger: bool,
    ) -> torch.Tensor:
        size = self.spec.image_size
        image = np.zeros((3, size, size), dtype=np.float32)
        image[:] = 0.05
        self._draw_square(image, red_xy, color=(1.0, 0.05, 0.05), half=4)
        self._draw_square(image, blue_xy, color=(0.05, 0.2, 1.0), half=4)
        if visual_trigger:
            p = self.spec.visual_patch_size
            image[:, :p, :p] = np.array([0.0, 1.0, 0.0], dtype=np.float32)[:, None, None]
        return torch.from_numpy(image)

    @staticmethod
    def _draw_square(
        image: np.ndarray,
        xy: tuple[int, int],
        color: tuple[float, float, float],
        half: int,
    ) -> None:
        x, y = xy
        image[:, y - half : y + half, x - half : x + half] = np.array(
            color, dtype=np.float32
        )[:, None, None]
