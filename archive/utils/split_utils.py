"""Leakage-safe dataset splitting helpers."""

import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List


_CHANNEL_SUFFIX = re.compile(r"_ch\d+$", re.IGNORECASE)
_SPLIT_NAMES = ("train", "val", "test")
_SPLIT_RATIOS = (0.70, 0.15, 0.15)


def source_id(path: Path) -> str:
    """Return the original source ID shared by all channel variants."""
    return _CHANNEL_SUFFIX.sub("", path.stem)


def _allocate_group_counts(n_groups: int) -> List[int]:
    """Allocate group counts with largest remainders so totals stay exact."""
    raw = [n_groups * ratio for ratio in _SPLIT_RATIOS]
    counts = [int(value) for value in raw]
    remaining = n_groups - sum(counts)
    order = sorted(
        range(len(raw)), key=lambda i: (raw[i] - counts[i], -i), reverse=True,
    )
    for i in order[:remaining]:
        counts[i] += 1
    return counts


def stratified_group_split(
    files: Iterable[Path],
    label_getter: Callable[[Path], str],
    seed: int = 42,
) -> Dict[str, List[Path]]:
    """Split by L3 while keeping every source's channel variants together."""
    grouped = defaultdict(lambda: defaultdict(list))
    for path in files:
        label = str(label_getter(path))
        grouped[label][source_id(path)].append(path)

    rng = random.Random(seed)
    splits = {name: [] for name in _SPLIT_NAMES}
    for label in sorted(grouped):
        groups = list(grouped[label].values())
        rng.shuffle(groups)
        counts = _allocate_group_counts(len(groups))
        start = 0
        for name, count in zip(_SPLIT_NAMES, counts):
            selected = groups[start:start + count]
            splits[name].extend(path for group in selected for path in sorted(group))
            start += count

    for paths in splits.values():
        rng.shuffle(paths)
    return splits
