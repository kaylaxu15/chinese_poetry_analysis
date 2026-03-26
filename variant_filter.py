"""
variant_filter.py
-----------------
Automated three-layer variant detection for large sets of ancient-only
characters.

Layers
------
1. Unicode Unihan database
   Fields: kSemanticVariant, kZVariant, kSimplifiedVariant,
           kTraditionalVariant, kCompatibilityVariant

2. OpenCC conversion
   Handles formal simplification reform (繞→绕 etc.).

3. Embedding similarity threshold
   If a character's top-1 aligned modern neighbour has cosine similarity
   above NEAR_VARIANT_THRESHOLD (default 0.85), it is treated as a
   high-confidence functional synonym.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import opencc
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NEAR_VARIANT_THRESHOLD = 0.85

VARIANT_FIELDS = {
    "kSemanticVariant",
    "kZVariant",
    "kSimplifiedVariant",
    "kTraditionalVariant",
    "kCompatibilityVariant",
    "kSpoofingVariant",
}

# ---------------------------------------------------------------------------
# Unihan parser
# ---------------------------------------------------------------------------

def _parse_unihan_codepoint(raw: str) -> Optional[str]:
    m = re.match(r"U\+([0-9A-Fa-f]+)", raw)
    if not m:
        return None
    cp = int(m.group(1), 16)
    try:
        return chr(cp)
    except (ValueError, OverflowError):
        return None


def load_unihan_variants(path: str | Path) -> dict[str, set[str]]:
    path = Path(path)
    if not path.exists():
        print(
            f"[variant_filter] WARNING: Unihan_Variants.txt not found at {path}.\n"
            "Layer 1 (Unihan) will be skipped."
        )
        return {}

    variants: dict[str, set[str]] = {}

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            codepoint_raw, field_name, value = parts[0], parts[1], parts[2]
            if field_name not in VARIANT_FIELDS:
                continue

            ch = _parse_unihan_codepoint(codepoint_raw)
            if ch is None:
                continue

            if ch not in variants:
                variants[ch] = set()

            for token in value.split():
                target = _parse_unihan_codepoint(token)
                if target and target != ch:
                    variants[ch].add(target)
                    if target not in variants:
                        variants[target] = set()
                    variants[target].add(ch)

    print(f"[variant_filter] Unihan: loaded {len(variants)} characters.")
    return variants


# ---------------------------------------------------------------------------
# Core filter
# ---------------------------------------------------------------------------

@dataclass
class FilterResult:
    character: str
    is_variant: bool
    layer: Optional[int]
    reason: str
    modern_equivalents: set[str] = field(default_factory=set)


class VariantFilter:
    def __init__(
        self,
        modern_vocab: set[str],
        modern_aligned: dict[str, np.ndarray],
        classical_wv,
        unihan_path: str | Path = "Unihan_Variants.txt",
        threshold: float = NEAR_VARIANT_THRESHOLD,
    ):
        self.modern_vocab = modern_vocab
        self.modern_aligned = modern_aligned
        self.classical_wv = classical_wv
        self.threshold = threshold

        self._converter = opencc.OpenCC("t2s")

        # Layer 1
        self._unihan: dict[str, set[str]] = load_unihan_variants(unihan_path)

    # ------------------------------------------------------------------
    # Layer 1: Unihan
    # ------------------------------------------------------------------

    def _layer1_unihan(self, ch: str) -> Optional[FilterResult]:
        if ch not in self._unihan:
            return None
        matches = self._unihan[ch] & self.modern_vocab
        if matches:
            return FilterResult(
                character=ch,
                is_variant=True,
                layer=1,
                reason=f"Unihan variant of {matches}",
                modern_equivalents=matches,
            )
        return None

    # ------------------------------------------------------------------
    # Layer 2: OpenCC only
    # ------------------------------------------------------------------

    def _layer2_opencc(self, ch: str) -> Optional[FilterResult]:
        converted = self._converter.convert(ch).strip()
        if converted and converted != ch:
            hits = set(converted) & self.modern_vocab
            if hits:
                return FilterResult(
                    character=ch,
                    is_variant=True,
                    layer=2,
                    reason=f"OpenCC converts to {converted!r}, found {hits}",
                    modern_equivalents=hits,
                )
        return None

    # ------------------------------------------------------------------
    # Layer 3: Embedding similarity
    # ------------------------------------------------------------------

    def _layer3_embedding(self, ch: str) -> Optional[FilterResult]:
        if ch not in self.classical_wv:
            return None

        classical_vec = self.classical_wv[ch]
        best_word, best_sim = None, -1.0

        for word, vec in self.modern_aligned.items():
            sim = float(np.dot(classical_vec, vec) /
                        (np.linalg.norm(classical_vec) * np.linalg.norm(vec) + 1e-9))
            if sim > best_sim:
                best_sim = sim
                best_word = word

        if best_sim >= self.threshold:
            return FilterResult(
                character=ch,
                is_variant=True,
                layer=3,
                reason=f"Embedding match {best_word!r} sim={best_sim:.3f}",
                modern_equivalents={best_word} if best_word else set(),
            )

        return None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def classify(self, ch: str) -> FilterResult:
        result = (
            self._layer1_unihan(ch)
            or self._layer2_opencc(ch)
            or self._layer3_embedding(ch)
        )
        if result:
            return result
        return FilterResult(
            character=ch,
            is_variant=False,
            layer=None,
            reason="passed all layers — likely truly disappeared",
        )

    def is_variant(self, ch: str) -> bool:
        return self.classify(ch).is_variant

    def build_report(self, candidates: list[tuple[str, int]]):
        rows = []
        for ch, freq in candidates:
            r = self.classify(ch)
            rows.append({
                "character": ch,
                "frequency": freq,
                "is_variant": r.is_variant,
                "filter_layer": r.layer,
                "reason": r.reason,
                "modern_equivalents": ", ".join(sorted(r.modern_equivalents)),
            })
        return rows