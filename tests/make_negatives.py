#!/usr/bin/env python3

"""Generate the synthetic non-leaf images used by the out-of-distribution tests.

These are procedurally generated rather than downloaded so the suite stays
license-clean and byte-reproducible. Each one is a plausible thing a user might
accidentally photograph, and none of them should ever produce a confident
diagnosis.

Usage: python3 tests/make_negatives.py
"""

from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "test_images" / "_negatives"
SIZE = 512


def sky() -> np.ndarray:
    """Blue gradient — the classic "pointed the camera up" shot."""
    grad = np.linspace(220, 120, SIZE, dtype=np.float32)
    img = np.zeros((SIZE, SIZE, 3), np.float32)
    img[:, :, 0] = grad[:, None]           # R low->lower
    img[:, :, 1] = grad[:, None] + 15
    img[:, :, 2] = 245                      # B dominant
    return np.clip(img, 0, 255).astype(np.uint8)


def skin() -> np.ndarray:
    """Flat skin tone — a hand or face filling the frame."""
    rng = np.random.default_rng(7)
    base = np.array([222, 178, 150], np.float32)
    noise = rng.normal(0, 6, (SIZE, SIZE, 3))
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def pavement() -> np.ndarray:
    """Grey concrete texture."""
    rng = np.random.default_rng(11)
    g = rng.normal(140, 18, (SIZE, SIZE))
    g = cv2.GaussianBlur(g, (0, 0), 2)
    return np.clip(np.stack([g, g, g], -1), 0, 255).astype(np.uint8)


def screenshot() -> np.ndarray:
    """A UI screenshot: white background, coloured blocks, text-like bars."""
    img = np.full((SIZE, SIZE, 3), 250, np.uint8)
    cv2.rectangle(img, (0, 0), (SIZE, 64), (60, 90, 200), -1)
    for i in range(6):
        y = 110 + i * 55
        cv2.rectangle(img, (40, y), (40 + 260 + i * 20, y + 18), (190, 190, 190), -1)
    cv2.rectangle(img, (300, 380), (470, 470), (200, 120, 60), -1)
    return img


def noise() -> np.ndarray:
    """Uniform RGB noise — maximally out of distribution."""
    rng = np.random.default_rng(3)
    return rng.integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8)


def blurred_leaf() -> np.ndarray:
    """A green leaf-ish blob, heavily defocused: passes the vegetation check
    but must be caught by the blur guard."""
    img = np.full((SIZE, SIZE, 3), 30, np.uint8)
    cv2.ellipse(img, (SIZE // 2, SIZE // 2), (170, 110), 25, 0, 360, (60, 150, 70), -1)
    return cv2.GaussianBlur(img, (0, 0), 12)


def dark_leaf() -> np.ndarray:
    """Underexposed leaf — vegetation-shaped but far too dark to diagnose."""
    img = np.full((SIZE, SIZE, 3), 8, np.uint8)
    cv2.ellipse(img, (SIZE // 2, SIZE // 2), (170, 110), 25, 0, 360, (14, 30, 16), -1)
    return img


GENERATORS = {
    "sky.jpg": sky,
    "skin.jpg": skin,
    "pavement.jpg": pavement,
    "screenshot.jpg": screenshot,
    "noise.jpg": noise,
    "blurred_leaf.jpg": blurred_leaf,
    "dark_leaf.jpg": dark_leaf,
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in GENERATORS.items():
        rgb = fn()
        cv2.imwrite(str(OUT / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(f"wrote {OUT / name}")


if __name__ == "__main__":
    main()
