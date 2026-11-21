"""Bitmap fonts for the Lapis mono screen.

The device ships BMFont (.fnt) descriptors next to ".mono" sprite atlases.
A .mono file is: b"MONO", u16 width, u16 height, u16 stride, u16 bpp?,
u16 flags, then two identical 1-bit planes. We read the first plane.
"""

from __future__ import annotations

import pathlib
import re

import numpy as np

FONT_DIR = pathlib.Path("/usr/share/lapis-hwserviced/gui/mono/font")


def load_mono(path: pathlib.Path) -> np.ndarray:
    raw = path.read_bytes()
    if raw[:4] != b"MONO":
        raise ValueError("not a .mono atlas: " + str(path))
    width = int.from_bytes(raw[4:6], "little")
    height = int.from_bytes(raw[6:8], "little")
    stride = int.from_bytes(raw[8:10], "little")
    body = raw[12 : 12 + stride * height]
    packed = np.frombuffer(body, dtype=np.uint8).reshape(height, stride)
    bits = np.unpackbits(packed, axis=1)[:, :width]
    return bits.astype(bool)


class Font:
    def __init__(self, stem: str, directory: pathlib.Path = FONT_DIR):
        self.stem = stem
        fnt = (directory / (stem + ".fnt")).read_text()
        self.chars: dict[int, dict] = {}
        page_file = ""
        self.line_height = 16
        self.base = 12
        for line in fnt.splitlines():
            if line.startswith("common "):
                values = dict(re.findall(r"(\w+)=(-?\d+)", line))
                self.line_height = int(values.get("lineHeight", 16))
                self.base = int(values.get("base", 12))
            elif line.startswith("page "):
                match = re.search(r'file="([^"]+)"', line)
                if match:
                    page_file = match.group(1)
            elif line.startswith("char "):
                values = dict(re.findall(r"(\w+)=(-?\d+)", line))
                self.chars[int(values["id"])] = {
                    key: int(values[key])
                    for key in ("x", "y", "width", "height", "xoffset", "yoffset", "xadvance")
                }
        atlas = None
        if page_file:
            candidate = directory / (pathlib.Path(page_file).stem + ".mono")
            if candidate.exists():
                atlas = candidate
        if atlas is None:
            for candidate in sorted(directory.glob("*.mono")):
                if stem.split("_")[0].lower() in candidate.name.lower():
                    atlas = candidate
                    break
        if atlas is None:
            raise FileNotFoundError("no .mono atlas for " + stem)
        self.atlas = load_mono(atlas)

    def has(self, ch: str) -> bool:
        return ord(ch) in self.chars

    def advance(self, text: str, scale: int = 1) -> int:
        total = 0
        for ch in text:
            glyph = self.chars.get(ord(ch))
            total += (glyph["xadvance"] if glyph else 4) * scale
        return total

    def draw(self, target: np.ndarray, x: int, y: int, text: str, value: bool = True, scale: int = 1) -> None:
        """Stamp glyph pixels into a boolean layer (top-left anchored)."""
        cursor = int(x)
        for ch in text:
            glyph = self.chars.get(ord(ch))
            if glyph is None:
                cursor += 4 * scale
                continue
            width, height = glyph["width"], glyph["height"]
            if width and height:
                patch = self.atlas[
                    glyph["y"] : glyph["y"] + height,
                    glyph["x"] : glyph["x"] + width,
                ]
                gx = cursor + glyph["xoffset"] * scale
                gy = int(y) + glyph["yoffset"] * scale
                if scale != 1:
                    patch = np.repeat(np.repeat(patch, scale, axis=0), scale, axis=1)
                stamp(target, patch, gx, gy, value)
            cursor += glyph["xadvance"] * scale

    def draw_right(self, target: np.ndarray, right: int, y: int, text: str, value: bool = True, scale: int = 1) -> None:
        self.draw(target, right - self.advance(text, scale), y, text, value, scale)

    def draw_center(self, target: np.ndarray, center: int, y: int, text: str, value: bool = True, scale: int = 1) -> None:
        self.draw(target, center - self.advance(text, scale) // 2, y, text, value, scale)


def stamp(target: np.ndarray, patch: np.ndarray, x: int, y: int, value: bool = True) -> None:
    height, width = target.shape
    ph, pw = patch.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + pw), min(height, y + ph)
    if x0 >= x1 or y0 >= y1:
        return
    sub = patch[y0 - y : y1 - y, x0 - x : x1 - x] & True
    if value:
        target[y0:y1, x0:x1] |= sub
    else:
        target[y0:y1, x0:x1] &= ~sub


_FONTS: dict[str, Font] = {}


def font(stem: str) -> Font:
    if stem not in _FONTS:
        _FONTS[stem] = Font(stem)
    return _FONTS[stem]


def serif() -> Font:
    return font("avicenna_18")


def mono() -> Font:
    return font("KohiMono20-Medium")


def digits() -> Font:
    return font("kohi_11")
