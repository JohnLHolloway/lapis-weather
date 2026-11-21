"""The world, drawn for a 400x240 one-bit screen.

Everything is composed as luminance (0 = ink, 1 = paper) and flattened to bits
at the very end, so gradients become ordered dither rather than grey.

One-bit landscapes read best when the palette is pushed to the extremes:
paper-white sky, solid black silhouettes, dither only in the thin places
between them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from fonts import mono, serif

W, H = 400, 240
CONTENT_H = 215          # the system status toolbar owns the last 25 rows


# --------------------------------------------------------------- primitives

def bayern(n: int = 8) -> np.ndarray:
    matrix = np.array([[0, 2], [3, 1]], dtype=np.float32)
    size = 2
    while size < n:
        matrix = np.block([
            [4 * matrix + 0, 4 * matrix + 2],
            [4 * matrix + 3, 4 * matrix + 1],
        ])
        size *= 2
    return (matrix + 0.5) / (size * size)


BAYER = bayern(8)
_TILED = np.tile(BAYER, (H // 8 + 1, W // 8 + 1))[:H, :W]
_RNG = np.random.default_rng(20261120)
_GRAIN = (_RNG.random((H, W), dtype=np.float32) - 0.5) * (0.85 / 64.0)
THRESHOLD = np.clip(_TILED + _GRAIN, 0.0005, 0.9995)
PIXEL_Y = np.arange(H, dtype=np.float32)[:, None]
PIXEL_X = np.arange(W, dtype=np.float32)[None, :]
PIXEL_YY = np.repeat(PIXEL_Y, W, axis=1)


def clamp(value, low=0.0, high=1.0):
    return np.clip(value, low, high)


def smoothstep(edge0: float, edge1: float, value):
    t = np.clip((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def mix(a, b, t):
    return a * (1.0 - t) + b * t


def value_noise(x: np.ndarray, seed: float) -> np.ndarray:
    xi = np.floor(x)
    xf = x - xi
    def hashed(i):
        v = np.sin(i * 127.1 + seed * 311.7) * 43758.5453
        return v - np.floor(v)
    a = hashed(xi)
    b = hashed(xi + 1.0)
    t = xf * xf * (3 - 2 * xf)
    return a + (b - a) * t


def fbm(x: np.ndarray, seed: float, octaves: int = 4, gain: float = 0.5) -> np.ndarray:
    total = np.zeros_like(x, dtype=np.float32)
    amplitude = 1.0
    frequency = 1.0
    norm = 0.0
    for index in range(octaves):
        total = total + amplitude * value_noise(x * frequency, seed + index * 17.0)
        norm += amplitude
        amplitude *= gain
        frequency *= 2.03
    return total / norm


def blit(image: np.ndarray, mask: np.ndarray, x: int, y: int, tone: float, alpha: float = 1.0) -> None:
    target_h, target_w = image.shape
    mask_h, mask_w = mask.shape
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(target_w, int(x) + mask_w), min(target_h, int(y) + mask_h)
    if x0 >= x1 or y0 >= y1:
        return
    sub = mask[y0 - y : y1 - y, x0 - x : x1 - x]
    region = image[y0:y1, x0:x1]
    if alpha >= 1.0:
        region[sub] = tone
    else:
        region[sub] = region[sub] * (1.0 - alpha) + tone * alpha


def stroke(mask: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> None:
    steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    xs = np.linspace(x0, x1, steps).round().astype(int)
    ys = np.linspace(y0, y1, steps).round().astype(int)
    keep = (xs >= 0) & (xs < mask.shape[1]) & (ys >= 0) & (ys < mask.shape[0])
    mask[ys[keep], xs[keep]] = True


# ------------------------------------------------------------------ sprites

def pine(height: int, seed: int) -> np.ndarray:
    width = max(7, int(height * 0.42))
    mask = np.zeros((height, width), dtype=bool)
    centre = width // 2
    tiers = 3 if height < 34 else 4
    for tier in range(tiers):
        top = int(height * (0.02 + tier * 0.19))
        bottom = int(min(height * 0.96, height * (0.48 + tier * 0.16)))
        half = (width / 2.0) * (0.38 + 0.62 * (tier + 1) / tiers)
        span = max(1, bottom - top)
        for row in range(top, bottom):
            frac = (row - top) / span
            reach = int(half * (0.18 + 0.82 * frac)) + 1
            mask[row, max(0, centre - reach) : min(width, centre + reach + 1)] = True
    trunk = max(1, width // 8)
    mask[int(height * 0.92) :, centre - trunk : centre + trunk + 1] = True
    jitter = np.random.default_rng(seed)
    for row in range(2, height - 2, 2):
        edge = int(np.argmax(mask[row]))
        if edge > 0 and jitter.random() < 0.6:
            mask[row, edge] = False
        edge = width - 1 - int(np.argmax(mask[row][::-1]))
        if edge < width - 1 and jitter.random() < 0.6:
            mask[row, edge] = False
    return mask


def round_tree(height: int, seed: int) -> np.ndarray:
    width = max(9, int(height * 0.82))
    mask = np.zeros((height, width), dtype=bool)
    centre = width // 2
    crown = int(height * 0.74)
    for row in range(crown):
        frac = row / max(1, crown - 1)
        reach = int((width / 2.0) * math.sin(math.pi * min(1.0, frac * 0.92 + 0.08)) ** 0.7)
        mask[row, max(0, centre - reach) : min(width, centre + reach + 1)] = True
    trunk = max(1, width // 9)
    mask[int(height * 0.68) :, centre - trunk : centre + trunk + 1] = True
    return mask


PINE_SPRITES = [pine(size, size) for size in (10, 14, 20, 26, 34, 44, 58, 86, 120, 165)]
ROUND_SPRITES = [round_tree(size, size) for size in (16, 22, 30, 42)]


def cabin(scale: int = 1) -> tuple[np.ndarray, np.ndarray]:
    w, h = 22 * scale, 16 * scale
    body = np.zeros((h, w), dtype=bool)
    body[int(h * 0.42) :, 1 : w - 1] = True
    peak = int(h * 0.44)
    for row in range(peak):
        reach = int((row / peak) * (w / 2.0)) + 1
        body[row, max(0, w // 2 - reach) : min(w, w // 2 + reach)] = True
    window = np.zeros((h, w), dtype=bool)
    window[int(h * 0.56) : int(h * 0.80), int(w * 0.26) : int(w * 0.46)] = True
    window[int(h * 0.56) : int(h * 0.58), int(w * 0.26) : int(w * 0.46)] = False
    return body, window


CABIN, CABIN_WINDOW = cabin(1)


def lighthouse(height: int = 52) -> tuple[np.ndarray, np.ndarray]:
    width = 15
    body = np.zeros((height, width), dtype=bool)
    for row in range(int(height * 0.34), height):
        frac = (row - height * 0.34) / (height * 0.66)
        half = int(2 + frac * 2.2)
        body[row, width // 2 - half : width // 2 + half + 1] = True
    body[int(height * 0.24) : int(height * 0.34), width // 2 - 3 : width // 2 + 4] = True
    for step in range(4):
        body[int(height * 0.24) - step, width // 2 - 4 + step : width // 2 + 5 - step] = True
    lamp = np.zeros((height, width), dtype=bool)
    lamp[int(height * 0.26) : int(height * 0.33), width // 2 - 2 : width // 2 + 3] = True
    return body, lamp


LIGHTHOUSE, LIGHTHOUSE_LAMP = lighthouse(52)


def branch(width: int = 190, height: int = 96, seed: int = 5) -> np.ndarray:
    """A bare limb reaching in from the corner, with needle tufts."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((height, width), dtype=bool)
    start = (0.0, height * 0.16)
    end = (width * 0.92, height * 0.62)
    for thickness in (0.0, 1.0, 1.8):
        stroke(mask, start[0], start[1] + thickness, end[0], end[1] + thickness)
    for step in range(7):
        frac = 0.16 + step * 0.12
        bx = start[0] + (end[0] - start[0]) * frac
        by = start[1] + (end[1] - start[1]) * frac
        length = width * rng.uniform(0.10, 0.22)
        angle = rng.uniform(-0.95, -0.45)
        tx = bx + math.cos(angle) * length
        ty = by + math.sin(angle) * length
        stroke(mask, bx, by, tx, ty)
        for needle in range(3):
            nx = tx + rng.uniform(-5, 5)
            ny = ty + rng.uniform(-4, 6)
            stroke(mask, tx, ty, nx, ny)
        # tuft
        for k in range(5):
            stroke(mask, tx, ty, tx + rng.uniform(-7, 7), ty + rng.uniform(-6, 5))
    return mask


def reeds(count: int = 9, seed: int = 3) -> list[tuple[np.ndarray, int, int]]:
    rng = np.random.default_rng(seed)
    tufts = []
    for _ in range(count):
        height = int(rng.uniform(46, 92))
        width = int(rng.uniform(14, 30))
        mask = np.zeros((height + 8, width), dtype=bool)
        base_x = width // 2
        lean = rng.uniform(-0.35, 0.35)
        tip_x = base_x + lean * height * 0.6
        stroke(mask, base_x, height + 6, tip_x, 6)
        stroke(mask, base_x + 1, height + 6, tip_x + 1, 7)
        seed_head = np.zeros_like(mask)
        for row in range(0, 9):
            reach = int(2 + row * 0.35)
            seed_head[2 + row, int(tip_x) - reach : int(tip_x) + reach + 1] = True
        mask |= seed_head
        tufts.append((mask, int(rng.uniform(-10, 400)), 0))
    return tufts


# --------------------------------------------------------------------- tone

PRESETS = {
    "day":   dict(sky_top=0.99, sky_horizon=1.0, haze=0.97, cloud=0.84, ridge_far=0.80,
                  ridge_mid=0.22, ridge_near=0.0, tree=0.02, water_far=0.86, water=0.04,
                  star=0.0, moon=0.0),
    "dusk":  dict(sky_top=0.95, sky_horizon=1.0, haze=0.94, cloud=0.80, ridge_far=0.58,
                  ridge_mid=0.16, ridge_near=0.0, tree=0.008, water_far=0.52, water=0.02,
                  star=0.0, moon=0.6),
    "night": dict(sky_top=0.0, sky_horizon=0.05, haze=0.06, cloud=0.0, ridge_far=0.05,
                  ridge_mid=0.02, ridge_near=0.0, tree=0.0, water_far=0.0, water=0.0,
                  star=1.0, moon=1.0),
}


def tones(ctx: "Ctx") -> dict:
    day = ctx.day
    w_day = float(smoothstep(0.60, 0.95, day))
    w_night = float(smoothstep(0.30, 0.03, day))
    w_dusk = max(0.0, 1.0 - w_day - w_night)
    out = {}
    for key in PRESETS["day"]:
        out[key] = (
            PRESETS["day"][key] * w_day
            + PRESETS["dusk"][key] * w_dusk
            + PRESETS["night"][key] * w_night
        )

    weather = ctx.weather
    cloud = float(weather.get("cloud", 0.0))
    rain = float(weather.get("rain", 0.0))
    snow = float(weather.get("snow", 0.0))
    fog = float(weather.get("fog", 0.0))
    storm = float(weather.get("thunder", 0.0))

    out["sky_top"] = out["sky_top"] * (1.0 - 0.22 * cloud - 0.46 * rain - 0.22 * storm)
    out["sky_horizon"] = out["sky_horizon"] * (1.0 - 0.10 * cloud - 0.34 * rain - 0.18 * storm)
    out["ridge_far"] = mix(out["ridge_far"], out["ridge_far"] * 0.86, rain)
    out["ridge_mid"] = mix(out["ridge_mid"], out["ridge_mid"] * 0.86, rain)
    if snow > 0.02:
        out["sky_top"] = mix(out["sky_top"], 1.0, 0.42 * snow)
        out["sky_horizon"] = mix(out["sky_horizon"], 1.0, 0.36 * snow)
        out["ridge_far"] = mix(out["ridge_far"], 0.90, 0.5 * snow)
        out["ridge_mid"] = mix(out["ridge_mid"], 0.72, 0.45 * snow)
        out["water_far"] = mix(out["water_far"], 0.9, 0.5 * snow)
    if fog > 0.02:
        out["haze"] = mix(out["haze"], out["sky_horizon"], 0.7 * fog)
        out["ridge_far"] = mix(out["ridge_far"], out["haze"] * 0.96, 0.75 * fog)
        out["ridge_mid"] = mix(out["ridge_mid"], out["haze"] * 0.92, 0.5 * fog)
        out["tree"] = mix(out["tree"], out["haze"] * 0.55, 0.3 * fog)
    out["cloud"] = mix(out["cloud"], out["sky_horizon"] * 0.82, 0.35 * cloud)
    out["rain"] = rain
    out["snow"] = snow
    out["fog"] = fog
    out["cloud_cover"] = cloud
    return out


# -------------------------------------------------------------------- vista

FOOTER_TOP = 150        # solid ink footer: the whole caption block lives here

RIDGE_NOTES = {
    "lake":   {"horizon": 98,  "water": 112, "ridges": [(-10, 40)]},
    "pines":  {"horizon": 100, "water": None, "ridges": [(-8, 34), (2, 14)]},
    "ridge":  {"horizon": 94,  "water": None, "ridges": [(-14, 54), (-2, 32), (10, 14)]},
    "meadow": {"horizon": 102, "water": None, "ridges": [(-6, 28)]},
    "shore":  {"horizon": 96,  "water": 110, "ridges": [(-6, 34)]},
}

VISTAS = ("lake", "pines", "ridge", "meadow", "shore")


class Vista:
    def __init__(self, kind: str, seed: int):
        self.kind = kind
        self.seed = seed
        notes = RIDGE_NOTES[kind]
        self.horizon = notes["horizon"]
        self.water_line = notes["water"]
        rng = np.random.default_rng(seed * 7919 + 13)
        self.pan = [float(rng.uniform(0, 300)) for _ in range(4)]
        self.ridges = []
        for index, (offset, amplitude) in enumerate(notes["ridges"]):
            self.ridges.append({
                "offset": offset,
                "amplitude": amplitude,
                "seed": float(rng.uniform(1, 400)),
                "depth": 0.28 * (len(notes["ridges"]) - index),
            })
        self.trees = []
        self._plant_trees(rng)
        self.grass = self._grow_grass(rng)
        self.stars = self._make_stars(rng)
        self.wildflowers = rng.random((70, 2))
        self.branch = None
        self.reeds = reeds(9, int(rng.integers(1, 40))) if kind in ("lake", "shore", "meadow") else []

    def _plant_trees(self, rng) -> None:
        baseline = self.horizon + 2
        for band, (count, height_range, tone_fade) in enumerate((
            (30, (7, 14), 0.85),
            (18, (16, 30), 1.0),
            (7, (30, 48), 1.0),
        )):
            if self.kind == "meadow" and band:
                continue
            for _ in range(count):
                height = int(rng.integers(height_range[0], height_range[1]))
                sprite = PINE_SPRITES[min(len(PINE_SPRITES) - 1, max(0, int(height / 10)))]
                self.trees.append({
                    "x": float(rng.uniform(-20, W + 20)),
                    "y": float(baseline - sprite.shape[0] + band * 7 + rng.uniform(-2, 2)),
                    "sprite": sprite,
                    "depth": float(rng.uniform(0.04, 0.14) + band * 0.04),
                    "fade": tone_fade,
                    "cluster": int(rng.integers(0, 12)),
                })
        if self.kind in ("meadow", "shore"):
            for _ in range(3 if self.kind == "meadow" else 2):
                sprite = ROUND_SPRITES[int(rng.integers(0, len(ROUND_SPRITES)))]
                self.trees.append({
                    "x": float(rng.uniform(10, W - 30)),
                    "y": float(baseline - sprite.shape[0] + 26),
                    "sprite": sprite,
                    "depth": 0.22,
                    "fade": 1.0,
                })

    def _grow_grass(self, rng) -> list[tuple[float, float, float, float]]:
        blades = []
        count = 150 if self.kind == "meadow" else 60
        for _ in range(count):
            blades.append((
                float(rng.uniform(-10, W + 10)),
                float(rng.uniform(200, 244)),
                float(rng.uniform(6, 20)),
                float(rng.uniform(-8, 8)),
            ))
        return blades

    def _make_stars(self, rng) -> np.ndarray:
        count = 210
        xs = rng.integers(0, W, count)
        ys = (rng.random(count) ** 1.5 * (self.horizon - 10)).astype(int)
        bright = rng.random(count) ** 1.8
        return np.stack([xs, ys, bright], axis=1)


# ---------------------------------------------------------------------- sky

def paint_sky(image: np.ndarray, ctx: "Ctx", vista: Vista, tone: dict) -> None:
    horizon_y = vista.horizon
    curve = np.clip(PIXEL_Y[:horizon_y] / max(1.0, horizon_y), 0, 1) ** 1.7
    gradient = tone["sky_top"] + (tone["sky_horizon"] - tone["sky_top"]) * curve
    image[:horizon_y, :] = gradient
    image[horizon_y:, :] = tone["ridge_near"]

    if ctx.golden > 0.02:
        band = np.exp(-(((PIXEL_Y - horizon_y + 8) / 18.0) ** 2)) * 0.22 * ctx.golden
        image[:horizon_y, :] = clamp(image[:horizon_y, :] + band[:horizon_y] * (1 - 0.6 * tone["cloud_cover"]))
        image[horizon_y - 2 : horizon_y + 1, :] = clamp(
            image[horizon_y - 2 : horizon_y + 1, :] + 0.12 * ctx.golden
        )

    if ctx.day < 0.5:
        glow = np.exp(-((PIXEL_Y[:horizon_y] - horizon_y + 3) / 20.0) ** 2) * (0.085 * (1.0 - ctx.day))
        image[:horizon_y, :] = np.maximum(image[:horizon_y, :], glow * (1.0 - 0.85 * tone["cloud_cover"]))

    if tone["star"] > 0.02:
        fade = tone["star"] * (1.0 - 0.72 * tone["cloud_cover"]) * (1.0 - 0.8 * tone.get("fog", 0.0))
        xs = vista.stars[:, 0].astype(int)
        ys = vista.stars[:, 1].astype(int)
        bright = vista.stars[:, 2] * fade
        keep = bright > 0.30
        image[ys[keep], xs[keep]] = np.maximum(image[ys[keep], xs[keep]], 0.55 + 0.45 * bright[keep])
        faint = bright > 0.10
        image[ys[faint], xs[faint]] = np.maximum(image[ys[faint], xs[faint]], 0.33)


def paint_sun_and_moon(image: np.ndarray, ctx: "Ctx", vista: Vista, tone: dict) -> None:
    horizon_y = vista.horizon
    sun_x = W * 0.5 + (ctx.sun_az - 180.0) / 180.0 * W * 0.58
    sun_y = horizon_y - math.sin(math.radians(max(-12.0, ctx.sun_alt))) * horizon_y * 1.15

    if ctx.sun_alt > -7:
        height = float(clamp((7.0 + ctx.sun_alt) / 30.0))
        strength = 0.55 * height * (1 - 0.6 * tone["cloud_cover"])
        _ring(image, sun_x, sun_y, 9.0, 44.0, 0.30 * strength, 0.922)
        disc = np.sqrt((PIXEL_X - sun_x) ** 2 + (PIXEL_Y - sun_y) ** 2) <= 8.0
        image[disc] = 1.0

    if ctx.moon_alt > -4 and ctx.day < 0.68:
        moon_x = W * 0.5 + (ctx.moon_az - 180.0) / 180.0 * W * 0.58
        moon_y = horizon_y - math.sin(math.radians(max(-8.0, ctx.moon_alt))) * horizon_y * 1.1
        strength = tone["moon"] * float(clamp((0.68 - ctx.day) / 0.68)) * float(clamp((ctx.moon_alt + 5) / 9.0))
        if strength > 0.02:
            _glow(image, moon_x, moon_y, 55.0, 0.14 * strength * (0.45 + ctx.moon_illum))
            _moon_disc(image, moon_x, moon_y, 14.0, ctx.moon_phase, ctx.moon_illum)


def _glow(image: np.ndarray, cx: float, cy: float, radius: float, strength: float) -> None:
    dist = np.sqrt((PIXEL_X - cx) ** 2 + (PIXEL_YY - cy) ** 2) / radius
    image += np.exp(-(dist ** 2) * 3.0) * strength


def _ring(image, cx, cy, inner, outer, strength, tone) -> None:
    dist = np.sqrt((PIXEL_X - cx) ** 2 + (PIXEL_YY - cy) ** 2)
    band = np.exp(-(((dist - (inner + outer) / 2) / ((outer - inner) / 2.2)) ** 2)) * strength
    image[:] = image * (1 - band) + tone * band


def _moon_disc(image, cx, cy, radius, phase, illum) -> None:
    xx = PIXEL_X - cx
    yy = PIXEL_YY - cy
    disc = np.sqrt(xx ** 2 + yy ** 2) <= radius
    if illum > 0.93:
        image[disc] = 0.98
        return
    if illum < 0.06:
        return
    k = math.cos(2.0 * math.pi * phase)
    chord = np.sqrt(np.clip(radius ** 2 - yy ** 2, 0, None))
    terminator = k * chord
    lit = disc & (xx > terminator) if phase < 0.5 else disc & (xx < -terminator)
    image[lit] = 0.95


# ------------------------------------------------------------------- clouds

def paint_clouds(image: np.ndarray, ctx: "Ctx", vista: Vista, tone: dict) -> None:
    cover = float(tone["cloud_cover"])
    if cover < 0.05 or ctx.day < 0.22:
        return
    rng = np.random.default_rng(4242)
    count = int(2 + cover * 6)
    span = W + 340
    for index in range(count):
        depth = 0.25 + 0.75 * (index / max(1, count))
        speed = 1.0 + depth * 3.4 + ctx.wind * 0.18
        x = ((rng.uniform(0, span) + ctx.t * speed) % span) - 170
        y = vista.horizon - rng.uniform(14, 100) * (0.45 + depth * 0.55)
        width = rng.uniform(55, 130) * (0.6 + depth * 0.7)
        height = rng.uniform(14, 26) * (0.6 + depth * 0.6)
        opacity = min(0.92, (0.18 + cover * rng.uniform(0.45, 0.9)) * (0.35 + 0.65 * ctx.day))
        _cloud(image, x, y, width, height, tone["cloud"], opacity, rng)


def _cloud(image, cx, cy, width, height, tone, opacity, rng) -> None:
    """Puffs, drawn as a woodcut: a dark contour with a faint body."""
    body = np.zeros((H, W), dtype=np.float32)
    for index in range(6):
        offset_x = rng.uniform(-0.5, 0.5) * width
        offset_y = -abs(rng.normal(0.0, 0.3)) * height + height * 0.1
        puff_w = width * rng.uniform(0.28, 0.55)
        puff_h = height * rng.uniform(0.55, 1.0)
        body = np.maximum(body, np.exp(
            -(((PIXEL_X - cx - offset_x) / puff_w) ** 2 + ((PIXEL_YY - cy - offset_y) / puff_h) ** 2) * 1.7
        ))
    solid = body > 0.55
    # the contour of the cloud, one pixel wide
    edge = solid.copy()
    edge[1:, :] &= solid[:-1, :]
    edge[:-1, :] &= solid[1:, :]
    edge[:, 1:] &= solid[:, :-1]
    edge[:, :-1] &= solid[:, 1:]
    line = solid & ~edge
    image[:] = image * (1.0 - body * opacity * 0.10) + tone * body * opacity * 0.10
    image[line] = min(0.9, tone * 0.55)


# ------------------------------------------------------------------- ridges

def ridge_profile(x: np.ndarray, seed: float, pan: float, drift: float) -> np.ndarray:
    """Sharp-peaked ridged noise: two scales of 1 - |2n - 1|."""
    u = (x + pan + drift) / 240.0
    broad = fbm(u, seed, octaves=3)
    ridge_broad = (1.0 - np.abs(broad * 2.0 - 1.0)) ** 1.7
    fine = fbm(u * 3.4 + 9.0, seed + 11.0, octaves=3)
    ridge_fine = (1.0 - np.abs(fine * 2.0 - 1.0)) ** 2.2
    return 0.68 * ridge_broad + 0.32 * ridge_fine


def ridge_height(vista: Vista, ridge: dict, ctx: "Ctx", x: np.ndarray) -> np.ndarray:
    base = vista.horizon + ridge["offset"]
    amplitude = ridge["amplitude"]
    drift = ctx.t * 0.5 * ridge["depth"]
    return base - ridge_profile(x, ridge["seed"], vista.pan[0] * 6.0, drift) * amplitude


def _ground(image: np.ndarray, ctx: "Ctx", vista: Vista, tone: dict) -> None:
    top = vista.horizon
    depth = np.clip((PIXEL_Y[top:] - top) / max(1.0, H - top), 0, 1)
    lit = 0.03 + 0.16 * ctx.day
    ground = mix(tone["ridge_near"] + lit, tone["ridge_near"], depth ** 0.6)
    if tone["snow"] > 0.15:
        snow_ground = 0.62 + 0.30 * ctx.day
        cover = min(1.0, (tone["snow"] - 0.15) * 2.4)
        ground = ground * (1 - cover) + snow_ground * cover
    image[top:, :] = ground


def paint_ridges(image: np.ndarray, ctx: "Ctx", vista: Vista, tone: dict) -> None:
    xs = np.arange(W, dtype=np.float32)
    count = max(1, len(vista.ridges) - 1)
    for index, ridge in enumerate(vista.ridges):
        heights = ridge_height(vista, ridge, ctx, xs)
        base = vista.horizon + ridge["offset"] + 6
        coverage = (PIXEL_YY >= heights[None, :]) & (PIXEL_YY <= base)
        # far ridges are pale with distance, the nearest is nearly ink
        depth_ratio = index / count
        ridge_tone = mix(tone["ridge_far"], tone["ridge_near"], depth_ratio ** 1.4)
        image[coverage] = ridge_tone
        # a one-pixel contour: the classic trick for one-bit distance
        edge = np.zeros((H, W), dtype=bool)
        rows = np.clip(heights.astype(int), 0, H - 1)
        edge[rows, np.arange(W)] = True
        rim = mix(tone["ridge_near"], 0.95, 0.9 * (1.0 - ctx.day) ** 1.5)
        image[edge] = rim

        if tone["snow"] > 0.10 and index < len(vista.ridges) - 1:
            texture = fbm(xs / 7.0, ridge["seed"] + 9.0, octaves=2)[None, :]
            cap = coverage & (PIXEL_YY < heights[None, :] + 7) & (texture > 0.58)
            image[cap] = 0.9


def paint_fog(image: np.ndarray, ctx: "Ctx", vista: Vista, tone: dict) -> None:
    fog = tone["fog"]
    if fog <= 0.04:
        return
    band = np.exp(-((PIXEL_Y - vista.horizon + 10) / (12 + 20 * fog)) ** 2)
    veil = np.clip(fog * band * 1.35, 0.0, 0.97)
    mist = 0.80 + 0.18 * ctx.day
    image[:] = image * (1 - veil) + mist * veil


def paint_treeline(image: np.ndarray, ctx: "Ctx", vista: Vista, tone: dict) -> None:
    haze = tone["sky_horizon"] * 0.94
    for tree in sorted(vista.trees, key=lambda item: item["depth"]):
        drift = ctx.t * 0.4 * tree["depth"]
        x = int(round(tree["x"] - drift))
        fence = int(tree.get("cluster", 0) * 30)
        x = (x + fence) % (W + 140) - 60
        tree_tone = mix(mix(tone["tree"], haze, 0.55 * (1.0 - tree["fade"])), tone["tree"], 0.25)
        blit(image, tree["sprite"], x, int(tree["y"]), tree_tone)


# -------------------------------------------------------------------- water

def paint_water(image: np.ndarray, ctx: "Ctx", vista: Vista, tone: dict) -> None:
    line = vista.water_line
    if line is None:
        return
    height = H - line
    yy = np.arange(line, H, dtype=np.float32)[:, None]
    depth = (yy - line) / height
    # black water with a pale lid of reflected sky at the far shore
    water = tone["water"] + (tone["water_far"] - tone["water"]) * np.exp(-((depth / 0.045) ** 1.2))
    water = water + np.sin(yy * 0.9 + ctx.t * 0.7) * 0.02

    if tone["snow"] > 0.25:
        ice = 0.58 + 0.30 * ctx.day
        water = water * 0.2 + ice * 0.8
    image[line:, :] = clamp(water)

    glints = np.zeros((H, W), dtype=bool)
    if ctx.day > 0.25:
        # a few quiet ripples close to the far shore
        rng = np.random.default_rng(int(vista.seed) + 61)
        for index in range(3):
            row = line + 4 + int(index * (height - 8) * 0.22)
            if row >= FOOTER_TOP - 8:
                continue
            start_x = int(rng.uniform(10, W - 90))
            length = int(rng.uniform(30, 90))
            stroke(glints, start_x, row, start_x + length, row)
        image[glints] = 0.85
    elif ctx.moon_alt > 0 and ctx.moon_illum > 0.2:
        # the moon lays a broken path on the water
        moon_x = W * 0.5 + (ctx.moon_az - 180.0) / 180.0 * W * 0.58
        for index in range(9):
            row = line + 4 + index * 5
            if row >= H:
                break
            spread = 2 + index * 1.6
            offset = math.sin(ctx.t * 0.8 + index) * 2.0
            stroke(glints, moon_x - spread + offset, row, moon_x + spread + offset, row)
        image[glints] = 0.80


# ------------------------------------------------------------- atmospherics

def paint_weather(image: np.ndarray, ctx: "Ctx", vista: Vista, tone: dict) -> None:
    rain = tone["rain"]
    snow = tone["snow"]
    background = image.copy()

    if rain > 0.06:
        rng = np.random.default_rng(9001)
        count = int(14 + rain * 80)
        xs = rng.uniform(0, W + 60, count)
        ys = rng.uniform(0, H + 30, count)
        lengths = rng.uniform(3, 7, count)
        speeds = rng.uniform(140, 260, count)
        slant = -0.5 - ctx.wind * 0.012
        field = np.zeros((H, W), dtype=bool)
        for x, y, length, speed in zip(xs, ys, lengths, speeds):
            head = (y + ctx.t * speed) % (H + 30) - 15
            stroke(field, x, head, x + slant * length, head + length)
        light_ground = background < 0.45
        image[field & light_ground] = 0.95
        image[field & ~light_ground] = np.minimum(background[field & ~light_ground], 0.22)

    if snow > 0.06:
        rng = np.random.default_rng(3131)
        count = int(20 + snow * 90)
        xs = rng.uniform(0, W, count)
        ys = rng.uniform(0, H, count)
        speeds = rng.uniform(7, 20, count)
        sway = rng.uniform(0.4, 1.3, count)
        for x, y, speed, wobble in zip(xs, ys, speeds, sway):
            py = int((y + ctx.t * speed) % (H + 8))
            px = int((x + math.sin(ctx.t * wobble + y) * 6) % W)
            image[py, px] = 1.0 if background[py, px] < 0.5 else 0.0


# ---------------------------------------------------------------- foreground

def paint_foreground(image: np.ndarray, ctx: "Ctx", vista: Vista, tone: dict) -> None:
    if getattr(vista, "branch", None) is not None:
        sway = math.sin(ctx.t * 0.3) * 1.2
        blit(image, vista.branch, int(-6 + sway), int(-10), tone["tree"] * 0.6)

    if vista.reeds:
        for index, (mask, x, _) in enumerate(vista.reeds):
            sway = math.sin(ctx.t * 0.55 + index * 1.3) * (2.0 + ctx.wind * 0.12)
            height = mask.shape[0]
            blit(image, mask, int(x + sway), H - height + 4, tone["tree"] * 0.8)

    blades = np.zeros((H, W), dtype=bool)
    for x, base, height, lean in vista.grass:
        sway = math.sin(ctx.t * 0.45 + x * 0.06 + ctx.wind * 0.07) * (1.5 + ctx.wind * 0.1)
        stroke(blades, x, base, x + lean + sway, base - height)
    image[blades] = tone["ridge_near"]

    if vista.kind == "shore":
        house_x = int(W * 0.16)
        house_y = vista.horizon - 30
        blit(image, LIGHTHOUSE, house_x, house_y, tone["ridge_near"])
        if ctx.nightish:
            lamp_x = house_x + 7
            lamp_y = house_y + 15
            image[lamp_y - 1 : lamp_y + 1, lamp_x - 1 : lamp_x + 1] = 1.0
            beam = (ctx.t * 0.55) % (2 * math.pi)
            for direction in (beam, beam + math.pi):
                for step in range(6, 150, 3):
                    bx = lamp_x + math.cos(direction) * step
                    by = lamp_y + math.sin(direction) * step * 0.18
                    if 0 <= int(by) < H and 0 <= int(bx) < W:
                        strength = 0.16 * (1 - step / 170) ** 1.6
                        image[int(by), int(bx)] = max(image[int(by), int(bx)], strength)

    if vista.kind in ("pines", "shore") and ctx.nightish:
        house_x = int(W * 0.70)
        house_y = vista.horizon + 8
        blit(image, CABIN, house_x, house_y, tone["ridge_near"])
        window = np.zeros((H, W), dtype=bool)
        blit(window, CABIN_WINDOW, house_x, house_y, 1.0)
        image[window] = 0.97

    if vista.kind == "meadow" and ctx.day > 0.3 and tone["snow"] < 0.2:
        for fx, fy in vista.wildflowers[:22]:
            image[int(192 + fy * 24), int(fx * W)] = 0.9

    phase = (ctx.t % 53.0) / 53.0
    if 0.02 < phase < 0.4 and tone["rain"] < 0.35 and tone["snow"] < 0.3:
        span = W * 1.3
        base_x = -30 + span * (phase - 0.02) / 0.38
        base_y = vista.horizon - 56 - math.sin(phase * 3.0) * 8
        birds = np.zeros((H, W), dtype=bool)
        for index in range(5):
            bx = base_x + index * 9 + math.sin(phase * 20 + index) * 2
            by = base_y + math.sin(phase * 16 + index * 1.7) * 5 + index * 2
            stroke(birds, bx - 3, by, bx, by + 2)
            stroke(birds, bx, by + 2, bx + 3, by)
        image[birds] = 0.04 if tone["sky_horizon"] > 0.5 else 0.9

    if ctx.nightish and vista.kind == "meadow":
        for index in range(14):
            fx = (index * 41 + int(ctx.t * 4)) % W
            fy = 178 + int(16 * abs(math.sin(index * 1.7)))
            if math.sin(ctx.t * 1.5 + index) > 0.5:
                image[fy, fx] = 0.95


# ---------------------------------------------------------------------- hud

@dataclass
class Readout:
    clock: str = "0:00"
    meridiem: str = ""
    date: str = ""
    place: str = ""
    temp: str = ""
    phrase: str = ""
    small: str = ""
    right_small: str = ""
    detail: dict = field(default_factory=dict)
    show_detail: bool = False
    hide_text: bool = False
    page: int = 0
    forecast: list = field(default_factory=list)


def scrim(image: np.ndarray) -> None:
    """Darken the corners so paper-white type always has somewhere to sit."""
    left = np.exp(-(((PIXEL_X - 20) / 150) ** 2 + ((PIXEL_YY - 268) / 88) ** 2))
    right = np.exp(-(((PIXEL_X - 388) / 158) ** 2 + ((PIXEL_YY - 262) / 92) ** 2))
    top = np.exp(-(((PIXEL_Y + 20) / 20) ** 2)) * 0.08
    shade = 1.0 - np.clip(left * 0.72 + right * 0.34 + top, 0, 0.9)
    image *= shade
    image[CONTENT_H - 2 :, :] *= 0.4


def paint_hud(image: np.ndarray, ctx: "Ctx", readout: Readout) -> tuple[np.ndarray, np.ndarray]:
    """Two masks: pixels forced to paper, and pixels forced to ink."""
    white = np.zeros((H, W), dtype=bool)
    black = np.zeros((H, W), dtype=bool)
    if readout.hide_text:
        return white, black

    f = serif()

    if readout.place:
        light_background = float(image[6:24, 6:140].mean()) > 0.5
        f.draw(black if light_background else white, 10, 7, readout.place)

    if readout.temp:
        light_background = float(image[0:34, 296:398].mean()) > 0.5
        target = black if light_background else white
        f.draw_right(target, 386, 4, readout.temp, scale=2)
        degree_mark(target, 389, 6, 5)
        f.draw_right(target, 392, 29, readout.phrase)

    f.draw(white, 10, 152, readout.clock, scale=3)
    if readout.meridiem:
        f.draw(white, 10 + f.advance(readout.clock, 3) + 4, 172, readout.meridiem)
    if readout.date:
        f.draw(white, 12, 198, readout.date)
    if readout.small:
        f.draw_right(white, 392, 160, readout.small)
    if readout.right_small:
        f.draw_right(white, 392, 180, readout.right_small)

    if readout.show_detail:
        _detail_card(image, white, readout.detail)

    return white, black


def _detail_card(image: np.ndarray, ink: np.ndarray, detail: dict) -> None:
    top, height = 26, 126
    x0, x1 = 26, W - 26
    card = np.zeros((H, W), dtype=bool)
    card[top : top + height, x0:x1] = True
    paper = THRESHOLD < 0.95
    image[card & paper] = 1.0
    image[card & ~paper] = 0.0
    border = np.zeros((H, W), dtype=bool)
    border[top : top + 1, x0:x1] = True
    border[top + height - 1 : top + height, x0:x1] = True
    border[top : top + height, x0 : x0 + 1] = True
    border[top : top + height, x1 - 1 : x1] = True
    ink[border] = True
    f = serif()
    y = top + 8
    for label, value in detail.items():
        f.draw(ink, x0 + 12, y, label, value=True)
        f.draw_right(ink, x1 - 12, y, str(value), value=True)
        y += 15


# -------------------------------------------------------------------- frame

@dataclass
class Ctx:
    t: float
    when: datetime
    sun_alt: float
    sun_az: float
    moon_alt: float
    moon_az: float
    moon_phase: float
    moon_illum: float
    day: float
    golden: float
    weather: dict
    wind: float
    snowline: bool
    nightish: bool


def compose(ctx: "Ctx", vista: Vista, readout: Readout) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tone = tones(ctx)
    image = np.zeros((H, W), dtype=np.float32)
    paint_sky(image, ctx, vista, tone)
    paint_sun_and_moon(image, ctx, vista, tone)
    paint_clouds(image, ctx, vista, tone)
    _ground(image, ctx, vista, tone)
    paint_ridges(image, ctx, vista, tone)
    paint_fog(image, ctx, vista, tone)
    paint_treeline(image, ctx, vista, tone)
    paint_water(image, ctx, vista, tone)
    paint_weather(image, ctx, vista, tone)
    paint_foreground(image, ctx, vista, tone)
    np.clip(image, 0.0, 1.0, out=image)
    scrim(image)
    image[FOOTER_TOP:, :] = 0.0
    white, black = paint_hud(image, ctx, readout)
    return image, white, black


def frame_from(image: np.ndarray, white: np.ndarray, black: np.ndarray) -> np.ndarray:
    # expand contrast: near-black becomes ink, near-white becomes paper, so the
    # dither only ever appears where the picture actually has a middle
    image = np.clip((image - 0.07) / 0.85, 0.0, 1.0)
    bits = image > THRESHOLD
    bits = (bits | white) & ~black
    bits[outline(black)] = True
    bits[outline(white)] = False
    return bits


def render(ctx: Ctx, vista: Vista, readout: Readout) -> np.ndarray:
    return frame_from(*compose(ctx, vista, readout))


def degree_mark(mask: np.ndarray, x: int, y: int, size: int = 5) -> None:
    """A tiny ring, because the device font has no degree glyph."""
    if size >= 5:
        pattern = [".###.", "#...#", "#...#", "#...#", ".###."]
    else:
        pattern = [".#.", "#.#", ".#."]
    height, width = mask.shape
    for row, line in enumerate(pattern):
        for column, cell in enumerate(line):
            if cell != "#":
                continue
            px, py = x + column, y + row
            if 0 <= px < width and 0 <= py < height:
                mask[py, px] = True


def outline(mask: np.ndarray) -> np.ndarray:
    """One-pixel dilation minus the mask itself: a paper halo for type."""
    grown = mask.copy()
    grown[1:, :] |= mask[:-1, :]
    grown[:-1, :] |= mask[1:, :]
    grown[:, 1:] |= mask[:, :-1]
    grown[:, :-1] |= mask[:, 1:]
    return grown & ~mask


def pack(bits: np.ndarray) -> bytes:
    return np.packbits(bits, axis=1).tobytes()
