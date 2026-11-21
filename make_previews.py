"""Render the preview stills and the little film used in the chat report."""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
import zlib
from datetime import datetime, timedelta

import numpy as np

import data
import render as R
from preview import png_write

LAT, LON = 40.0, -100.0
PLACE = "home"

# name, when, vista, weather code, cloud, wind, temperature, phrase
STILLS = [
    ("dawn-fog",    "2026-11-20T07:20", "pines",  45, 0.55, 4,  4.0,  "fog in the trees"),
    ("noon-lake",   "2026-11-20T12:30", "lake",    2, 0.35, 9,  16.0, "a few clouds"),
    ("dusk-ridge",  "2026-11-20T16:45", "ridge",   3, 0.85, 14, 9.0,  "under a low sky"),
    ("night-meadow","2026-11-20T23:10", "meadow",  1, 0.15, 3,  2.0,  "clear"),
]

ZERO = bytes([0])


def png_write_rgb(path, bits, scale=1):
    """RGB frames: ffmpeg reads grey PNGs as video range and crushes them."""
    grey = np.where(bits, 255, 0).astype(np.uint8)
    if scale > 1:
        grey = np.repeat(np.repeat(grey, scale, axis=0), scale, axis=1)
    rgb = np.repeat(grey[:, :, None], 3, axis=2)
    height, width, _ = rgb.shape
    raw = b"".join(ZERO + rgb[row].tobytes() for row in range(height))

    def chunk(tag, body):
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6))
    png += chunk(b"IEND", b"")
    with open(path, "wb") as handle:
        handle.write(png)


def context(when: datetime, code: int, cloud: float, wind: float, t: float) -> R.Ctx:
    sun = data.sun_position(when, LAT, LON)
    moon = data.moon_info(when)
    moon_pos = data.sun_position(when - timedelta(days=moon["phase"]), LAT, LON)
    weather = data.describe(code, cloud * 100.0, 0.0, 6.0)
    day = float(max(0.0, min(1.0, (sun["altitude"] + 2.0) / 10.0)))
    return R.Ctx(
        t=t, when=when,
        sun_alt=sun["altitude"], sun_az=sun["azimuth"],
        moon_alt=moon_pos["altitude"], moon_az=moon_pos["azimuth"],
        moon_phase=moon["phase"], moon_illum=moon["illum"],
        day=day,
        golden=float(max(0.0, min(1.0, 1 - abs(sun["altitude"] - 3) / 12))),
        weather=weather.as_dict(), wind=wind,
        snowline=weather.snow > 0.1,
        nightish=day < 0.35,
    )


def readout(when: datetime, temp_c: float, phrase: str, sunrise: str, sunset: str) -> R.Readout:
    hour = when.hour % 12 or 12
    return R.Readout(
        clock=str(hour) + ":" + ("%02d" % when.minute),
        meridiem="am" if when.hour < 12 else "pm",
        date=when.strftime("%a %d %b").lower(),
        place=PLACE,
        temp=data.format_temp(temp_c, "imperial"),
        phrase=phrase,
        small=sunrise + "  /  " + sunset,
        right_small=data.moon_name(data.moon_info(when)["phase"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="assets")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--film", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for name, when_text, vista, code, cloud, wind, temp, phrase in STILLS:
        when = datetime.fromisoformat(when_text)
        ctx = context(when, code, cloud, wind, t=26.0)
        bits = R.render(ctx, R.Vista(vista, 11), readout(when, temp, phrase, "07:03", "17:14"))
        path = os.path.join(args.out, "stillpoint-" + name + ".png")
        png_write(path, bits, args.scale)
        print("wrote", path)

    if args.film:
        frames_dir = os.path.join(args.out, "film")
        os.makedirs(frames_dir, exist_ok=True)
        when = datetime.fromisoformat("2026-11-20T11:25")
        fps, seconds = 10, 9
        for index in range(fps * seconds):
            t = 20.0 + index / fps
            ctx = context(when, 2, 0.45, 9, t)
            bits = R.render(ctx, R.Vista("lake", 11), readout(when, 14.0, "a few clouds", "07:03", "17:14"))
            png_write_rgb(os.path.join(frames_dir, "f%04d.png" % index), bits, 3)
        target = os.path.join(args.out, "stillpoint-a-lake-at-dusk.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(fps), "-i", os.path.join(frames_dir, "f%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            "-movflags", "+faststart", target,
        ], check=True)
        print("wrote", target)


if __name__ == "__main__":
    main()
