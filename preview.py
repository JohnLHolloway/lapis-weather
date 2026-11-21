"""Render Stillpoint frames to PNG without the device, for looking at."""

from __future__ import annotations

import argparse
import struct
import zlib
from datetime import datetime

import numpy as np

import data
import render as R


def png_write(path: str, bits: np.ndarray, scale: int = 1, invert: bool = False) -> None:
    grey = np.where(bits, 255, 0).astype(np.uint8)
    if invert:
        grey = 255 - grey
    if scale > 1:
        grey = np.repeat(np.repeat(grey, scale, axis=0), scale, axis=1)
    height, width = grey.shape
    raw = b"".join(b"\x00" + grey[row].tobytes() for row in range(height))

    def chunk(tag: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body))
            + tag
            + body
            + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)
        )

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9))
    png += chunk(b"IEND", b"")
    with open(path, "wb") as handle:
        handle.write(png)


def make_ctx(when: datetime, code: int, cloud: float, wind: float, lat: float, lon: float, t: float = 0.0) -> R.Ctx:
    sun = data.sun_position(when, lat, lon)
    moon = data.moon_info(when)
    moon_pos = data.sun_position(when - __import__("datetime").timedelta(days=moon["phase"]), lat, lon) if False else None
    # the moon rides the sun's own path, lagged by its phase
    from datetime import timedelta
    moon_pos = data.sun_position(when - timedelta(days=moon["phase"]), lat, lon)
    weather = data.describe(code, cloud * 100.0, 0.0, 6.0)
    day = data.daylight(sun["altitude"])
    golden = float(np.clip(1 - abs(sun["altitude"] - 3) / 12, 0, 1))
    return R.Ctx(
        t=t,
        when=when,
        sun_alt=sun["altitude"],
        sun_az=sun["azimuth"],
        moon_alt=moon_pos["altitude"],
        moon_az=moon_pos["azimuth"],
        moon_phase=moon["phase"],
        moon_illum=moon["illum"],
        day=day,
        golden=golden,
        weather=weather.as_dict(),
        wind=wind,
        snowline=weather.snow > 0.1,
        nightish=day < 0.35,
    )


def readout(when: datetime, temp_c: float, phrase: str, place: str) -> R.Readout:
    hour = when.hour % 12 or 12
    return R.Readout(
        clock=str(hour) + ":" + ("%02d" % when.minute),
        meridiem="am" if when.hour < 12 else "pm",
        date=when.strftime("%a %d %b").lower(),
        place=place,
        temp=data.format_temp(temp_c, "imperial"),
        phrase=phrase,
        small="sunrise 6:48",
        right_small="moon waxing gibbous",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/stillpoint.png")
    parser.add_argument("--when", default=None)
    parser.add_argument("--vista", default="lake")
    parser.add_argument("--code", type=int, default=2)
    parser.add_argument("--cloud", type=float, default=0.45)
    parser.add_argument("--wind", type=float, default=8.0)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--t", type=float, default=12.0)
    args = parser.parse_args()

    when = datetime.fromisoformat(args.when) if args.when else datetime.now()
    ctx = make_ctx(when, args.code, args.cloud, args.wind, 40.0, -100.0, args.t)
    vista = R.Vista(args.vista, 3)
    bits = R.render(ctx, vista, readout(when, 13.0, "a low sky", "home"))
    png_write(args.out, bits, args.scale)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
