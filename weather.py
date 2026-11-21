#!/usr/bin/env python3
"""Stillpoint — a quiet window on the weather, for the Lapis one-bit screen.

A native on-device app: it opens a window through hwserviced's AppWindowManager,
draws 400x240 1-bit frames, and answers the device's own buttons and gestures.
"""

from __future__ import annotations

import argparse
import fcntl
import math
import os
import re
import select
import signal
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

# Gio must be imported before numpy: this PyGObject build registers its
# marshallers during introspection import, and loading numpy first leaves
# DBusProxy unable to marshal an object path.
try:
    from gi.repository import Gio, GLib
except ImportError:  # offline preview mode does not need the bus
    Gio = None
    GLib = None

import numpy as np

import data
import render as R

SERVICE = "ai.pamir.Lapis1"
MANAGER_PATH = "/ai.pamir/Lapis1/AppWindowManager"
MANAGER_IFACE = "ai.pamir.Lapis1.AppWindowManager"

WIRE_MAGIC = b"LPFW"
WIRE_VERSION = 1
HEADER_BYTES = 56
HEADER = struct.Struct("<4sHHHHIIIQQQII")

FRAME_BYTES = 12_000
FRAME_FLAGS = 1 | 2 | 4 | 8      # full frame, back input, leaves back, takes input

KIND_VISIBILITY = 2
KIND_REQUESTED = 3
KIND_PRESENTED = 4
KIND_DROPPED = 5
KIND_CLOSING = 6
KIND_BACK = 7
KIND_INPUT = 8

INPUT_A_DOWN = 1
INPUT_A_UP = 2
INPUT_FINGERPRINT_DOWN = 3
INPUT_FINGERPRINT_UP = 4
INPUT_SWIPE_UP = 5
INPUT_SWIPE_DOWN = 6
INPUT_SWIPE_LEFT = 7
INPUT_SWIPE_RIGHT = 8
INPUT_TAP = 9
INPUT_TAP_HOLD = 10

ROTATE_SECONDS = 240.0
FADE_SECONDS = 6.5


def log(message: str) -> None:
    print(message, flush=True)


# --------------------------------------------------------------------- world

@dataclass
class Reading:
    weather: data.Weather
    sunrise: datetime | None
    sunset: datetime | None
    fetched_at: float
    hours: list


class Stillpoint:
    def __init__(self, args):
        self.args = args
        self.config = data.load_config()
        if args.place:
            self.config["place"] = args.place
        self.units = self.config.get("units", "imperial")
        self.lat = float(self.config["latitude"])
        self.lon = float(self.config["longitude"])
        self.place = str(self.config.get("place", "home"))
        self.drift_hours = float(getattr(args, "drift", 0.0) or 0.0)
        self.hide_text = False
        self.show_detail = False
        self.paused = False
        self.rotation = float(self.config.get("rotate_seconds", ROTATE_SECONDS))
        initial = args.vista if args.vista not in (None, "auto") else self.config.get("vista", "lake")
        if initial not in R.VISTAS:
            initial = "lake"
        self.vista_kind = initial
        self.vista = R.Vista(self.vista_kind, int(time.time()) % 997)
        self.previous_vista = None
        self.previous_kind = None
        self.fade_start = 0.0
        self.scene_started = time.monotonic()
        self.manual_until = 0.0
        self.have_weather = False
        self.last_target = None
        self.last_checked = 0.0
        self.reading: Reading | None = None
        self.frame_count = 0
        self.last_frame = b"\x00" * FRAME_BYTES
        self.started = time.monotonic()
        self.dump_budget = int(getattr(args, "dump", 0) or 0)
        self.dump_dir = str(getattr(args, "dump_dir", "/tmp/stillpoint-frames"))
        if self.dump_budget:
            os.makedirs(self.dump_dir, exist_ok=True)

        self.window_id = 0
        self.window_generation = 0
        self.visibility_generation = 0
        self.visibility_token = 0
        self.visible = False
        self.sequence = 0
        self.pending_request = False
        self.socket: socket.socket | None = None
        self.bus = None
        self.proxy = None
        self.loop = None

    # ---------------------------------------------------------------- weather
    def refresh_weather(self) -> None:
        try:
            weather, hours, sun = data.fetch_weather(self.lat, self.lon)
            self.reading = Reading(
                weather=weather,
                sunrise=sun.get("sunrise"),
                sunset=sun.get("sunset"),
                fetched_at=time.monotonic(),
                hours=hours,
            )
            log("weather: " + weather.phrase + ", " + str(round(weather.temp_c)) + "C")
            if not self.have_weather:
                self.have_weather = True
                ctx = self.build_ctx(time.monotonic() - self.started)
                target = self.choose_vista(ctx)
                if target != self.vista_kind:
                    self.advance_vista(forced=target, manual=False)
        except Exception as error:  # keep the last good reading
            log("weather fetch failed: " + repr(error))

    def discover_place(self) -> None:
        if self.place not in ("", "home"):
            return
        found = data.discover_place()
        if not found:
            return
        self.lat = found["latitude"]
        self.lon = found["longitude"]
        self.place = found["place"]
        self.config.update({"latitude": self.lat, "longitude": self.lon, "place": self.place})
        data.save_config(self.config)
        log("found a place: " + self.place)

    def start_weather_loop(self) -> None:
        def loop():
            while True:
                self.refresh_weather()
                time.sleep(900)
        threading.Thread(target=loop, daemon=True).start()
        threading.Thread(target=self.discover_place, daemon=True).start()

    # ------------------------------------------------------------------ state
    def now(self) -> datetime:
        return datetime.now() + timedelta(hours=self.drift_hours)

    def weather_now(self) -> data.Weather:
        when = self.now()
        if self.reading is None:
            return data.Weather(code=2, phrase="quiet weather", cloud=0.3)
        weather = self.reading.weather
        if abs(self.drift_hours) >= 0.5 and self.reading.hours:
            hour = data.hour_at(self.reading.hours, when)
            if hour is not None:
                weather = data.describe(hour["code"], hour["cloud"], hour["precip"] or 0.0, hour["temp_c"])
                weather.wind_kmh = hour["wind"] or weather.wind_kmh
                weather.temp_c = hour["temp_c"]
        return weather

    def build_ctx(self, t: float) -> R.Ctx:
        when = self.now()
        sun = data.sun_position(when, self.lat, self.lon)
        moon = data.moon_info(when)
        moon_pos = data.sun_position(when - timedelta(days=moon["phase"]), self.lat, self.lon)
        weather = self.weather_now()
        day = float(np.clip((sun["altitude"] + 2.0) / 10.0, 0.0, 1.0))
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
            wind=weather.wind_kmh / 3.6,
            snowline=weather.snow > 0.1,
            nightish=day < 0.35,
        )

    def readout(self, ctx: R.Ctx, weather: data.Weather) -> R.Readout:
        when = self.now()
        hour12 = when.hour % 12 or 12
        sunrise = self.reading.sunrise if self.reading else None
        sunset = self.reading.sunset if self.reading else None
        sunrise_text = sunrise.strftime("%H:%M") if sunrise else ""
        sunset_text = sunset.strftime("%H:%M") if sunset else ""
        moon = data.moon_info(when)
        moon_text = data.moon_name(moon["phase"])
        battery = data.battery()
        detail = {
            "temperature": data.format_temp(weather.temp_c, self.units),
            "feels like": data.format_temp(weather.apparent_c, self.units),
            "wind": str(int(round(weather.wind_kmh))) + " km/h",
            "humidity": str(int(round(weather.humidity))) + "%",
            "clouds": str(int(round(weather.cloud * 100))) + "%",
            "sunrise": sunrise_text,
            "sunset": sunset_text,
            "moon": moon_text,
            "power": (str(battery["percent"]) + "%" if battery["percent"] is not None else "unknown"),
        }
        return R.Readout(
            clock=str(hour12) + ":" + ("%02d" % when.minute),
            meridiem="am" if when.hour < 12 else "pm",
            date=when.strftime("%a %d %b").lower(),
            place=self.place,
            temp=data.format_temp(weather.temp_c, self.units),
            phrase=weather.phrase,
            small=sunrise_text + "  /  " + sunset_text,
            right_small=moon_text,
            detail=detail,
            hide_text=self.hide_text,
            show_detail=self.show_detail,
        )

    # ----------------------------------------------------------------- vista
    def choose_vista(self, ctx: R.Ctx) -> str:
        """Which view this weather wants to be seen from."""
        weather = ctx.weather
        snow = weather.get("snow", 0.0)
        fog = weather.get("fog", 0.0)
        rain = weather.get("rain", 0.0)
        thunder = weather.get("thunder", 0.0)
        cloud = weather.get("cloud", 0.0)

        if snow > 0.22:
            return "pines"          # snow caught in the trees
        if fog > 0.33:
            return "lake"           # fog lying on the water
        if thunder > 0.15 or rain > 0.55:
            return "ridge"          # weather to watch from a distance
        if rain > 0.18:
            return "pines"          # shelter under the trees
        if ctx.nightish and cloud < 0.4:
            return "ridge"          # stars over the high ground
        if cloud > 0.88 and ctx.wind > 5.5:
            return "shore"          # a low sky and wind off the water
        if ctx.wind > 8.0:
            return "shore"          # wind everywhere else too
        if ctx.day > 0.75 and cloud < 0.45:
            return "meadow"         # a bright, open day
        if cloud > 0.75:
            return "lake"           # still air under a heavy sky: a mirror
        return "lake"

    def advance_vista(self, delta: int = 1, forced: str | None = None, manual: bool = True) -> None:
        order = ["lake", "ridge", "meadow", "pines", "shore"]
        if forced:
            kind = forced
        else:
            kind = order[(order.index(self.vista_kind) + delta) % len(order)]
        if kind == self.vista_kind:
            return
        self.previous_kind = self.vista_kind
        self.previous_vista = self.vista
        self.vista_kind = kind
        self.vista = R.Vista(kind, int(time.time() * 1000) % 9973)
        self.fade_start = time.monotonic()
        self.scene_started = time.monotonic()
        self.manual_until = 0.0
        self.have_weather = False
        self.last_target = None
        self.last_checked = 0.0
        log("vista: " + kind)

    # ----------------------------------------------------------------- frame
    def paint(self) -> bytes:
        now = time.monotonic()
        t = now - self.started
        ctx = self.build_ctx(t)
        weather = self.weather_now()
        readout = self.readout(ctx, weather)
        image, white, black = R.compose(ctx, self.vista, readout)
        if self.previous_vista is not None:
            progress = (now - self.fade_start) / FADE_SECONDS
            if progress >= 1.0:
                self.previous_vista = None
            else:
                blend = float(np.clip(progress, 0, 1))
                blend = blend * blend * (3.0 - 2.0 * blend)
                old_image, _, _ = R.compose(ctx, self.previous_vista, R.Readout(hide_text=True))
                image = old_image * (1 - blend) + image * blend
        bits = R.frame_from(image, white, black)
        if self.dump_budget > 0:
            self.dump_budget -= 1
            try:
                from preview import png_write
                png_write(self.dump_dir + "/frame-" + str(1000 - self.dump_budget) + ".png", bits, 1)
            except Exception as error:
                log("dump failed: " + repr(error))
        return np.packbits(bits, axis=1).tobytes()

    # ------------------------------------------------------------------ bus
    # Literal strings only: this PyGObject build marshals module constants
    # into the wrong positional slot of the proxy constructors.
    def connect(self) -> None:
        self.bus = Gio.bus_get_sync(Gio.BusType.SYSTEM, None)
        self.proxy = Gio.DBusProxy.new_sync(
            connection=self.bus,
            flags=Gio.DBusProxyFlags.NONE,
            info=None,
            name="ai.pamir.Lapis1",
            object_path="/ai/pamir/Lapis1/AppWindowManager",
            interface_name="ai.pamir.Lapis1.AppWindowManager",
            cancellable=None,
        )
        capabilities = self.call("GetInputCapabilities", "()", ()).unpack()[0]
        log("input capabilities: " + str(capabilities))

        # one window at a time: if a Stillpoint window already exists, stand down
        windows = self.call("ListWindows", "()", ()).unpack()[0]
        for path, generation, pid, title, icon in windows:
            if title.lower() == self.args.title.lower():
                log("already running as " + path + " (pid " + str(pid) + "); exiting")
                return False

        server_end, client_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        server_end.setblocking(False)
        client_end.setblocking(False)
        fd_list = Gio.UnixFDList.new()
        fd_list.append(server_end.fileno())
        reply = self.call(
            "CreateWindow", "(ssh)", (str(self.args.title), str(self.args.icon), 0), fds=fd_list,
        )
        path = reply.unpack()[0]
        match = re.search(r"/w(\d+)/g(\d+)$", path)
        if not match:
            raise RuntimeError("unexpected window path " + path)
        self.window_id = int(match.group(1))
        self.window_generation = int(match.group(2))
        server_end.close()
        self.socket = client_end
        log("window " + path + " (id " + str(self.window_id) + ")")
        return True

    def call(self, method, signature, arguments, fds=None):
        params = GLib.Variant(str(signature), arguments)
        if fds is None:
            return self.proxy.call_sync(
                method_name=str(method),
                parameters=params,
                flags=Gio.DBusCallFlags.NONE,
                timeout_msec=5000,
                cancellable=None,
            )
        reply, _ = self.proxy.call_with_unix_fd_list_sync(
            method_name=str(method),
            parameters=params,
            flags=Gio.DBusCallFlags.NONE,
            timeout_msec=5000,
            fd_list=fds,
            cancellable=None,
        )
        return reply

    # --------------------------------------------------------------- packets
    def send_frame(self) -> None:
        if self.socket is None or not self.visible:
            return
        self.sequence += 1
        payload = self.paint()
        packet = HEADER.pack(
            WIRE_MAGIC,
            WIRE_VERSION,
            1,
            HEADER_BYTES,
            FRAME_FLAGS,
            self.window_id,
            self.window_generation,
            self.visibility_generation,
            self.sequence,
            self.visibility_token,
            0,
            len(payload),
            0,
        ) + payload
        try:
            self.socket.send(packet)
            self.frame_count += 1
        except BlockingIOError:
            self.sequence -= 1
        except OSError as error:
            log("send failed: " + repr(error))

    def on_readable(self, fd, condition):
        while True:
            try:
                packet = self.socket.recv(4096)
            except BlockingIOError:
                return True
            except OSError:
                return False
            if not packet:
                return False
            self.handle(packet)

    def handle(self, packet: bytes) -> None:
        if len(packet) < HEADER_BYTES:
            return
        magic, version, kind, header_len, flags, window_id, generation, visibility_gen, sequence, token, argument, payload_len, reserved = HEADER.unpack(
            packet[:HEADER_BYTES]
        )
        if magic != WIRE_MAGIC:
            return
        if kind == KIND_VISIBILITY:
            self.visibility_generation = visibility_gen
            visible = bool(flags & 1)
            self.visible = visible
            self.visibility_token = token if visible else 0
            if visible:
                self.sequence = 0
                log("visible (reason " + str(flags >> 8) + ")")
        elif kind == KIND_REQUESTED:
            if self.visible:
                self.send_frame()
        elif kind == KIND_PRESENTED:
            pass
        elif kind == KIND_DROPPED:
            pass
        elif kind == KIND_CLOSING:
            log("closing (reason " + str(flags) + ")")
            self.quit()
        elif kind == KIND_BACK:
            pass
        elif kind == KIND_INPUT:
            code = flags & 0xFF
            self.on_input(code)

    def on_input(self, code: int) -> None:
        log("input " + str(code))
        if code == INPUT_A_DOWN:
            self.advance_vista(1)
        elif code == INPUT_SWIPE_LEFT:
            self.advance_vista(-1)
        elif code == INPUT_SWIPE_RIGHT:
            self.advance_vista(1)
        elif code == INPUT_SWIPE_UP:
            self.show_detail = True
        elif code == INPUT_SWIPE_DOWN:
            self.show_detail = False
        elif code == INPUT_TAP:
            self.hide_text = not self.hide_text
        elif code == INPUT_TAP_HOLD:
            pass  # free for later

    # ------------------------------------------------------------------ loop
    def tick(self) -> bool:
        now = time.monotonic()
        if self.paused or now < self.manual_until:
            return True
        if now - self.last_checked < 60.0:
            return True
        self.last_checked = now
        ctx = self.build_ctx(now - self.started)
        target = self.choose_vista(ctx)
        # move only once the same view has been asked for twice, and never
        # sooner than a few minutes after the last change
        if target == self.last_target and target != self.vista_kind and now - self.scene_started > 150.0:
            self.advance_vista(forced=target, manual=False)
        self.last_target = target
        return True

    def quit(self) -> None:
        if self.loop is not None:
            self.loop.quit()

    def run(self) -> int:
        if not self.connect():
            return 0
        self.start_weather_loop()
        self.loop = GLib.MainLoop()
        GLib.io_add_watch(
            self.socket.fileno(),
            GLib.IO_IN | GLib.IO_HUP | GLib.IO_ERR,
            self.on_readable,
        )
        GLib.timeout_add(1000, self.tick)
        signal.signal(signal.SIGTERM, lambda *_: self.loop.quit())
        signal.signal(signal.SIGINT, lambda *_: self.loop.quit())
        log("weather running at " + str(self.lat) + ", " + str(self.lon))
        self.loop.run()
        log("stillpoint stopped after " + str(self.frame_count) + " frames")
        return 0


def preview(args) -> int:
    """Offline: render one frame straight to a PNG."""
    from preview import png_write
    app = Stillpoint(args)
    when = app.now()
    ctx = app.build_ctx(12.0)
    weather = app.weather_now()
    bits = R.render(ctx, app.vista, app.readout(ctx, weather))
    png_write(args.out, bits, args.scale)
    print("wrote " + args.out)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Stillpoint, an ambient weather window")
    parser.add_argument("--vista", default="auto", choices=["auto", "lake", "pines", "ridge", "meadow", "shore"])
    parser.add_argument("--place", default=None)
    parser.add_argument("--drift", type=float, default=0.0, help="hours to drift through the day")
    parser.add_argument("--icon", default="sun")
    parser.add_argument("--title", default="Weather")
    parser.add_argument("--preview", default=None, help="write a PNG here and exit")
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--out", default="/tmp/stillpoint.png")
    parser.add_argument("--dump", type=int, default=0, help="write this many frames as PNGs")
    parser.add_argument("--dump-dir", default="/tmp/stillpoint-frames")
    args = parser.parse_args()
    try:
        with open("/proc/self/comm", "w") as handle:
            handle.write("stillpoint")
    except OSError:
        pass
    if args.preview:
        return preview(args)
    if Gio is None:
        print("Gio is not available; run with /usr/bin/python3 and PYTHONPATH set", file=sys.stderr)
        return 1

    # one window, one process: a second start quietly stands down
    guard = open("/tmp/weather.lock", "w")
    try:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("stillpoint is already running", file=sys.stderr)
        return 0
    guard.write(str(os.getpid()) + "\n")
    guard.flush()

    return Stillpoint(args).run()


if __name__ == "__main__":
    sys.exit(main())
