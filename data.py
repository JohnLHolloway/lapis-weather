"""Weather, sky and battery: the slow-changing facts of the day."""

from __future__ import annotations

import json
import math
import pathlib
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta

CONFIG_PATH = pathlib.Path(__file__).resolve().parent / "config.json"

RAD = math.pi / 180.0
SYNODIC = 29.530588853
KNOWN_NEW_MOON = datetime(2000, 1, 6, 18, 14, tzinfo=None)


# ---------------------------------------------------------------- astronomy

def sun_position(when: datetime, lat: float, lon: float) -> dict:
    """NOAA low-precision solar position, in degrees."""
    jd = when.timestamp() / 86400.0 + 2440587.5
    n = jd - 2451545.0
    mean_lon = (280.460 + 0.9856474 * n) % 360
    anomaly = ((357.528 + 0.9856003 * n) % 360) * RAD
    lam = (mean_lon + 1.915 * math.sin(anomaly) + 0.020 * math.sin(2 * anomaly)) * RAD
    eps = (23.439 - 0.0000004 * n) * RAD
    dec = math.asin(math.sin(eps) * math.sin(lam))
    ra = math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam))
    gmst = (18.697374558 + 24.06570982441908 * n) % 24
    lst = (gmst + lon / 15.0) % 24
    hour_angle = lst * 15.0 * RAD - ra
    lat_r = lat * RAD
    sin_alt = math.sin(lat_r) * math.sin(dec) + math.cos(lat_r) * math.cos(dec) * math.cos(hour_angle)
    altitude = math.asin(max(-1.0, min(1.0, sin_alt)))
    cos_az = (math.sin(dec) - math.sin(lat_r) * sin_alt) / (math.cos(lat_r) * math.cos(altitude) or 1e-6)
    azimuth = math.degrees(math.acos(max(-1.0, min(1.0, cos_az))))
    if math.sin(hour_angle) > 0:
        azimuth = 360.0 - azimuth
    return {
        "altitude": math.degrees(altitude),
        "azimuth": azimuth,
        "hour_angle": math.degrees(hour_angle),
    }


def moon_info(when: datetime) -> dict:
    days = (when - KNOWN_NEW_MOON).total_seconds() / 86400.0
    age = days % SYNODIC
    phase = age / SYNODIC
    illum = (1 - math.cos(phase * 2 * math.pi)) / 2
    return {"age": age, "phase": phase, "illum": illum, "waxing": phase < 0.5}


def moon_name(phase: float) -> str:
    if phase < 0.02 or phase > 0.98:
        return "new moon"
    if phase < 0.23:
        return "waxing crescent"
    if phase < 0.27:
        return "first quarter"
    if phase < 0.48:
        return "waxing gibbous"
    if phase < 0.52:
        return "full moon"
    if phase < 0.73:
        return "waning gibbous"
    if phase < 0.77:
        return "last quarter"
    return "waning crescent"


def daylight(altitude: float) -> float:
    return clamp((altitude + 6.0) / 14.0, 0.0, 1.0)


def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


# ------------------------------------------------------------------ weather

WMO = {
    0: ("clear", 0.04, 0.0, 0.0, 0.0, 0.0),
    1: ("mostly clear", 0.20, 0.0, 0.0, 0.0, 0.0),
    2: ("a low sky", 0.55, 0.0, 0.0, 0.0, 0.0),
    3: ("overcast", 0.95, 0.0, 0.0, 0.0, 0.0),
    45: ("fog", 0.55, 0.0, 0.0, 0.85, 0.0),
    48: ("rime fog", 0.55, 0.0, 0.0, 0.8, 0.0),
    51: ("soft drizzle", 0.75, 0.3, 0.0, 0.0, 0.0),
    53: ("drizzle", 0.85, 0.45, 0.0, 0.0, 0.0),
    55: ("steady drizzle", 0.95, 0.6, 0.0, 0.0, 0.0),
    56: ("freezing drizzle", 0.9, 0.4, 0.0, 0.0, 0.0),
    57: ("freezing drizzle", 0.95, 0.55, 0.0, 0.0, 0.0),
    61: ("light rain", 0.9, 0.5, 0.0, 0.0, 0.0),
    63: ("rain", 0.97, 0.75, 0.0, 0.0, 0.0),
    65: ("heavy rain", 1.0, 1.0, 0.0, 0.0, 0.0),
    66: ("freezing rain", 0.95, 0.7, 0.0, 0.0, 0.0),
    67: ("freezing rain", 1.0, 0.9, 0.0, 0.0, 0.0),
    71: ("light snow", 0.85, 0.0, 0.4, 0.0, 0.0),
    73: ("snow", 0.95, 0.0, 0.7, 0.0, 0.0),
    75: ("heavy snow", 1.0, 0.0, 1.0, 0.0, 0.0),
    77: ("snow grains", 0.8, 0.0, 0.3, 0.0, 0.0),
    80: ("passing showers", 0.8, 0.45, 0.0, 0.0, 0.0),
    81: ("showers", 0.9, 0.65, 0.0, 0.0, 0.0),
    82: ("heavy showers", 1.0, 0.95, 0.0, 0.0, 0.0),
    85: ("snow showers", 0.9, 0.0, 0.6, 0.0, 0.0),
    86: ("heavy snow showers", 1.0, 0.0, 0.9, 0.0, 0.0),
    95: ("thunder", 1.0, 0.7, 0.0, 0.0, 1.0),
    96: ("thunder and hail", 1.0, 0.8, 0.0, 0.0, 1.0),
    99: ("thunder and hail", 1.0, 1.0, 0.0, 0.0, 1.0),
}


@dataclass
class Weather:
    code: int = 0
    phrase: str = "clear"
    temp_c: float = 12.0
    apparent_c: float = 12.0
    cloud: float = 0.05
    rain: float = 0.0
    snow: float = 0.0
    fog: float = 0.0
    thunder: float = 0.0
    wind_kmh: float = 0.0
    wind_dir: float = 0.0
    humidity: float = 50.0
    is_day: bool = True
    when: datetime = field(default_factory=datetime.now)

    def as_dict(self) -> dict:
        return {
            "cloud": self.cloud,
            "rain": self.rain,
            "snow": self.snow,
            "fog": self.fog,
            "thunder": self.thunder,
        }


def describe(code: int, cloud_pct: float | None, precip: float, temp_c: float) -> Weather:
    phrase, cloud, rain, snow, fog, thunder = WMO.get(code, ("quiet weather", 0.4, 0.0, 0.0, 0.0, 0.0))
    if cloud_pct is not None:
        cloud = clamp(cloud_pct / 100.0, 0.0, 1.0)
    if precip > 0.05 and temp_c < 1.2:
        snow = max(snow, clamp(precip / 4.0, 0.2, 1.0))
        rain = 0.0
    if rain > 0.2:
        cloud = max(cloud, 0.75)
    if snow > 0.2:
        cloud = max(cloud, 0.8)
    return Weather(
        code=code, phrase=phrase, temp_c=temp_c, cloud=cloud,
        rain=rain, snow=snow, fog=fog, thunder=thunder,
    )


FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_weather(lat: float, lon: float, timeout: float = 12.0) -> tuple[Weather, list[dict], dict]:
    params = (
        "latitude=" + ("%.4f" % lat) + "&longitude=" + ("%.4f" % lon)
        + "&current=temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,"
          "weather_code,cloud_cover,wind_speed_10m,wind_direction_10m,is_day"
        + "&hourly=temperature_2m,weather_code,cloud_cover,precipitation,wind_speed_10m,is_day"
        + "&daily=sunrise,sunset"
        + "&timezone=auto&forecast_days=2&wind_speed_unit=kmh&temperature_unit=celsius"
    )
    request = urllib.request.Request(
        FORECAST_URL + "?" + params, headers={"User-Agent": "stillpoint/1.0"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode())

    current = payload.get("current", {})
    weather = describe(
        int(current.get("weather_code", 0)),
        current.get("cloud_cover"),
        float(current.get("precipitation") or 0.0),
        float(current.get("temperature_2m") or 12.0),
    )
    weather.apparent_c = float(current.get("apparent_temperature") or weather.temp_c)
    weather.wind_kmh = float(current.get("wind_speed_10m") or 0.0)
    weather.wind_dir = float(current.get("wind_direction_10m") or 0.0)
    weather.humidity = float(current.get("relative_humidity_2m") or 50.0)
    weather.is_day = bool(current.get("is_day", 1))
    weather.when = datetime.now()

    hourly = payload.get("hourly", {})
    hours = []
    for index, stamp in enumerate(hourly.get("time", [])):
        hours.append({
            "time": datetime.fromisoformat(stamp),
            "temp_c": hourly["temperature_2m"][index],
            "code": hourly["weather_code"][index],
            "cloud": hourly["cloud_cover"][index],
            "precip": hourly["precipitation"][index],
            "wind": hourly["wind_speed_10m"][index],
            "is_day": bool(hourly["is_day"][index]),
        })

    daily = payload.get("daily", {})
    sun = {}
    if daily.get("sunrise"):
        sun["sunrise"] = datetime.fromisoformat(daily["sunrise"][0])
        sun["sunset"] = datetime.fromisoformat(daily["sunset"][0])
    return weather, hours, sun


def hour_at(hours: list[dict], when: datetime) -> dict | None:
    if not hours:
        return None
    return min(hours, key=lambda row: abs((row["time"] - when).total_seconds()))


# ------------------------------------------------------------------ battery

def battery() -> dict:
    out = {"percent": None, "charging": False}
    base = pathlib.Path("/sys/class/power_supply")
    if not base.exists():
        return out
    for entry in sorted(base.iterdir()):
        try:
            if (entry / "type").read_text().strip() != "Battery":
                continue
            out["percent"] = int((entry / "capacity").read_text().strip())
            status = (entry / "status").read_text().strip().lower()
            out["charging"] = status in ("charging", "full")
            break
        except Exception:
            continue
    return out


# ------------------------------------------------------------------- config

DEFAULT_CONFIG = {
    "latitude": 40.0,
    "longitude": -100.0,
    "place": "home",
    "units": "imperial",
    "vista": "auto",
    "auto_rotate": True,
}


def load_config() -> dict:
    config = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            config.update(json.loads(CONFIG_PATH.read_text()))
        except Exception:
            pass
    return config


def save_config(config: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n")


def discover_place() -> dict | None:
    """Seed a location from the network address, once."""
    try:
        request = urllib.request.Request("https://ipwho.is/", headers={"User-Agent": "stillpoint/1.0"})
        with urllib.request.urlopen(request, timeout=6) as response:
            info = json.loads(response.read().decode())
        if not info.get("success", True):
            return None
        return {
            "latitude": float(info["latitude"]),
            "longitude": float(info["longitude"]),
            "place": str(info.get("city") or "home").lower(),
        }
    except Exception:
        return None


def format_temp(temp_c: float, units: str) -> str:
    if units == "metric":
        return str(int(round(temp_c))) + "\u00b0"
    return str(int(round(temp_c * 9 / 5 + 32))) + "\u00b0"


def compass(degrees: float) -> str:
    names = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]
    return names[int((degrees + 22.5) % 360 // 45)]
