"""Tools Ari can call (stage v3+), plus the robot interface.

`SimulatedRobot` logs navigation commands instead of moving. On the real robot, implement
`RobotInterface.navigate` by publishing a goal to the ROS 2 navigation stack (e.g. a Nav2
`NavigateToPose` action) and keep everything else unchanged.

Live-data tools call free, keyless public APIs: Open-Meteo (weather) and AlAdhan (prayer times).
"""

from __future__ import annotations

import ast
import difflib
import json
import operator
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Protocol
from urllib.parse import urlencode

# EMINES / UM6P campus, Ben Guerir
CAMPUS_LAT, CAMPUS_LON = 32.2359, -7.9531
_WMO = {0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "freezing fog",
        51: "light drizzle", 53: "drizzle", 55: "dense drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
        71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers", 81: "rain showers",
        82: "violent rain showers", 95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with hail"}


def morocco_now() -> datetime:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Africa/Casablanca"))
    except Exception:  # no tz database (e.g. Windows without tzdata): Morocco is UTC+1 outside Ramadan
        return datetime.now(timezone(timedelta(hours=1)))


def fetch_json(url: str, params: dict, timeout: float = 10) -> dict:
    request = urllib.request.Request(f"{url}?{urlencode(params)}", headers={"User-Agent": "ari-agent/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, str]
    fn: Callable[..., str]

    def signature(self) -> str:
        params = "; ".join(f"{k}: {v}" for k, v in self.parameters.items()) or "no arguments"
        return f"- {self.name}: {self.description} Arguments: {params}"


class RobotInterface(Protocol):
    def navigate(self, place_key: str, place_name: str) -> dict: ...


@dataclass
class SimulatedRobot:
    commands: list[dict] = field(default_factory=list)

    def navigate(self, place_key: str, place_name: str) -> dict:
        command = {"action": "navigate", "target": place_key, "at": datetime.now().isoformat(timespec="seconds")}
        self.commands.append(command)
        return {"status": "navigation_started", "target": place_name, "simulated": True}


class ToolRegistry:
    def __init__(self, campus_file: Path, robot: RobotInterface | None = None, fetch: Callable[..., dict] = fetch_json):
        self.places: dict[str, dict] = json.loads(campus_file.read_text(encoding="utf-8"))["places"]
        self.fetch = fetch
        self.robot = robot or SimulatedRobot()
        navigable = ", ".join(p["name"] for p in self.places.values() if p.get("navigable"))
        self.tools: dict[str, Tool] = {t.name: t for t in [
            Tool("campus_directory", "Look up a place on the UM6P Ben Guerir campus (what it is, where, hours if published).",
                 {"place": "name of the place, e.g. 'learning center', 'medical center', 'pool'"}, self.campus_directory),
            Tool("campus_weather", "Live weather now and forecast for today and tomorrow on the Ben Guerir campus.",
                 {}, self.campus_weather),
            Tool("prayer_times", "Today's prayer times in Ben Guerir (Morocco calculation method).", {}, self.prayer_times),
            Tool("robot_navigate", "Physically guide the visitor to a destination. Call it for ANY 'take me to' "
                 f"request; it checks access itself and explains refusals. Reachable on foot: {navigable}.",
                 {"destination": "the place to go to"}, self.robot_navigate),
            Tool("get_time", "Get the current date and time in Morocco.", {}, self.get_time),
            Tool("calculator", "Evaluate an arithmetic expression.", {"expression": "e.g. '12*7 + 3'"}, self.calculator),
        ]}

    def describe(self) -> str:
        return "\n".join(t.signature() for t in self.tools.values())

    def run(self, name: str, args: dict) -> str:
        tool = self.tools.get(name)
        if tool is None:
            return f"ERROR: unknown tool {name!r}"
        try:
            return tool.fn(**{k: v for k, v in args.items() if k in tool.parameters})
        except TypeError as exc:
            return f"ERROR: bad arguments for {name}: {exc}"

    def _match_place(self, query: str) -> tuple[str, dict] | None:
        q = (query or "").lower().strip()
        if not q:
            return None
        for key, place in self.places.items():
            if any(alias in q or q in alias for alias in place["aliases"]):
                return key, place
        aliases = {alias: key for key, p in self.places.items() for alias in p["aliases"]}
        close = difflib.get_close_matches(q, list(aliases), n=1, cutoff=0.6)
        return (aliases[close[0]], self.places[aliases[close[0]]]) if close else None

    def campus_directory(self, place: str = "") -> str:
        match = self._match_place(place)
        if match is None:
            return f"ERROR: no place matching {place!r}. Known places: {', '.join(p['name'] for p in self.places.values())}"
        _, p = match
        return json.dumps({"name": p["name"], "where": p["where"], "hours": p["hours"],
                           "walk_minutes_from_emines": p.get("walk_minutes")}, ensure_ascii=False)

    def campus_weather(self) -> str:
        try:
            data = self.fetch("https://api.open-meteo.com/v1/forecast", {
                "latitude": CAMPUS_LAT, "longitude": CAMPUS_LON, "timezone": "Africa/Casablanca", "forecast_days": 2,
                "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min,weather_code,precipitation_probability_max"})
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            return f"ERROR: weather service unavailable ({exc})"
        now, daily = data["current"], data["daily"]
        days = [{"date": daily["time"][i], "sky": _WMO.get(daily["weather_code"][i], "variable"),
                 "min_c": daily["temperature_2m_min"][i], "max_c": daily["temperature_2m_max"][i],
                 "rain_chance_pct": daily["precipitation_probability_max"][i]} for i in range(len(daily["time"]))]
        return json.dumps({"place": "Ben Guerir campus", "now": {
            "temperature_c": now["temperature_2m"], "feels_like_c": now["apparent_temperature"],
            "sky": _WMO.get(now["weather_code"], "variable"), "wind_kmh": now["wind_speed_10m"]},
            "forecast": days, "source": "Open-Meteo"}, ensure_ascii=False)

    def prayer_times(self) -> str:
        try:
            data = self.fetch("https://api.aladhan.com/v1/timings",
                              {"latitude": CAMPUS_LAT, "longitude": CAMPUS_LON, "method": 21})["data"]
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
            return f"ERROR: prayer-times service unavailable ({exc})"
        names = ("Fajr", "Sunrise", "Dhuhr", "Asr", "Maghrib", "Isha")
        return json.dumps({"place": "Ben Guerir", "date": data["date"]["readable"],
                           "times": {n: data["timings"][n] for n in names}, "source": "AlAdhan"}, ensure_ascii=False)

    def robot_navigate(self, destination: str = "") -> str:
        match = self._match_place(destination)
        if match is None:
            return f"ERROR: unknown destination {destination!r}; ask the visitor to choose a known place."
        key, p = match
        if p.get("restricted"):
            return f"REFUSED: {p['name']} is a restricted area (authorised staff only)."
        if p.get("private"):
            return f"REFUSED: {p['name']} are private (residents only); I can only guide you to public places."
        if not p.get("navigable"):
            return f"REFUSED: {p['name']} is beyond my walking range ({p['where']}). Give directions instead."
        status = self.robot.navigate(key, p["name"])
        return json.dumps({**status, "eta_minutes": p.get("walk_minutes")}, ensure_ascii=False)

    @staticmethod
    def get_time() -> str:
        return morocco_now().strftime("%A %d %B %Y, %H:%M (Morocco time)")

    @staticmethod
    def calculator(expression: str = "") -> str:
        ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
               ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Mod: operator.mod}

        def ev(node: ast.AST) -> float:
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.BinOp) and type(node.op) in ops:
                if isinstance(node.op, ast.Pow) and abs(ev(node.right)) > 100:
                    raise ValueError("exponent too large")
                return ops[type(node.op)](ev(node.left), ev(node.right))
            if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
                return ops[type(node.op)](ev(node.operand))
            raise ValueError("only arithmetic is allowed")

        try:
            value = ev(ast.parse(str(expression).replace("x", "*").replace("^", "**"), mode="eval").body)
        except (SyntaxError, ValueError, ZeroDivisionError) as exc:
            return f"ERROR: {exc}"
        return str(round(value, 6))
