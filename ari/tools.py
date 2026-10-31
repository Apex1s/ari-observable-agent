"""Tools Ari can call (stage v3+), plus the robot interface.

`SimulatedRobot` logs navigation commands instead of moving. On the real robot, implement
`RobotInterface.navigate` by publishing a goal to the ROS 2 navigation stack (e.g. a Nav2
`NavigateToPose` action) and keep everything else unchanged.
"""

from __future__ import annotations

import ast
import difflib
import json
import operator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Protocol


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
    def __init__(self, campus_file: Path, schedule_file: Path, robot: RobotInterface | None = None):
        self.places: dict[str, dict] = json.loads(campus_file.read_text(encoding="utf-8"))["places"]
        self.programs: dict[str, dict] = json.loads(schedule_file.read_text(encoding="utf-8"))["programs"]
        self.robot = robot or SimulatedRobot()
        navigable = ", ".join(p["name"] for p in self.places.values() if p.get("navigable"))
        self.tools: dict[str, Tool] = {t.name: t for t in [
            Tool("campus_directory", "Find where a place on campus is (building, floor, opening hours).",
                 {"place": "name of the place, e.g. 'library', 'IT help desk'"}, self.campus_directory),
            Tool("class_schedule", "Get the class timetable of a program for a day.",
                 {"program": f"one of {', '.join(self.programs)}", "day": "weekday name, 'today' or 'tomorrow'"},
                 self.class_schedule),
            Tool("robot_navigate", f"Physically guide the visitor to a destination. Reachable destinations: {navigable}.",
                 {"destination": "the place to go to"}, self.robot_navigate),
            Tool("get_time", "Get the current date and time.", {}, self.get_time),
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
        return json.dumps({"name": p["name"], "building": p["building"], "floor": p["floor"], "hours": p["hours"]}, ensure_ascii=False)

    def class_schedule(self, program: str = "1A", day: str = "today") -> str:
        program = next((p for p in self.programs if p.lower() == str(program).lower()), None)
        if program is None:
            return f"ERROR: unknown program. Available: {', '.join(self.programs)}"
        day = str(day).lower().strip()
        if day in ("today", "aujourd'hui"):
            day = datetime.now().strftime("%A").lower()
        elif day in ("tomorrow", "demain"):
            day = (datetime.now() + timedelta(days=1)).strftime("%A").lower()
        classes = self.programs[program].get(day, [])
        return json.dumps({"program": program, "day": day, "classes": classes or "no classes"}, ensure_ascii=False)

    def robot_navigate(self, destination: str = "") -> str:
        match = self._match_place(destination)
        if match is None:
            return f"ERROR: unknown destination {destination!r}; ask the visitor to choose a known place."
        key, p = match
        if p.get("restricted"):
            return f"REFUSED: {p['name']} is a restricted area (badge access only)."
        if not p.get("navigable"):
            return f"REFUSED: {p['name']} is on the {p['floor']} of building {p['building']}; I cannot climb stairs. Give directions instead."
        status = self.robot.navigate(key, p["name"])
        return json.dumps({**status, "eta_minutes": p.get("walk_minutes")}, ensure_ascii=False)

    @staticmethod
    def get_time() -> str:
        return datetime.now().strftime("%A %d %B %Y, %H:%M")

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
