from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def loadConfig(path_value: str) -> dict[str, Any]:
    with open(path_value, encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def ensureParent(path_value: str) -> None:
    Path(path_value).parent.mkdir(parents=True, exist_ok=True)


def writeJson(path_value: str, value: Any) -> None:
    ensureParent(path_value)
    with open(path_value, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")


def readJsonl(path_value: str) -> list[dict[str, Any]]:
    with open(path_value, encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def writeJsonl(path_value: str, rows: list[dict[str, Any]]) -> None:
    ensureParent(path_value)
    with open(path_value, "w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
