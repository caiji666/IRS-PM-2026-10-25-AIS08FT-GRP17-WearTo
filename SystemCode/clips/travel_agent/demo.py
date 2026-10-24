"""Offline understanding demo; other contributors' tools remain explicit scaffolds."""

import json
from pathlib import Path

from travel_agent.agent import TOOLS
from travel_agent.understanding.service import run


def main():
    example = Path(__file__).resolve().parents[1] / "examples/understanding/seoul_winter.json"
    print("OFFLINE DEMO: synthetic weather and explicitly enabled draft KG; no live LLM")
    result = run(json.loads(example.read_text()))
    print(result.model_dump_json(indent=2))
    for tool in TOOLS:
        if tool.name != "understand_and_reason":
            print(tool.name, tool.invoke({"payload": {}}))


if __name__ == "__main__":
    main()
