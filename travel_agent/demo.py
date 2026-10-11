"""Offline contract smoke demo. No LLM and no real recommendations."""

from travel_agent.agent import TOOLS
from travel_agent.schemas import FashionContext, TripRequest


def main():
    context = FashionContext(
        request=TripRequest(destination="Demo city", activity="sightseeing", budget_minor=20000)
    )
    print("FIXTURE DEMO ONLY — not an autonomous agent or real recommendation")
    print(context.model_dump_json(indent=2))
    for tool in TOOLS:
        print(tool.name, tool.invoke({"payload": context.model_dump()}))


if __name__ == "__main__":
    main()
