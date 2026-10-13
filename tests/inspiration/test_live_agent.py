import os
from dotenv import load_dotenv

from travel_agent.agent import build_agent

load_dotenv()

# Create the agent with a configured LLM
agent = build_agent(os.environ["AGENT_MODEL"])

# Send a user request to the agent
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "I'm travelling to Seoul in December. "
                    "I prefer minimalist outfits for sightseeing and dinner."
                )
            }
        ]
    },
    config={"recursion_limit": 12}
)

print(result["messages"][-1].content)