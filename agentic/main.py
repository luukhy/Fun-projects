import os

from agno.agent import Agent
from agno.models.groq import Groq
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    print("Warning: GROQ_API_KEY not found in .env file!")

test_agent = Agent(
    model=Groq(id="llama-3.1-8b-instant"),
    description="You are a helpful test assistant running securely.",
)

test_agent.print_response(
    "Hello! Are you working securely from my .env file?", stream=True
)
