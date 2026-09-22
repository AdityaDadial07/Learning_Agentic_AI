import os
import sys
import asyncio

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

from langchain_mcp_adapters.client import MultiServerMCPClient

# ============================================================
# STEP 1: Load Environment Variables
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found. "
        "Please add it to your .env file."
    )

print("GOOGLE_API_KEY found:", bool(GOOGLE_API_KEY))

# ============================================================
# STEP 2: Initialize the LLM
# ============================================================

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GOOGLE_API_KEY,
)

# ============================================================
# STEP 3: Configure MCP Server
# ============================================================

client = MultiServerMCPClient(
    {
        "math": {
            "command": sys.executable,
            "args": ["math_server.py"],
            "transport": "stdio",
        }
    }
)

# ============================================================
# STEP 4: Load Tools From MCP Server
# ============================================================

async def create_math_agent():

    tools = await client.get_tools()

    print("\n=== Available MCP Tools ===")

    for t in tools:
        print(f" • {t.name}: {t.description}")

    print()
    
    # ========================================================
    # STEP 5: Create LangChain Agent
    # ========================================================

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=(
            "You are a helpful math assistant. "
            "Use the available tools to perform calculations. "
            "For multi-step questions, call tools in the "
            "correct order. "
            "Handle errors gracefully."
        ),
    )

    return agent

# ============================================================
# STEP 6: Run the Agent
# ============================================================

async def run_agent(question: str):

    agent = await create_math_agent()

    print(f"\nUser: {question}")
    print("-" * 60)

    result = await agent.ainvoke(
        {
            "messages": [
                ("user", question)
            ]
        }
    )

    final_message = result["messages"][-1]

    print(f"Agent: {final_message.content}")
    print("-" * 60)

# ============================================================
# STEP 7: Main
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        run_agent("What is 42 * 58?")
    )