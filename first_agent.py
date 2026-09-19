# My first simple agent 
import os
import math
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_agent

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
model = ChatGoogleGenerativeAI(model="gemini-3.6-flash",google_api_key=GOOGLE_API_KEY)

# ============================================================
# STEP 3: Define Tools
# ============================================================
@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def divide(a: float, b: float) -> str:
    """Divide a by b. Handles division by zero."""
    if b == 0:
        return "Error: Cannot divide by zero."

    return str(a / b)


@tool
def square_root(number: float) -> str:
    """
    Calculate the square root of a number.
    Handles negative inputs.
    """
    if number < 0:
        return (
            "Error: Cannot take square root "
            "of a negative number."
        )

    return str(math.sqrt(number))

# ============================================================
# STEP 4: Combine All Tools
# ============================================================
tools = [add, multiply, divide, square_root]


# ============================================================
# STEP 5: Display Available Tools
# ============================================================
print("\n=== Available Tools ===")
for t in tools:
    print(f" • {t.name}: {t.description}")

print()


# ============================================================
# STEP 6: Create the Agent
# ============================================================
agent = create_agent(
    model= model,
    tools=tools,
    system_prompt=(
        "You are a helpful math assistant. "
        "Use the available tools to perform calculations. "
        "For multi-step questions, call tools in the "
        "correct order. "
        "Handle errors gracefully."
    ),
)

# ============================================================
# STEP 7: Run the Agent
# ============================================================

def run_agent(question: str):
    """
    Run the agent and print the final answer.
    """

    print(f"\nUser: {question}")
    print("-" * 60)

    result = agent.invoke({
        "messages": [
            ("user", question)
        ]
    })

    final_message = result["messages"][-1]

    print(f"Agent: {final_message.content}")
    print("-" * 60)

# ============================================================
# STEP 8: Test the Agent
# ============================================================

if __name__ == "__main__":
    run_agent("What is 42 + 58?")