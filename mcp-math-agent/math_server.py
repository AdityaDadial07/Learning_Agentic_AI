import math
from mcp.server.fastmcp import FastMCP

# ============================================================
# STEP 1: Create MCP Server
# ============================================================
mcp= FastMCP("Math Server")

# ============================================================
# STEP 2: Define MCP Tools
# ============================================================
@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> str:
    """Divide a by b. Handles division by zero."""

    if b == 0:
        return "Error: Cannot divide by zero."

    return str(a / b)


@mcp.tool()
def square_root(number: float) -> str:
    """Calculate the square root of a number."""

    if number < 0:
        return "Error: Cannot take square root of a negative number."

    return str(math.sqrt(number))

# ============================================================
# STEP 3: Run MCP Server Using STDIO
# ============================================================
if __name__ == "__main__":
    mcp.run(transport="stdio")