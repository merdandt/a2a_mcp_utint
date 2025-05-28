from google.adk.tools.mcp_tool import MCPToolset
from google.adk.tools.mcp_tool.mcp_toolset import SseServerParams
from mcp import StdioServerParameters


async def return_mcp_tools_mechanic():
    print("Attempting to connect to MCP server for Mechanic...")
    tools, exit_stack = await MCPToolset.from_server(
        connection_params=StdioServerParameters(
            command="/opt/homebrew/bin/uv",
            args=[
                "--directory",
                "/Users/xskills/Development/Python/A2A_MCP/a2a_mcp_utint/mcp_server",
                "run",
                "mechanic_server.py"
            ],
            env={
                "MCP_PORT":"8000",
                "PYTHONPATH": "/Users/xskills/Development/Python/A2A_MCP/a2a_mcp_utint:${PYTHONPATH}"
            },
        )
    )
    print("MCP Toolset created successfully.")
    return tools, exit_stack


async def return_sse_mcp_tools_mechanic():
    print("Attempting to connect to MCP server for Mechanic...")
    server_params = SseServerParams(
        url="http://localhost:8080/sse",
    )
    tools, exit_stack = await MCPToolset.from_server(connection_params=server_params)
    print("MCP Toolset created successfully.")
    return tools, exit_stack
