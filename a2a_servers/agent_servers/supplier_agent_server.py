import asyncio
from dotenv import load_dotenv, find_dotenv

from a2a_servers.agent_servers.utils import generate_agent_card, generate_agent_task_manager
from a2a_servers.agents.adk_agent import ADKAgent
from a2a_servers.common.server.server import A2AServer
from a2a_servers.common.types import (
    AgentSkill,
)
from adk_agents_testing.mcp_tools.mcp_tool_supplier import return_sse_mcp_tools_supplier

load_dotenv(find_dotenv())

async def run_agent():
    AGENT_NAME = "parts_supplier_agent"
    AGENT_DESCRIPTION = "An agent that provides auto parts availability and manages parts orders."
    PORT = 10000
    HOST = "0.0.0.0"
    AGENT_URL = f"http://{HOST}:{PORT}"
    AGENT_VERSION = "1.0.0"
    MODEL = 'gemini-2.0-flash-lite'
    
    AGENT_SKILLS = [
        AgentSkill(
            id="SKILL_PARTS_AVAILABILITY",
            name="parts_availability",
            description="Checks inventory for specific auto parts and confirms compatibility.",
        ),
        AgentSkill(
            id="SKILL_PARTS_ORDERING",
            name="parts_ordering",
            description="Creates and manages parts orders with various delivery priorities.",
        ),
    ]

    AGENT_CARD = generate_agent_card(
        agent_name=AGENT_NAME,
        agent_description=AGENT_DESCRIPTION,
        agent_url=AGENT_URL,
        agent_version=AGENT_VERSION,
        can_stream=False,
        can_push_notifications=False,
        can_state_transition_history=True,
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=AGENT_SKILLS,
    )

    parts_tools, parts_exit_stack = await return_sse_mcp_tools_supplier()

    parts_supplier_agent = ADKAgent(
        model=MODEL,
        name="parts_supplier_agent",
        description="Manages auto parts inventory and parts ordering for an auto repair shop.",
        tools=parts_tools,
        instructions=(
            "You are an expert auto parts specialist. You can check parts availability, confirm "
            "vehicle compatibility, and create parts orders with various delivery priorities."
            "\n"
            "When handling parts requests:\n"
            "1. Check part availability with check_part_availability\n"
            "2. Confirm vehicle compatibility (make, model, year)\n"
            "3. Create parts orders with create_parts_order when needed\n"
            "4. Provide delivery estimates based on priority\n"
            "\n"
            "Always explain parts availability and any potential delays clearly. For out-of-stock items, "
            "provide restock dates when available and suggest alternatives when possible."
        ),
    )
    
    task_manager = generate_agent_task_manager(
        agent=parts_supplier_agent,
    )
    
    server = A2AServer(
        host=HOST,
        port=PORT,
        endpoint="/parts_supplier_agent",
        agent_card=AGENT_CARD,
        task_manager=task_manager
    )
    print(f"Starting {AGENT_NAME} A2A Server on {AGENT_URL}")
    await server.astart()


if __name__ == "__main__":
    asyncio.run(
        run_agent()
    )
