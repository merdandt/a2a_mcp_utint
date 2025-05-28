import asyncio
from dotenv import load_dotenv, find_dotenv

from a2a_servers.agent_servers.utils import generate_agent_card, generate_agent_task_manager
from a2a_servers.agents.adk_agent import ADKAgent
from a2a_servers.common.server.server import A2AServer
from a2a_servers.common.types import (
    AgentSkill,
)
from adk_agents_testing.mcp_tools.mcp_tool_stocks import return_sse_mcp_tools_stocks

load_dotenv(find_dotenv())

async def run_agent():
    AGENT_NAME = "auto_repair_shop_manager"
    AGENT_DESCRIPTION = "A shop manager that coordinates vehicle diagnostics, repair procedures, and parts ordering for auto repairs."
    PORT = 12000
    HOST = "0.0.0.0"
    AGENT_URL = f"http://{HOST}:{PORT}"
    AGENT_VERSION = "1.0.0"
    MODEL = 'gemini-2.5-pro-preview-05-06'
    AGENT_SKILLS = [
        AgentSkill(
            id="MANAGE_VEHICLE_REPAIRS",
            name="manage_vehicle_repairs",
            description="Coordinates vehicle diagnostics, repair planning, and parts ordering.",
        ),
    ]

    list_urls = [
        "http://localhost:11000/mechanic_agent",
        "http://localhost:10000/parts_supplier_agent",
    ]

    AGENT_CARD = generate_agent_card(
        agent_name=AGENT_NAME,
        agent_description=AGENT_DESCRIPTION,
        agent_url=AGENT_URL,
        agent_version=AGENT_VERSION,
        # enable streaming for multi-turn and incremental responses
        can_stream=True,
        can_push_notifications=False,
        can_state_transition_history=True,
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=AGENT_SKILLS,
    )

    host_agent = ADKAgent(
        model=MODEL,
        name="auto_repair_shop_manager",
        description="Auto Repair Shop Manager that coordinates vehicle diagnostics, parts ordering, and repair processes.",
        tools=[],
        instructions=(
            "You are an Auto Repair Shop Manager responsible for coordinating the entire repair process. "
            "You work with two specialized teams:\n\n"
            
            "1. Mechanic Team (mechanic_agent):\n"
            "   - Specializes in vehicle diagnostics using OBD-II scanners\n"
            "   - Provides detailed repair procedures\n"
            "   - Generates work orders\n"
            "   - Documents customer communications\n\n"
            
            "2. Parts Department (parts_supplier_agent):\n"
            "   - Checks parts availability and compatibility\n"
            "   - Creates parts orders with delivery tracking\n"
            "   - Manages inventory for various vehicle repairs\n\n"
            
            "Your responsibilities as Shop Manager:\n"
            "- Determine whether a request requires vehicle diagnostics, parts ordering, or both\n"
            "- Route diagnostic tasks to the Mechanic Team\n"
            "- Route parts availability and ordering tasks to the Parts Department\n"
            "- Coordinate between teams when repairs require both diagnostics and parts\n"
            "- Provide clear explanations to customers about repair needs, costs, and timelines\n"
            "- Present a unified workflow from diagnosis to repair completion\n\n"
        ),
        is_host_agent=True,
        remote_agent_addresses=list_urls,
    )

    task_manager = generate_agent_task_manager(
        agent=host_agent,
    )
    server = A2AServer(
        host=HOST,
        port=PORT,
        endpoint="/auto_repair_shop_manager",
        agent_card=AGENT_CARD,
        task_manager=task_manager
    )
    print(f"Starting {AGENT_NAME} A2A Server on {AGENT_URL}")
    await server.astart()


if __name__ == "__main__":
    asyncio.run(
        run_agent()
    )
