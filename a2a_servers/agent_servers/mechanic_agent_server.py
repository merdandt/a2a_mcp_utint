import asyncio
from typing import List, Any

from dotenv import load_dotenv, find_dotenv
from google.adk import Agent
from google.adk.tools import google_search

from a2a_servers.agent_servers.utils import generate_agent_task_manager, generate_agent_card
from a2a_servers.agents.adk_agent import ADKAgent
from a2a_servers.common.agent_task_manager import AgentTaskManager
from a2a_servers.common.server.server import A2AServer
from a2a_servers.common.types import (
    AgentCard,
    AgentCapabilities,
    AgentSkill,
)
from adk_agents_testing.mcp_tools.mcp_tool_mechanic import return_sse_mcp_tools_mechanic

load_dotenv(find_dotenv())

async def run_agent():
    AGENT_NAME = "mechanic_agent"
    AGENT_DESCRIPTION = "An expert auto mechanic agent that can diagnose vehicle issues, provide repair procedures, and generate work orders."
    HOST = "0.0.0.0"
    PORT = 11000
    AGENT_URL = f"http://{HOST}:{PORT}"
    AGENT_VERSION = "1.0.0"
    MODEL = 'gemini-2.0-flash-lite'
    
    AGENT_SKILLS = [
        AgentSkill(
            id="VEHICLE_DIAGNOSTICS",
            name="vehicle_diagnostics",
            description="Can scan vehicles for error codes and diagnose issues",
        ),
        AgentSkill(
            id="REPAIR_PROCEDURES",
            name="repair_procedures",
            description="Can provide detailed repair procedures for vehicle issues",
        ),
        AgentSkill(
            id="WORK_ORDERS",
            name="work_orders",
            description="Can generate comprehensive work orders for vehicle repairs",
        )
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

    mechanic_tools, mechanic_exit_stack = await return_sse_mcp_tools_mechanic()

    mechanic_agent = ADKAgent(
        model=MODEL,
        name="mechanic_agent",
        description="An expert auto mechanic that can diagnose vehicle issues, provide repair procedures, and generate work orders.",
        tools=mechanic_tools,
        instructions=(
            "You are an expert auto mechanic. You can diagnose vehicle issues using OBD-II scanners, "
            "provide detailed repair procedures, generate work orders, and log customer communications. "
            "\n"
            "When diagnosing vehicles:\n"
            "1. Start by scanning the vehicle for error codes using scan_vehicle_for_error_codes\n"
            "2. Get detailed repair procedures for each error code with get_repair_procedure\n"
            "3. Generate a comprehensive work order with generate_work_order\n"
            "4. Log customer communications with customer_communication_log\n"
            "\n"
            "Always explain diagnostic findings in clear, customer-friendly language while maintaining technical accuracy."
        ),
    )
    
    task_manager = generate_agent_task_manager(
        agent=mechanic_agent,
    )
    
    server = A2AServer(
        host=HOST,
        port=PORT,
        endpoint="/mechanic_agent",
        agent_card=AGENT_CARD,
        task_manager=task_manager
    )
    print(f"Starting {AGENT_NAME} A2A Server on {AGENT_URL}")
    await server.astart()


if __name__ == "__main__":
    asyncio.run(
        run_agent()
    )
