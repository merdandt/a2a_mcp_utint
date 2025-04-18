# echo_client.py
import asyncio
import logging
import traceback
from uuid import uuid4

# Assuming common types and client are importable
from common.client import A2AClient, card_resolver # card_resolver might be needed
from common.types import Message, TextPart, AgentCard # Import AgentCard if needed directly

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERVER_URL = "http://localhost:12000/host_agent" # URL from Echo Server's AgentCard

async def main():
    # In a real scenario, you might fetch the AgentCard first
    # try:
    #   agent_card = await card_resolver.fetch_agent_card(ECHO_SERVER_URL)
    #   client = A2AClient(agent_card=agent_card)
    # except Exception as e:
    #   logger.error(f"Failed to fetch AgentCard or initialize client: {e}")
    #   return

    # For simplicity, we'll use the URL directly
    client = A2AClient(url=SERVER_URL)

    task_id = f"echo-task-{uuid4().hex}"
    session_id = f"session-{uuid4().hex}"
    user_text = input("Enter your query: ")  # Example user input

    # Construct the user message
    user_message = Message(
        role="user",
        parts=[TextPart(text=user_text)]
    )

    # Prepare the parameters for tasks/send
    send_params = {
        "id": task_id,
        "sessionId": session_id,
        "message": user_message,
        # Optional: acceptedOutputModes, pushNotification, historyLength, metadata
    }

    try:
        logger.info(f"Sending task {task_id} to {SERVER_URL}...")
        # Use the client's send_task method
        # It handles constructing the JSONRPCRequest internally
        response = await client.send_task(payload=send_params)

        if response.error:
            logger.error(f"Task {task_id} failed: {response.error.message} (Code: {response.error.code})")
        elif response.result:
            task_result = response.result
            logger.info(f"Task {task_id} completed with state: {task_result.status.state}")
            if task_result.status.message and task_result.status.message.parts:
                 agent_part = task_result.status.message.parts[0]
                 if isinstance(agent_part, TextPart):
                     logger.info(f"Agent response: {agent_part.text}")
                 else:
                     logger.warning("Agent response was not TextPart")
            else:
                 logger.warning("No message part in agent response status")
        else:
            logger.error(f"Received unexpected response for task {task_id}: {response}")

    except Exception as e:
        logger.error(traceback.format_exc())
        logger.error(f"An error occurred while communicating with the agent: {e}")

if __name__ == "__main__":
    asyncio.run(main())
