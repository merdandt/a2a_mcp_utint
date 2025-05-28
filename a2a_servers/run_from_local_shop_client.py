# echo_client.py
import asyncio
import logging
import traceback
from uuid import uuid4

# Assuming common types and client are importable
from common.client import A2AClient
from common.client.card_resolver import A2ACardResolver
from common.types import Message, TextPart, TaskState

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERVER_URL = "http://localhost:12000/auto_repair_shop_manager"

async def main():
    # Resolve the agent card and initialize client
    resolver = A2ACardResolver(SERVER_URL)
    agent_card = resolver.get_agent_card()
    client = A2AClient(agent_card=agent_card)
    streaming_enabled = agent_card.capabilities.streaming

    # Generate a single session ID for the entire conversation
    session_id = f"session-{uuid4().hex}"
    
    print("Auto Repair Shop Conversation (type 'exit' to quit)")
    print("-------------------------------------------------")
    
    # Conversation loop: support streaming and multi-turn clarifications
    while True:
        # Initial user input (or exit)
        user_text = input("\nYour message (or 'exit' to quit): ")
        if user_text.lower() in ['exit', 'quit', 'bye']:
            print("Ending conversation. Goodbye!")
            break

        # Start a new task for this user request
        task_id = f"shop-task-{uuid4().hex}"
        # Build initial message payload
        message = Message(role="user", parts=[TextPart(text=user_text)])
        payload = {
            "id": task_id,
            "sessionId": session_id,
            "message": message.model_dump(),
        }

        need_more = True
        # Inner loop: if agent requests more info (INPUT_REQUIRED), prompt again
        while need_more:
            need_more = False
            try:
                logger.info(f"Sending task {task_id} to {SERVER_URL} (streaming={streaming_enabled})...")
                if streaming_enabled:
                    # Stream incremental updates
                    async for event in client.send_task_streaming(payload):
                        if event.error:
                            logger.error(f"Stream error: {event.error.message}")
                            break
                        result = event.result
                        # status updates
                        if hasattr(result, 'status') and result.status and result.status.message:
                            text = result.status.message.parts[0].text
                            print(f"\nAuto Repair Shop Manager: {text}")
                            # if final and requires input, loop again
                            if getattr(result, 'final', False) and result.status.state == TaskState.INPUT_REQUIRED:
                                need_more = True
                        # artifact updates can be handled here if needed
                else:
                    # Non-streaming single response
                    response = await client.send_task(payload)
                    if response.error:
                        logger.error(f"Task {task_id} failed: {response.error.message}")
                    else:
                        task = response.result
                        if task.status.message and task.status.message.parts:
                            text = task.status.message.parts[0].text
                            print(f"\nAuto Repair Shop Manager: {text}")
                        if task.status.state == TaskState.INPUT_REQUIRED:
                            need_more = True
            except Exception as e:
                logger.error(traceback.format_exc())
                logger.error(f"Communication error with agent: {e}")
                print("\nError connecting to the Auto Repair Shop Manager. Please try again.")
                break

            if need_more:
                # Prompt user to supply more details
                user_text = input("\nAdditional details (to satisfy input requirements): ")
                message = Message(role="user", parts=[TextPart(text=user_text)])
                payload['message'] = message.model_dump()
            else:
                # Task completed
                break

if __name__ == "__main__":
    asyncio.run(main())