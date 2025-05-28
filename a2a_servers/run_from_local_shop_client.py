# echo_client.py
import asyncio
import logging
import traceback
from uuid import uuid4

# Assuming common types and client are importable
from common.client import A2AClient
from common.types import Message, TextPart

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERVER_URL = "http://localhost:12000/auto_repair_shop_manager"

async def main():
    client = A2AClient(url=SERVER_URL)
    
    # Generate a single session ID for the entire conversation
    session_id = f"session-{uuid4().hex}"
    
    print("Auto Repair Shop Conversation (type 'exit' to quit)")
    print("-------------------------------------------------")
    
    # Conversation loop
    conversation_active = True
    while conversation_active:
        # Get user input
        user_text = input("\nYour message: ")
        
        # Check for exit command
        if user_text.lower() in ['exit', 'quit', 'bye']:
            print("Ending conversation. Goodbye!")
            conversation_active = False
            break
        
        # Generate a new task ID for each message
        task_id = f"echo-task-{uuid4().hex}"
        
        user_message = Message(
            role="user",
            parts=[TextPart(text=user_text)]
        )
        
        send_params = {
            "id": task_id,
            "sessionId": session_id,  # Keep the same session ID for the entire conversation
            "message": user_message.model_dump(),
        }
        
        try:
            logger.info(f"Sending task {task_id} to {SERVER_URL}...")
            response = await client.send_task(payload=send_params)
            
            if response.error:
                logger.error(f"Task {task_id} failed: {response.error.message} (Code: {response.error.code})")
            elif response.result:
                task_result = response.result
                logger.info(f"Task {task_id} completed with state: {task_result.status.state}")
                if task_result.status.message and task_result.status.message.parts:
                    agent_part = task_result.status.message.parts[0]
                    print("\nAuto Repair Shop Manager:")
                    print(agent_part.text)
                else:
                    logger.warning("No message part in agent response status")
            else:
                logger.error(f"Received unexpected response for task {task_id}: {response}")
        
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(f"An error occurred while communicating with the agent: {e}")
            print("\nError connecting to the Auto Repair Shop Manager. Please try again.")

if __name__ == "__main__":
    asyncio.run(main())