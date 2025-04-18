from typing import Any, AsyncIterable, Dict, List

from dotenv import load_dotenv, find_dotenv
from google.adk.agents.llm_agent import LlmAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv(find_dotenv())

class ADKAgent:
    """An agent that handles stock report requests."""

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(
            self,
            model: str,
            name: str,
            description: str,
            instructions: str,
            tools: List[Any],
    ):
        """
        Initializes the ADK agent with the given parameters.
        :param model: The model to use for the agent.
        :param name: The name of the agent.
        :param description: The description of the agent.
        :param instructions: The instructions for the agent.
        :param tools: The tools the agent can use.
        """

        self._agent = self._build_agent(
            model=model,
            name=name,
            description=description,
            instructions=instructions,
            tools=tools,
        )
        self._user_id = "remote_agent"
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    async def invoke(self, query, session_id) -> str:
        """
        Invokes the agent with the given query and session ID.
        :param query: The query to send to the agent.
        :param session_id: The session ID to use for the agent.
        :return:  The response from the agent.
        """
        session = self._runner.session_service.get_session(
            app_name=self._agent.name, user_id=self._user_id, session_id=session_id
        )
        content = types.Content(
            role="user", parts=[types.Part.from_text(text=query)]
        )
        if session is None:
            session = self._runner.session_service.create_session(
                app_name=self._agent.name,
                user_id=self._user_id,
                state={},
                session_id=session_id,
            )
        events_async = self._runner.run_async(
            session_id=session.id, user_id=session.user_id, new_message=content
        )

        events = []

        async for event in events_async:
            print(event)
            events.append(event)

        if not events or not events[-1].content or not events[-1].content.parts:
            return ""
        return "\n".join([p.text for p in events[-1].content.parts if p.text])

    async def stream(self, query, session_id) -> AsyncIterable[Dict[str, Any]]:
        """
        Streams the response from the agent for the given query and session ID.
        :param query: The query to send to the agent.
        :param session_id: The session ID to use for the agent.
        :return:  An async iterable of the response from the agent.
        """
        session = self._runner.session_service.get_session(
            app_name=self._agent.name, user_id=self._user_id, session_id=session_id
        )
        content = types.Content(
            role="user", parts=[types.Part.from_text(text=query)]
        )
        if session is None:
            session = self._runner.session_service.create_session(
                app_name=self._agent.name,
                user_id=self._user_id,
                state={},
                session_id=session_id,
            )
        async for event in self._runner.run_async(
                user_id=self._user_id, session_id=session.id, new_message=content
        ):
            if event.is_final_response():
                response = ""
                if (
                        event.content
                        and event.content.parts
                        and event.content.parts[0].text
                ):
                    response = "\n".join([p.text for p in event.content.parts if p.text])
                elif (
                        event.content
                        and event.content.parts
                        and any([True for p in event.content.parts if p.function_response])):
                    response = next((p.function_response.model_dump() for p in event.content.parts))
                yield {
                    "is_task_complete": True,
                    "content": response,
                }
            else:
                yield {
                    "is_task_complete": False,
                    "updates": "Processing the request...",
                }

    @staticmethod
    def _build_agent(
            model: str,
            name: str,
            description: str,
            instructions: str,
            tools: List[Any],
    ) -> LlmAgent:
        """
        Builds the LLM agent for the reimbursement agent.

        :param model: The model to use for the agent.
        :param name: The name of the agent.
        :param description: The description of the agent.
        :param instructions: The instructions for the agent.
        :param tools: The tools the agent can use.
        :return: The LLM agent.
        """
        return LlmAgent(
            model=model,
            name=name,
            description=description,
            instruction=instructions,
            tools=tools,
        )
