import json

from runtime.agent import Agent
from runtime.events import EventRecorder


class AgentRoutingError(Exception):
    """Raised when agent routing fails."""


class RoutingResult:
    """Result of an agent routing decision."""

    def __init__(self, agent: Agent, reason: str):
        self.agent = agent
        self.reason = reason

    def __repr__(self):
        return (
            f"RoutingResult(agent={self.agent.name!r}, "
            f"reason={self.reason!r})"
        )


class AgentRouter:
    """Routes user tasks to the most appropriate registered agent.

    Uses an LLM to classify the task and select one agent from
    the registered collection.
    """

    def __init__(
        self,
        agents: list[Agent],
        *,
        recorder: EventRecorder | None = None,
    ):
        if not agents:
            raise AgentRoutingError("Router requires at least one registered agent.")

        self.agents = {agent.name: agent for agent in agents}
        self.recorder = recorder or EventRecorder()

    def route(self, task: str) -> RoutingResult:
        """Route a task to the most appropriate agent.

        Args:
            task: The user's task description.

        Returns:
            RoutingResult with the selected agent and reason.

        Raises:
            AgentRoutingError: If routing fails.
        """
        self.recorder.record(
            "AgentRoutingStarted",
            task=task,
        )

        agent_descriptions = "\n".join(
            f'- {name}: {agent.description or "No description provided."}'
            for name, agent in self.agents.items()
        )

        agent_names = list(self.agents.keys())

        prompt = (
            "You are an agent router. Your ONLY job is to select "
            "the single best agent for the given task.\n\n"
            "Available agents:\n"
            f"{agent_descriptions}\n\n"
            "Respond with EXACTLY this JSON (no other text):\n"
            '{"selected_agent": "<agent_name>", '
            '"reason": "<one sentence why>"}\n\n'
            f"Valid agent names: {agent_names}\n\n"
            f"Task: {task}"
        )

        try:
            llm = list(self.agents.values())[0].llm
            response = llm.generate(
                [{"role": "user", "content": prompt}],
                tools=None,
            )
        except Exception as error:
            raise AgentRoutingError(
                f"LLM routing call failed: {error}"
            ) from error

        result = self._parse_response(response)

        self.recorder.record(
            "AgentRoutingCompleted",
            task=task,
            selected_agent=result.agent.name,
            reason=result.reason,
        )

        return result

    def _parse_response(self, response: dict) -> RoutingResult:
        """Parse the LLM response into a RoutingResult.

        Raises AgentRoutingError if the response cannot be parsed
        or the selected agent does not exist.
        """
        try:
            message = response["message"]
            content = message.get("content", "")

            start = content.index("{")
            end = content.rindex("}") + 1
            data = json.loads(content[start:end])

            selected = data["selected_agent"]
            reason = data.get("reason", "")
        except (KeyError, json.JSONDecodeError, ValueError) as error:
            raise AgentRoutingError(
                f"Failed to parse routing response: {error}\n"
                f"Raw response: {content}"
            ) from error

        if selected not in self.agents:
            raise AgentRoutingError(
                f"LLM selected unknown agent '{selected}'. "
                f"Valid agents: {list(self.agents.keys())}"
            )

        return RoutingResult(
            agent=self.agents[selected],
            reason=reason,
        )
