from abc import ABC
from collections.abc import Hashable
from typing import Generic, TypeVar

ComponentType = TypeVar("ComponentType")

UNNAMED_AGENT = "_unnamed"


class MultiAgentComponent(ABC, Generic[ComponentType]):
    """A class to host multiple component per env (one per agent)"""

    def __init__(
        self,
        dict_agent_to_component: dict[str, ComponentType],
        default_component: ComponentType,
    ) -> None:
        self._agent_name_to_component = dict_agent_to_component
        self._agent_name_to_component[UNNAMED_AGENT] = default_component

    @property
    def agent_names(self) -> list[str]:
        """All the names given by the user

        Returns:
            list[str]: All the names given by the user
        """
        return list(self._agent_name_to_component.keys())

    @property
    def components(self) -> list[ComponentType]:
        """All the components given by the user

        Returns:
            list[ComponentType]: All the components given by the user
        """
        return list(self._agent_name_to_component.values())

    def get_agents_from_state(
        self, agents: list[Hashable]
    ) -> dict[str, list[Hashable]]:
        """Gets and sort per agent all the agents in the state that matches the name of one agent

        Args:
            agents (list[Hashable]): Agents in the state

        Returns:
            dict[str, list[Hashable]]: All state agents sorted per name
        """
        _state_agent_per_agent: dict[str, list[Hashable]] = {
            agent_name: [] for agent_name in self.agent_names
        }

        _agents_copy = iter(agents.copy())

        for _agent_name in self.agent_names:
            while True:
                try:
                    _state_agent = next(_agents_copy)
                    if _agent_name in str(_state_agent):
                        _state_agent_per_agent[_agent_name].append(_state_agent)
                    else:
                        _state_agent_per_agent[UNNAMED_AGENT].append(_state_agent)
                except StopIteration:
                    break
        return _state_agent_per_agent

    def get_agent_component(self, agent: Hashable) -> ComponentType:
        """Gets the given state agent's component

        Args:
            agent (Hashable): The state agent ID

        Raises:
            ValueError: If no values are found within the component dictionnary, this error is thrown

        Returns:
            ComponentType: The found component
        """
        for _agent_name in self.agent_names:
            if _agent_name in str(agent):
                return self._agent_name_to_component[_agent_name]

        return self._agent_name_to_component[UNNAMED_AGENT]
