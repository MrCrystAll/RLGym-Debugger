from collections.abc import Hashable
from typing import Any, Dict, List
from rlgym.api import ObsBuilder

from rlgym_debugger.api.multi_agents.multi_agent_component import MultiAgentComponent


class MultiAgentsObsBuilder(
    ObsBuilder[Hashable, Any, Any, Any],
    MultiAgentComponent[ObsBuilder[Hashable, Any, Any, Any]],
):
    def reset(
        self, agents: List[Hashable], initial_state: Any, shared_info: Dict[str, Any]
    ) -> None:
        _state_agents_per_agent = self.get_agents_from_state(agents)

        for _agent_name in self.agent_names:
            _state_agents = _state_agents_per_agent[_agent_name]
            _obs_builder = self._agent_name_to_component[_agent_name]

            _obs_builder.reset(_state_agents, initial_state, shared_info)

    def build_obs(
        self, agents: List[Hashable], state: Any, shared_info: Dict[str, Any]
    ) -> Dict[Hashable, Any]:
        _all_obs = {}
        _agents_per_agent_name = self.get_agents_from_state(agents)

        for _agent_name in self.agent_names:
            _state_agents = _agents_per_agent_name[_agent_name]
            _obs_builder = self._agent_name_to_component[_agent_name]

            _inside_obs = _obs_builder.build_obs(_state_agents, state, shared_info)
            _all_obs.update(_inside_obs)

        return _all_obs

    def get_obs_space(self, agent: Hashable) -> Any:
        return self.get_agent_component(agent).get_obs_space(agent)
