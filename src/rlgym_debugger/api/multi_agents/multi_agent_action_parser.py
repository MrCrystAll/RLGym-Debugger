from collections.abc import Hashable
from typing import Any, Dict, List
from rlgym.api import ActionParser
from operator import itemgetter

from rlgym_debugger.api.multi_agents.multi_agent_component import MultiAgentComponent


class MultiAgentsActionParser(
    ActionParser[Hashable, Any, Any, Any, Any],
    MultiAgentComponent[ActionParser[Hashable, Any, Any, Any, Any]],
):
    def reset(
        self, agents: List[Hashable], initial_state: Any, shared_info: Dict[str, Any]
    ) -> None:
        _agents_per_agent_name = self.get_agents_from_state(agents)

        for _agent_name in self.agent_names:
            _state_agents = _agents_per_agent_name[_agent_name]
            _act_parser = self._agent_name_to_component[_agent_name]

            _act_parser.reset(_state_agents, initial_state, shared_info)

    def parse_actions(
        self, actions: Dict[Hashable, Any], state: Any, shared_info: Dict[str, Any]
    ) -> Dict[Hashable, Any]:
        _all_act = {}
        _agents_per_agent_name = self.get_agents_from_state(list(actions.keys()))

        for _agent_name in _agents_per_agent_name.keys():
            _state_agents = _agents_per_agent_name[_agent_name]
            _act_parser = self._agent_name_to_component[_agent_name]

            _agent_actions = [itemgetter(*_state_agents)(actions)]
            _agent_actions = dict(zip(_state_agents, _agent_actions))

            _agent_engine_actions = _act_parser.parse_actions(
                _agent_actions, state, shared_info
            )
            _all_act.update(_agent_engine_actions)

        return _all_act

    def get_action_space(self, agent: Hashable) -> Any:
        return self.get_agent_component(agent).get_action_space(agent)
