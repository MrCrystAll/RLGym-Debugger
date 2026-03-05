from collections.abc import Hashable
from typing import Any, Callable

from rlgym_debugger.api.human_player import DeviceInterface
from rlgym_debugger.api.base import Interceptor


class HumanPlayer(Interceptor[Hashable, Any, Any, Any]):
    """A class to allow the developer to test their own environment"""

    def __init__(
        self, player: DeviceInterface, agent_fn: Callable[[Any], Hashable]
    ) -> None:
        self._agent_fn = agent_fn

        self.player = player

    def _get_human_output(self):
        return self.player.get_output()

    def intercept_actions(
        self,
        observations: dict[Hashable, Any],
        actions: dict[Hashable, Any],
        state: Any,
        shared_info: dict[str, Any],
    ):
        agent = self._agent_fn(state)

        actions[agent] = self._get_human_output()

    def intercept(
        self,
        observations: dict[Hashable, Any],
        next_observations: dict[Hashable, Any],
        rewards: dict[Hashable, Any],
        terminated: dict[Hashable, bool],
        truncated: dict[Hashable, bool],
        shared_info: dict[str, Any],
    ):
        pass

    def reset(
        self, agents: list[Hashable], initial_state: Any, shared_info: dict[str, Any]
    ):
        pass
