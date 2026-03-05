from typing import Any, Generic

from rlgym.api import AgentID, RewardType

from api.interceptor import Interceptor


class RewardPrinter(
    Generic[AgentID, RewardType], Interceptor[AgentID, Any, Any, RewardType]
):
    def reset(
        self, agents: list[AgentID], initial_state: Any, shared_info: dict[str, Any]
    ):
        pass

    def intercept(
        self,
        observations: dict[AgentID, Any],
        next_observations: dict[AgentID, Any],
        rewards: dict[AgentID, RewardType],
        terminated: dict[AgentID, bool],
        truncated: dict[AgentID, bool],
        shared_info: dict[str, Any],
    ):
        print(rewards)

    def intercept_actions(
        self,
        observations: dict[AgentID, Any],
        actions: dict[AgentID, Any],
        shared_info: dict[str, Any],
    ):
        pass
