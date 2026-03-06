from typing import Any, Generic

from rlgym.api import AgentID, StateType, ObsType, RewardType

from rlgym_debugger.api.base.interceptor import Interceptor


class MultiInterceptor(
    Generic[AgentID, StateType, ObsType, RewardType],
    Interceptor[AgentID, StateType, ObsType, RewardType],
):
    """This class is used to aggregate multiple interceptors more easily"""

    def __init__(
        self, *interceptors: Interceptor[AgentID, StateType, ObsType, RewardType]
    ) -> None:
        self.interceptors = interceptors

    def reset(
        self,
        agents: list[AgentID],
        initial_state: StateType,
        shared_info: dict[str, Any],
    ):
        for interceptor in self.interceptors:
            interceptor.reset(agents, initial_state, shared_info)

    def intercept_actions(
        self,
        observations: dict[AgentID, ObsType],
        actions: dict[AgentID, ObsType],
        state: StateType,
        shared_info: dict[str, Any],
    ):
        for interceptor in self.interceptors:
            interceptor.intercept_actions(observations, actions, state, shared_info)

    def intercept(
        self,
        observations: dict[AgentID, ObsType],
        next_observations: dict[AgentID, Any],
        rewards: dict[AgentID, RewardType],
        terminated: dict[AgentID, bool],
        truncated: dict[AgentID, bool],
        shared_info: dict[str, Any],
    ):
        for interceptor in self.interceptors:
            interceptor.intercept(
                observations,
                next_observations,
                rewards,
                terminated,
                truncated,
                shared_info,
            )
