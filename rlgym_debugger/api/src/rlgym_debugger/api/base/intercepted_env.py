from typing import Dict, Generic, Tuple

from rlgym.api import (
    RLGym,
    AgentID,
    ObsType,
    ActionType,
    EngineActionType,
    RewardType,
    StateType,
    ObsSpaceType,
    ActionSpaceType,
)
from rlgym.api.config import (
    ActionParser,
    DoneCondition,
    ObsBuilder,
    Renderer,
    RewardFunction,
    SharedInfoProvider,
    StateMutator,
    TransitionEngine,
)

from rlgym_debugger.api.base.interceptor import Interceptor


class InterceptedRLGym(
    Generic[
        AgentID,
        ObsType,
        ActionType,
        EngineActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
    ],
    RLGym[
        AgentID,
        ObsType,
        ActionType,
        EngineActionType,
        RewardType,
        StateType,
        ObsSpaceType,
        ActionSpaceType,
    ],
):
    def __init__(
        self,
        state_mutator: StateMutator[StateType],
        obs_builder: ObsBuilder[AgentID, ObsType, StateType, ObsSpaceType],
        action_parser: ActionParser[
            AgentID, ActionType, EngineActionType, StateType, ActionSpaceType
        ],
        reward_fn: RewardFunction[AgentID, StateType, RewardType],
        transition_engine: TransitionEngine[AgentID, StateType, EngineActionType],
        interceptor: Interceptor[AgentID, StateType, ObsType, ActionType, RewardType],
        termination_cond: DoneCondition[AgentID, StateType] | None = None,
        truncation_cond: DoneCondition[AgentID, StateType] | None = None,
        shared_info_provider: SharedInfoProvider[AgentID, StateType] | None = None,
        renderer: Renderer[StateType] | None = None,
    ):
        super().__init__(
            state_mutator,
            obs_builder,
            action_parser,
            reward_fn,
            transition_engine,
            termination_cond,
            truncation_cond,
            shared_info_provider,
            renderer,
        )
        self.interceptor = interceptor

        self._current_observations = None

    def reset(self) -> Dict[AgentID, ObsType]:
        self._current_observations = super().reset()
        return self._current_observations

    def step(
        self, actions: Dict[AgentID, ActionType]
    ) -> Tuple[
        Dict[AgentID, ObsType],
        Dict[AgentID, RewardType],
        Dict[AgentID, bool],
        Dict[AgentID, bool],
    ]:
        assert self._current_observations is not None, (
            "Environment has not been reset or was given invalid observations"
        )

        self.interceptor.intercept_actions(
            self._current_observations, actions, self.state, self.shared_info
        )

        data = super().step(actions)

        self.interceptor.intercept(self._current_observations, *data, self.shared_info)

        self._current_observations = data[0]

        return data
