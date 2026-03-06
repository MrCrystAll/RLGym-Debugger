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
    """The main RLGym class. This class is responsible
        for managing the environment and the interactions between
        the different components of the environment.
        It is the main interface for the user to interact with an environment.
    """
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
        """
        :param state_mutator: The StateMutator used to modify the state of the environment.
        :type state_mutator: StateMutator[StateType]
        :param obs_builder: The ObsBuilder used to build observations for the agents.
        :type obs_builder: ObsBuilder[AgentID, ObsType, StateType, ObsSpaceType]
        :param action_parser: The ActionParser used to parse
            actions from the agents into engine actions.
        :type action_parser: ActionParser[ AgentID, ActionType,
            EngineActionType, StateType, ActionSpaceType ]
        :param reward_fn: The RewardFunction used to calculate rewards for the agents.
        :type reward_fn: RewardFunction[AgentID, StateType, RewardType]
        :param transition_engine: The TransitionEngine used to
            transition the environment from one state to another.
        :type transition_engine: TransitionEngine[AgentID, StateType, EngineActionType]
        :param interceptor: The Interceptor used to
            intercept and modify the actions and environment data
        :type interceptor: Interceptor[AgentID, StateType, ObsType, ActionType, RewardType]
        :param termination_cond: The DoneCondition used to determine
            if the episode is done, defaults to None
        :type termination_cond: DoneCondition[AgentID, StateType] | None, optional
        :param truncation_cond: The DoneCondition used to determine
            if the episode is truncated, defaults to None
        :type truncation_cond: DoneCondition[AgentID, StateType] | None, optional
        :param shared_info_provider: The SharedInfoProvider used to
            provide shared information across all config objects, defaults to None
        :type shared_info_provider: SharedInfoProvider[AgentID, StateType] | None, optional
        :param renderer: The Renderer used to render the environment, defaults to None
        :type renderer: Renderer[StateType] | None, optional
        """
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
