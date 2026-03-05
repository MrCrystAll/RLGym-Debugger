from abc import abstractmethod
from typing import Any, Generic

from rlgym.api import AgentID, StateType, ObsType, RewardType


class Interceptor(Generic[AgentID, StateType, ObsType, RewardType]):
    """The base class to intercept and act on the environment"""

    @abstractmethod
    def reset(
        self,
        agents: list[AgentID],
        initial_state: StateType,
        shared_info: dict[str, Any],
    ):
        """Function to be called each time the environment is reset. Note that this does not need to return anything,
        the environment will call `build_obs` automatically after reset, so the initial observation for a policy will be
        constructed in the same way as every other observation.

        :param agents: List of AgentIDs for which this ObsBuilder will return an Obs
        :type agents: list[AgentID]
        :param initial_state: The initial game state of the reset environment.
        :type initial_state: StateType
        :param shared_info: A dictionary with shared information across all config objects.
        :type shared_info: dict[str, Any]
        """

    @abstractmethod
    def intercept(
        self,
        observations: dict[AgentID, ObsType],
        next_observations: dict[AgentID, Any],
        rewards: dict[AgentID, RewardType],
        terminated: dict[AgentID, bool],
        truncated: dict[AgentID, bool],
        shared_info: dict[str, Any],
    ):
        """The function intercepts and allows the modification of the given arguments

        :param observations: The observations returned by the environment **before** stepping
        :type observations: dict[AgentID, ObsType]
        :param next_observations: The observations returned by the environment **after** stepping
        :type next_observations: dict[AgentID, ObsType]
        :param rewards: The rewards returned by the environment
        :type rewards: dict[AgentID, RewardType]
        :param terminated: The termination signals returned by the environment
        :type terminated: dict[AgentID, bool]
        :param truncated: The truncation signals returned by the environment
        :type truncated: dict[AgentID, bool]
        :param shared_info: The shared information of the environment
        :type shared_info: dict[str, Any]
        """

    @abstractmethod
    def intercept_actions(
        self,
        observations: dict[AgentID, ObsType],
        actions: dict[AgentID, ObsType],
        state: StateType,
        shared_info: dict[str, Any],
    ):
        """The function intercepts the actions and the observations used to build the actions before being sent to the env.step method

        :param observations: The observations used to build the action
        :type observations: dict[AgentID, ObsType]
        :param actions: The actions themselves
        :type actions: dict[AgentID, ObsType]
        :param state: The game state of the environment.
        :type state: StateType
        :param shared_info: The shared info of the environment
        :type shared_info: dict[str, Any]
        """
