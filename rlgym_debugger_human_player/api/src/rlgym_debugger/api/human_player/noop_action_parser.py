from abc import ABC
from typing import Any, Dict, Generic

from rlgym.api import (
    ActionParser,
    EngineActionType,
    StateType,
    ActionSpaceType,
    AgentID,
)


class NoopActionParser(
    ABC,
    Generic[AgentID, EngineActionType, StateType, ActionSpaceType],
    ActionParser[
        AgentID, EngineActionType, EngineActionType, StateType, ActionSpaceType
    ],
):
    """A parser that only returns the action, this is an abstract class.
    
    You need to implement it for your environment with the size of your engine action"""

    def parse_actions(
        self,
        actions: Dict[AgentID, EngineActionType],
        state: StateType,
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, EngineActionType]:
        return actions
