from abc import abstractmethod
from typing import Generic

from rlgym.api import EngineActionType


class DeviceInterface(Generic[EngineActionType]):
    """A class that returns an input to be used in the game, used to map a device's inputs to a environment action"""

    @abstractmethod
    def get_output(self) -> EngineActionType:
        """Returns an action directly usable by the transition engine

        :return: An action directly usable by the transition engine
        :rtype: EngineActionType
        """
