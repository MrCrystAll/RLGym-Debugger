from collections.abc import Hashable
from typing import Any, Dict, List

import numpy as np
from rlgym.rocket_league.api import GameState

from rlgym_debugger.api.human_player.noop_action_parser import NoopActionParser


class RLNoopActionParser(
    NoopActionParser[Hashable, np.ndarray, GameState, tuple[str, int]]
):
    """The Rocket League implementation of the noop action parser"""

    def get_action_space(self, agent: Hashable) -> tuple[str, int]:
        return "no-format", 8

    def reset(
        self,
        agents: List[Hashable],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        pass
