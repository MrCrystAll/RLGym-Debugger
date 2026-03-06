from typing import Any, Dict
from rlgym.api import StateMutator
from rlgym.rocket_league.api import GameState


class SetIDMutator(StateMutator[GameState]):
    def __init__(self, blue_agents: list[str], orange_agents: list[str]) -> None:
        self.blue_agents = blue_agents
        self.orange_agents = orange_agents

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        n_blue = n_orange = 0
        for agent, car in state.cars.copy().items():
            if car.is_blue:
                state.cars[self.blue_agents[n_blue]] = state.cars.pop(agent)
                n_blue += 1
            else:
                state.cars[self.orange_agents[n_orange]] = state.cars.pop(agent)
                n_orange += 1

        print(list(state.cars.keys()))
