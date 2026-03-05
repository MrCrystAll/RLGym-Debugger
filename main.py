import random
import time

from rlgym.api import RLGym

from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.reward_functions import TouchReward
from rlgym.rocket_league.state_mutators import (
    FixedTeamSizeMutator,
    MutatorSequence,
    KickoffMutator,
)
from rlgym.rocket_league.done_conditions import NoopCondition, GoalCondition
from rlgym.rocket_league.sim import RocketSimEngine

from rlgym_tools.rocket_league.renderers.rocketsimvis_renderer import (
    RocketSimVisRenderer,
)

import numpy as np

from rlgym_debugger.api.human_player.human_player import HumanPlayer
from rlgym_debugger.api.multi_agents.multi_agent_action_parser import (
    MultiAgentsActionParser,
)
from rlgym_debugger.api.multi_agents.set_id_mutator import SetIDMutator
from rlgym_debugger.rocket_league.human_player.keyboard_player import KeyboardInterface
from rlgym_debugger.rocket_league.human_player.noop_action_parser import (
    RLNoopActionParser,
)

if __name__ == "__main__":
    tick_skip = 1

    env = RLGym(
        obs_builder=DefaultObs(),
        action_parser=MultiAgentsActionParser(
            {"human": RLNoopActionParser()}, default_component=LookupTableAction()
        ),
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(1, 1),
            KickoffMutator(),
            SetIDMutator(["human"], ["orange-0"]),
        ),
        reward_fn=TouchReward(),
        termination_cond=GoalCondition(),
        truncation_cond=NoopCondition(),
        transition_engine=RocketSimEngine(),
        renderer=RocketSimVisRenderer(),
    )

    running = True
    render = True

    human_player = HumanPlayer(KeyboardInterface(), lambda _: "human")

    print("Starting environment")
    while running:
        try:
            obs = env.reset()

            truncated = {agent: False for agent in env.agents}
            terminated = {agent: False for agent in env.agents}

            while not (any(truncated.values()) or any(terminated.values())):
                if render:
                    env.render()
                    time.sleep(tick_skip / 120.0)

                actions = {
                    agent: np.asarray([random.randint(0, 89)]) for agent in env.agents
                }

                human_player.intercept_actions(obs, actions, env.state, env.shared_info)

                env_return = env.step(actions)

                obs = env_return[0]
                terminated = env_return[2]
                truncated = env_return[3]
        except KeyboardInterrupt:
            print("Interruption detected")
            running = False
            break

    env.close()
    print("Ending")
