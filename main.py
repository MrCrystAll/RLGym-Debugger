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

from rlgym_tools.rocket_league.renderers.rocketsimvis_renderer import RocketSimVisRenderer

import numpy as np

from reward_printing.reward_printer import RewardPrinter

if __name__ == "__main__":
    tick_skip = 1
    
    env = RLGym(
        obs_builder=DefaultObs(),
        action_parser=RepeatAction(LookupTableAction(), repeats=tick_skip),
        state_mutator=MutatorSequence(FixedTeamSizeMutator(1, 1), KickoffMutator()),
        reward_fn=TouchReward(),
        termination_cond=GoalCondition(),
        truncation_cond=NoopCondition(),
        transition_engine=RocketSimEngine(),
        renderer=RocketSimVisRenderer()
    )

    running = True
    render = True

    reward_printer = RewardPrinter()

    print("Starting environment")
    while running:
        try:
            obs = env.reset()
            reward_printer.reset(env.agents, env.state, env.shared_info)

            truncated = {agent: False for agent in env.agents}
            terminated = {agent: False for agent in env.agents}

            while not (any(truncated.values()) or any(terminated.values())):
                if render:
                    env.render()
                    time.sleep(tick_skip / 120.0)
                
                actions = {
                    agent: np.asarray([random.randint(0, 89)]) for agent in env.agents
                }
                
                reward_printer.intercept_actions(obs, actions, env.shared_info)

                env_return = env.step(actions)
                reward_printer.intercept(obs, *env_return, env.shared_info)
                
                obs = env_return[0]
        except KeyboardInterrupt:
            print("Interruption detected")
            running = False
            break

    env.close()
    print("Ending")
