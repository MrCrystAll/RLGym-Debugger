from pdb import run
import random

from rlgym.api import RLGym

from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.reward_functions import TouchReward
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, MutatorSequence, KickoffMutator
from rlgym.rocket_league.done_conditions import NoopCondition, GoalCondition
from rlgym.rocket_league.sim import RocketSimEngine

import numpy as np

if __name__ == "__main__":
    env = RLGym(
        obs_builder=DefaultObs(),
        action_parser=RepeatAction(LookupTableAction()),
        state_mutator=MutatorSequence(FixedTeamSizeMutator(1, 1), KickoffMutator()),
        reward_fn=TouchReward(),
        termination_cond=GoalCondition(),
        truncation_cond=NoopCondition(),
        transition_engine=RocketSimEngine()
    )
    
    running = True
    
    print("Starting environment")
    while running:
        try:
            obs = env.reset()
            
            truncated = {agent: False for agent in env.agents}
            terminated = {agent: False for agent in env.agents}
            
            while not (any(truncated.values()) or any(terminated.values())):
                actions = {agent: np.asarray([random.randint(0, 89)]) for agent in env.agents}
                
                next_obs, rewards, terminated, truncated = env.step(actions)
        except KeyboardInterrupt:
            print("Interruption detected")
            running = False
            break
        
    env.close()
    print("Ending")