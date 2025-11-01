
import os
from typing import Optional
from environment.agent import Agent

class SubmittedAgent(Agent):
    """
    A rule-based agent that implements a simple strategy.
    - If off-stage, it tries to recover.
    - If on-stage, it moves towards the opponent.
    - If close to the opponent, it attacks.
    - It will jump if the opponent is above it.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time = 0

    def predict(self, obs):
        """
        This is the main decision-making function of the agent.
        It takes the game state (`obs`) and returns an action.
        """
        self.time += 1
        
        # Use the observation helper to get understandable game state information
        pos = self.obs_helper.get_section(obs, 'player_pos')
        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        
        # Check if the opponent is knocked out
        opp_KO = self.obs_helper.get_section(obs, 'opponent_state') in [5, 11]
        
        # Start with no action
        action = self.act_helper.zeros()

        # Rule 1: If off the stage, try to get back
        if pos[0] > 10.67/2:
            action = self.act_helper.press_keys(['a'])  # Go left
        elif pos[0] < -10.67/2:
            action = self.act_helper.press_keys(['d'])  # Go right
        
        # Rule 2: If on stage and opponent is not knocked out, move towards them
        elif not opp_KO:
            if (opp_pos[0] > pos[0]):
                action = self.act_helper.press_keys(['d'])  # Go right
            else:
                action = self.act_helper.press_keys(['a'])  # Go left

        # Rule 3: Jump if below the stage or if the opponent is above you
        # (We add `self.time % 2 == 0` to make it jump less frequently)
        if (pos[1] > 1.6 or pos[1] > opp_pos[1]) and self.time % 2 == 0:
            action = self.act_helper.press_keys(['space'], action)

        # Rule 4: If very close to the opponent, perform a light attack
        if (pos[0] - opp_pos[0])**2 + (pos[1] - opp_pos[1])**2 < 4.0:
            action = self.act_helper.press_keys(['j'], action)
            
        return action

    # The following methods are not used by this simple agent,
    # but are required by the Agent interface.
    def _initialize(self) -> None:
        pass

    def _gdown(self) -> str:
        pass

    def save(self, file_path: str) -> None:
        pass

    def learn(self, env, total_timesteps, log_interval: int = 4):
        pass
