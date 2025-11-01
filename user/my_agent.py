import os
import random
from typing import Optional
from environment.agent import Agent

class SubmittedAgent(Agent):
    """
    An advanced rule-based agent with more sophisticated combat logic.
    - It prioritizes heavy attacks on stunned opponents.
    - It has a chance to dodge defensively when at high damage.
    - It uses jumps to recover more effectively when off-stage.
    - Its attacks are randomized to be less predictable.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        """
        The agent's "brain". It analyzes the game state (`obs`)
        and decides which action to take.
        """
        # === 1. OBSERVE: Get critical information from the game state ===
        my_pos = self.obs_helper.get_section(obs, 'player_pos')
        my_damage = self.obs_helper.get_section(obs, 'player_damage')
        
        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        # A "state" is a number representing what the opponent is currently doing.
        # States 5 and 11 mean the opponent is in "Stun" or "BrawlhallaStun".
        is_opp_stunned = self.obs_helper.get_section(obs, 'opponent_state') in [5, 11]
        
        # Calculate the distance to the opponent
        distance_to_opp = ((my_pos[0] - opp_pos[0])**2 + (my_pos[1] - opp_pos[1])**2)**0.5
        is_opp_close = distance_to_opp < 4.0

        # Start with a decision to do nothing
        action = self.act_helper.zeros()

        # === 2. THINK & DECIDE: A prioritized list of rules ===

        # --- Rule 1: The Ultimate Priority - Punish Stunned Opponents ---
        if is_opp_stunned and is_opp_close:
            # If the opponent is stunned and close, this is the best time to attack.
            # We will use a HEAVY attack ('k') for maximum punishment.
            action = self.act_helper.press_keys(['k'])
            return action # Decision made, no need to check other rules.

        # --- Rule 2: Self-Preservation - Defensive Dodge ---
        # If my damage is high (over 100) and the opponent is close...
        if my_damage > 100 and is_opp_close:
            # ...there's a 50% chance I'll try to dodge ('l') to safety.
            if random.random() < 0.5:
                action = self.act_helper.press_keys(['l'])
                return action # Decision made.

        # --- Rule 3: Recovery - Get Back on Stage ---
        if my_pos[0] > 10.67/2: # If I'm off the right side
            # Move left AND jump to get back
            action = self.act_helper.press_keys(['a', 'space'])
            return action
        elif my_pos[0] < -10.67/2: # If I'm off the left side
            # Move right AND jump to get back
            action = self.act_helper.press_keys(['d', 'space'])
            return action

        # --- Rule 4: On-Stage Combat Logic ---
        # If none of the high-priority rules above were triggered, engage in normal combat.
        
        # 4a. Movement: Move towards the opponent
        if (opp_pos[0] > my_pos[0]):
            action = self.act_helper.press_keys(['d'])  # Go right
        else:
            action = self.act_helper.press_keys(['a'])  # Go left

        # 4b. Jumping: Jump if the opponent is significantly above me
        if (my_pos[1] > opp_pos[1] + 2.0):
             action = self.act_helper.press_keys(['space'], action)

        # 4c. Attacking: If close to the opponent, choose an attack
        if is_opp_close:
            # Instead of always doing the same attack, let's be unpredictable.
            if random.random() < 0.7:
                # 70% chance to do a fast, light attack ('j')
                action = self.act_helper.press_keys(['j'], action)
            else:
                # 30% chance to do a slow but powerful heavy attack ('k')
                action = self.act_helper.press_keys(['k'], action)

        return action

    # The methods below are not used by this agent, but are required by the interface.
    def _initialize(self) -> None:
        pass
    def _gdown(self) -> str:
        pass
    def save(self, file_path: str) -> None:
        pass
    def learn(self, env, total_timesteps, log_interval: int = 4):
        pass