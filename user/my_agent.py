
import os
import random
from typing import Optional
from environment.agent import Agent

class SubmittedAgent(Agent):
    """
    An even more advanced rule-based agent. New features:
    - A top-priority "Survival Mode" to prevent falling off edges.
    - It now combines horizontal movement and jumping for a much more effective recovery.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        """
        The agent's "brain" with a new, high-priority survival instinct.
        """
        # === 1. OBSERVE: Get critical information from the game state ===
        my_pos = self.obs_helper.get_section(obs, 'player_pos')
        my_damage = self.obs_helper.get_section(obs, 'player_damage')
        
        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        is_opp_stunned = self.obs_helper.get_section(obs, 'opponent_state') in [5, 11]
        
        distance_to_opp = ((my_pos[0] - opp_pos[0])**2 + (my_pos[1] - opp_pos[1])**2)**0.5
        is_opp_close = distance_to_opp < 4.0

        action = self.act_helper.zeros()

        # === 2. THINK & DECIDE: A new, prioritized list of rules ===

        # --- NEW: Rule 0: SURVIVAL FIRST! ---
        # This is now the highest priority rule. If the agent is in danger of falling,
        # it will ignore everything else and try to recover.
        is_off_sides = my_pos[0] > 9 or my_pos[0] < -9 # Simplified horizontal check
        is_dangerously_low = my_pos[1] > 4.0 # Check if falling deep into the pit

        if is_off_sides or is_dangerously_low:
            # Determine horizontal direction to get back to center
            horizontal_direction_key = 'a' if my_pos[0] > 0 else 'd'
            
            # Focus on the core recovery action: horizontal movement and the recovery move (heavy attack).
            # This avoids potential input conflicts between jump and heavy attack on the same frame.
            action = self.act_helper.press_keys([horizontal_direction_key, 'k'])
            return action # CRITICAL: Survival is the only thing that matters now.

        # --- If not in danger, proceed with combat logic ---

        # --- Rule 1: Punish Stunned Opponents ---
        if is_opp_stunned and is_opp_close:
            action = self.act_helper.press_keys(['k'])
            return action

        # --- Rule 2: Defensive Dodge ---
        if my_damage > 100 and is_opp_close and random.random() < 0.5:
            action = self.act_helper.press_keys(['l'])
            return action

        # --- Rule 3: On-Stage Combat Logic ---
        
        # 3a. Movement: Move towards the opponent
        if (opp_pos[0] > my_pos[0]):
            action = self.act_helper.press_keys(['d'])
        else:
            action = self.act_helper.press_keys(['a'])

        # 3b. Jumping: Jump if the opponent is significantly above me
        if (my_pos[1] > opp_pos[1] + 2.0):
             action = self.act_helper.press_keys(['space'], action)

        # 3c. Attacking: If close, choose an attack unpredictably
        if is_opp_close:
            if random.random() < 0.7:
                action = self.act_helper.press_keys(['j'], action)
            else:
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
