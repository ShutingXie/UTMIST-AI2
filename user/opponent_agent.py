
from environment.agent import Agent

# This is the original, simple rule-based agent.
# It only moves towards the opponent and uses light attacks.
# It does not have the more advanced survival or combat logic.
class SubmittedAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time = 0

    def predict(self, obs):
        self.time += 1
        pos = self.obs_helper.get_section(obs, 'player_pos')
        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        opp_KO = self.obs_helper.get_section(obs, 'opponent_state') in [5, 11]
        action = self.act_helper.zeros()

        # If off the edge, come back
        if pos[0] > 10.67/2:
            action = self.act_helper.press_keys(['a'])
        elif pos[0] < -10.67/2:
            action = self.act_helper.press_keys(['d'])
        elif not opp_KO:
            # Head toward opponent
            if (opp_pos[0] > pos[0]):
                action = self.act_helper.press_keys(['d'])
            else:
                action = self.act_helper.press_keys(['a'])

        # Jump if below map or opponent is above you
        if (pos[1] > 1.6 or pos[1] > opp_pos[1]) and self.time % 2 == 0:
            action = self.act_helper.press_keys(['space'], action)

        # Attack if near
        if (pos[0] - opp_pos[0])**2 + (pos[1] - opp_pos[1])**2 < 4.0:
            action = self.act_helper.press_keys(['j'], action)
        return action
