import os
import gdown
from typing import Optional
from environment.agent import Agent
from stable_baselines3 import PPO

class SubmittedAgent(Agent):
    """
    An agent that loads a pre-trained PPO model from a public URL.
    """
    def __init__(self, file_path: Optional[str] = None):
        # The 'file_path' is provided by the environment after downloading.
        super().__init__(file_path)

    def _initialize(self) -> None:
        """
        This method is called by the environment to load the model.
        """
        # If file_path is not provided during initialization, the _gdown method will be called first.
        # Then, this _initialize method will be called again with the path to the downloaded file.
        if self.file_path is None:
            # This path will be populated by the _gdown method in the next step of the server logic.
            pass
        else:
            self.model = PPO.load(self.file_path)

    def _gdown(self) -> str:
        """
        This method is called by the tournament server to download the model file.
        """
        data_path = "rl-model.zip"
        if not os.path.isfile(data_path):
            print(f"Downloading model from Google Drive: {data_path}...")
            url = "https://drive.google.com/file/d/1DwCALcgY2ttDoscsCTcEp2DuipSvA7Jn/view?usp=sharing"
            gdown.download(url, output=data_path, fuzzy=True)
        return data_path

    def predict(self, obs):
        """
        Uses the loaded RL model to make a decision based on the observation.
        """
        action, _ = self.model.predict(obs)
        return action