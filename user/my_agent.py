import os
import gdown
import uuid  # Import the uuid library to generate unique IDs
from typing import Optional

import torch
from torch import nn as nn
from torch.nn import functional as F
import gymnasium as gym

from environment.agent import Agent
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# --- Class Definitions Copied from train_agent.py ---
class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int = 64, action_dim: int = 10, hidden_dim: int = 64):
        super(MLPPolicy, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim, dtype=torch.float32)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class MLPExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 64, hidden_dim: int = 64):
        super(MLPExtractor, self).__init__(observation_space, features_dim)
        self.model = MLPPolicy(
            obs_dim=observation_space.shape[0], 
            action_dim=10,
            hidden_dim=hidden_dim,
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)
    
    @classmethod
    def get_policy_kwargs(cls, features_dim: int = 64, hidden_dim: int = 64) -> dict:
        return dict(
            features_extractor_class=cls,
            features_extractor_kwargs=dict(features_dim=features_dim, hidden_dim=hidden_dim)
        )

# --- End of Copied Class Definitions ---


class SubmittedAgent(Agent):
    """
    An agent that loads a pre-trained PPO model from a public URL.
    Each instance of the agent will download the model to a unique file path.
    """
    def __init__(self, file_path: Optional[str] = None):
        # Generate a unique ID for this agent instance to prevent file conflicts.
        self.agent_id = str(uuid.uuid4())
        super().__init__(file_path)

    def _initialize(self) -> None:
        if self.file_path is None:
            pass
        else:
            # Use the `custom_objects` dictionary to robustly load the model architecture.
            custom_objects = {
                "policy_kwargs": MLPExtractor.get_policy_kwargs()
            }
            self.model = PPO.load(self.file_path, custom_objects=custom_objects)

    def _gdown(self) -> str:
        # Create a unique filename for this agent instance.
        data_path = f"rl-model-{self.agent_id}.zip"
        
        # Always download a fresh copy for each instance to avoid conflicts.
        print(f"Downloading model from Google Drive to unique path: {data_path}...")
        url = "https://drive.google.com/file/d/1DwCALcgY2ttDoscsCTcEp2DuipSvA7Jn/view?usp=sharing"
        gdown.download(url, output=data_path, fuzzy=True)
        return data_path

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action