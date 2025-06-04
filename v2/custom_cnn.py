import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

class SmallCNN(BaseFeaturesExtractor):
    """CNN feature extractor optimized for 72x80 grayscale frames."""
    def __init__(self, observation_space: spaces.Box):
        assert len(observation_space.shape) == 3
        n_input_channels = observation_space.shape[0]
        super().__init__(observation_space, features_dim=1)

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, 512), nn.ReLU())
        self._features_dim = 512

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
