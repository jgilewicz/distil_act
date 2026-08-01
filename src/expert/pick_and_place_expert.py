import numpy as np

from env.pick_and_place_env import PickAndPlaceEnvironment
from expert.base import Expert


class PickAndPlaceExpert(Expert):
    def __init__(self, env: PickAndPlaceEnvironment) -> None:
        self.env = env

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError
