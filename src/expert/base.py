from abc import ABC, abstractmethod

import numpy as np


class Expert(ABC):
    @abstractmethod
    def compute_action(self, obs: np.ndarray) -> np.ndarray: ...

    def reset(self) -> None:
        pass
