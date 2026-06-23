from abc import ABC, abstractmethod

import numpy as np


class Expert(ABC):
    @abstractmethod
    def compute_action(self, obs: np.ndarray) -> np.ndarray: ...

    # stateful experts can override this to reset internal state between episodes
    def reset(self) -> None:
        pass
