from abc import ABC, abstractmethod

import numpy as np


class Environment(ABC):
    @abstractmethod
    def reset(self) -> np.ndarray: ...

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[np.ndarray, bool]: ...
