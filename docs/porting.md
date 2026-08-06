# Porting to a new task

The pipeline above the env/expert layer is fully task-agnostic — dataset recording, ACT training, distillation, evaluation, and all utilities need zero changes. Only two files need to be written and two scripts need their imports swapped.

## 1. New environment

Implement the same interface as `ReachEnvironment`:

```python
# src/env/my_env.py
from env.base import Environment

class MyEnvironment(Environment):
    def reset(self) -> np.ndarray:
        # randomise initial state / target, return obs vector
        ...

    def step(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        # apply action, step physics; bool = task success / termination
        ...
```

The obs vector must have `qpos` in its first `joint_dim` elements — the eval scripts index `obs[:joint_dim]` to extract joint positions for normalisation.

## 2. New expert

Extend the `Expert` ABC in `src/expert/base.py`:

```python
# src/expert/my_expert.py
from expert.base import Expert

class MyExpert(Expert):
    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        # deterministic obs → joint control vector
        # any method works: IK, analytical solution, motion primitive
        ...
```

The expert only needs to produce good-enough demonstrations — it is discarded after data collection.

## 3. Config (`config/simulation.yaml`)

Replace the reach-specific env block with your task's parameters and point `scene_xml_path` at your MuJoCo XML. If your robot has a different DOF, also update `action_dim` and `joint_dim` in `config/train.yaml` and `config/distill.yaml`.

## 4. Swap imports in two scripts

`scripts/collect_data.py` and `scripts/eval_act.py` / `scripts/eval_distil.py` import `ReachEnvironment` and `ReachExpert` directly — replace those with your new classes. Everything else (`train_act.py`, `train_distil.py`, `distillation_measure.py`, all of `src/`) is unchanged.
