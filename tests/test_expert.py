import numpy as np
import pytest
import torch
from expert.reach_expert import ReachExpert

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ReachExpert's curobo backend requires a GPU"
)


def test_compute_action_shape(env):
    expert = ReachExpert(env)
    obs = env.reset()
    action = expert.compute_action(obs)
    assert action.shape == (env.model.nu,)


def test_ik_convergence(env):
    env.rng = np.random.default_rng(1)
    expert = ReachExpert(env)
    obs = env.reset()
    terminated = False
    for _ in range(400):
        action = expert.compute_action(obs)
        obs, terminated = env.step(action)
        if terminated:
            break
    assert terminated, "Expert failed to reach target within 400 steps"
