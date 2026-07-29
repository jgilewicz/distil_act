import numpy as np
import mujoco


def test_reset_obs_shape(env):
    obs = env.reset()
    assert obs.shape == (18,)


def test_step_returns_correct_shapes(env):
    env.reset()
    action = np.zeros(env.model.nu)
    obs, terminated = env.step(action)
    assert obs.shape == (18,)
    assert isinstance(terminated, (bool, np.bool_))


def test_step_terminated_at_target(env):
    env.reset()
    ee_pos = env.data.xpos[env.ee_id].copy()
    env.data.mocap_pos[env.target_mocap_id] = ee_pos
    mujoco.mj_forward(env.model, env.data)
    action = np.zeros(env.model.nu)
    _, terminated = env.step(action)
    assert terminated
