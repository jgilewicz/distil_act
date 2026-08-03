# curobo patterns

Both experts (`src/expert/reach_expert.py`, `src/expert/pick_expert.py`) use
NVIDIA `curobo` against `models/piper.urdf` — a URDF converted from the
PiPER MJCF via `mjcf_urdf_simple_converter`, since curobo needs a URDF, not
MJCF. Requires CUDA; there's no CPU fallback path in this repo (tests
`skipif(not torch.cuda.is_available())`).

## Two curobo entry points — pick the right one

| Task shape | curobo API | Example |
|---|---|---|
| Single target pose, no obstacles to plan around | `InverseKinematics` | `ReachExpert` — solve one pose per step, re-solved every step as the target/current state changes |
| Multi-phase trajectory (approach, contact, retreat) with scene collision | `MotionPlanner` / `plan_grasp` | `PickExpert` — one plan per episode, replayed phase by phase |

If your new task is "move the end effector to reach a point" — use
`InverseKinematics`. If it's "approach an object, interact with it, then
move away," and there's geometry to avoid on the way — use `MotionPlanner`.
Don't reach for `MotionPlanner` by default; it needs a scene collision YAML
(see below) that IK doesn't.

## Shared setup: `RobotBuilder` and `models/config.yaml`

Both experts build (or load) the same robot collision config, independent of
task:

```python
if not os.path.exists(config_path):
    builder = RobotBuilder(
        urdf_path=urdf_path,
        asset_path=asset_path,
        tool_frames=["link6"],   # explicit — see gotcha below
        device_cfg=device_cfg,
    )
    builder.fit_collision_spheres(sphere_density=..., coverage_weight=..., protrusion_weight=..., compute_metrics=...)
    builder.compute_collision_matrix(prune_collisions=..., num_samples=...)
    config = builder.build()
    builder.save(config, config_path)
```

This is expensive (spheres are fit numerically), so it runs once and caches
to `models/config.yaml` — every expert since reach reuses that cache. **Do
not add a per-task collision config** unless the new task uses a different
robot; `config_path` defaults to the same `models/config.yaml` for every
task today.

**Gotcha:** `tool_frames=["link6"]` must be passed explicitly. The converted
URDF's gripper joints collapse to fixed joints, which leaves `link7`/`link8`
as ambiguous leaf frames — without an explicit tool frame, `RobotBuilder`
can't tell which leaf is the actual end effector.

## `InverseKinematics` (position-only IK)

`Pose(position=...)` with `quaternion` unset defaults to identity — not
PiPER's actual rest orientation at `link6`. Left alone, the solver fights an
unreachable combined position+orientation target and never converges.
`ReachExpert` zeroes the rotation axes via `ToolPoseCriteria`:

```python
ToolPoseCriteria(
    terminal_pose_axes_weight_factor=[1, 1, 1, 0, 0, 0],       # x,y,z on; roll,pitch,yaw off
    non_terminal_pose_axes_weight_factor=[1, 1, 1, 0, 0, 0],
    device_cfg=device_cfg,
)
```

This makes the solve genuinely position-only IK. If your new task needs a
specific end-effector orientation (not just a target point), set the last
three weights on and pass an actual target quaternion — don't leave them at
`[1,1,1,0,0,0]` and expect orientation to be respected.

Check `result.success` / `result.position_error` and log (not silently
swallow) a failed solve — see both experts' pattern of `logger.error(...)`
on non-convergence rather than raising or returning garbage silently.

## `MotionPlanner` / `plan_grasp` (multi-phase trajectories)

Needs a **scene collision YAML** in `models/` that mirrors the scene XML's
static geometry (table, obstacles) — `models/pick.yaml` mirrors
`models/pick_scene.xml`'s table + box. This is a second, task-specific
config on top of the shared robot config — if your new task adds a new
scene XML with new obstacles, it needs a matching curobo world config or the
planner won't know to avoid them.

`plan_grasp` returns named sub-trajectory results (e.g.
`approach_interpolated_trajectory`, `grasp_interpolated_trajectory`,
`lift_interpolated_trajectory`), each paired with a `*_last_tstep` that
crops the interpolation buffer to the actually-used portion — always slice
with `last_tstep`, the raw trajectory array is padded. `PickExpert` drives
its own phase state machine (`approach → grasp → close → lift`) by stepping
through these trajectories in order; `close` isn't returned by the planner
at all — it's synthesized locally by holding the final grasp waypoint for
`close_steps` steps so the gripper actuator has time to physically close
before `lift` starts. If your new task needs a "settle" phase, follow this
same pattern rather than expecting curobo to produce one.

Gripper ctrl (open/closed) is **never** part of the curobo solve in either
expert — it's a separate actuator (`robot_gripper`) set directly per phase
via a plain dict lookup (`gripper_by_phase`). curobo only ever solves/plans
over the 6 arm joints.

## Joint ordering — the recurring bug source

curobo's joint order (`ik.joint_names` / `planner.joint_names`) is not
guaranteed to match MuJoCo's actuator order, and MuJoCo joints are prefixed
`robot_` after merging (see `build_model` in `src/env/base.py`) while
curobo's URDF-derived names aren't. Both experts handle this by building
explicit index arrays once, in `__init__`, and reindexing every step:

```python
self.ik_qpos_idx = [env.model.joint(f"robot_{name}").qposadr[0] for name in self.ik.joint_names]
self.ik_joint_order = [self.ik.joint_names.index(name.removeprefix("robot_")) for name in arm_joint_names]
```

When writing a new expert, build these mappings once in `__init__`, never
per-step — re-deriving them every `compute_action` call is wasted work and
an easy place to introduce an off-by-one under review.

## Testing

`tests/test_expert.py` is the pattern to copy: gate the whole module with
`pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason=...)`
since curobo needs a GPU and CI runners don't have one (see root
`CLAUDE.md`'s CI section). Test the action shape unconditionally and, for a
convergence-style task, run enough steps that a correct expert should
succeed and assert `terminated`.
