# Headless rendering (no display)

MuJoCo reads `MUJOCO_GL` **at import time** — it must be set in the shell before the process starts.

`just collect-headless` already sets `MUJOCO_GL=egl`. If you run the script directly:

```bash
MUJOCO_GL=egl SHOW_VIEWER=false uv run python3 scripts/collect_data.py
```

## Choosing a backend

| Backend | When to use | Requirement |
|---------|-------------|-------------|
| `egl` | GPU or any Mesa EGL (recommended for servers) | `libegl1` |
| `osmesa` | No GPU / EGL unavailable (software fallback) | `libosmesa6` |
| `disabled` | Physics-only, no rendering at all | nothing |

Install EGL (Mesa, CPU-only machines):

```bash
apt-get install -y libegl1 libgl1
```

If EGL still fails (`gladLoadGL error`), fall back to OSMesa:

```bash
apt-get install -y libosmesa6
MUJOCO_GL=osmesa SHOW_VIEWER=false uv run python3 scripts/collect_data.py
```

The Docker image ships with `libegl1` + `libgl1` and sets `MUJOCO_GL=disabled` for tests (physics only). Switch it to `egl` for any container that needs to render frames.
