import mujoco
import mujoco.viewer
import numpy as np
import cv2
from env.env import ReachEnvironment


class SceneRenderer:
    def __init__(
        self,
        env: ReachEnvironment,
        height: int = 480,
        width: int = 640,
        camera_list: list[str] = None,
        show_viewer: bool = True,
    ) -> None:
        self.env = env
        self.renderer = mujoco.Renderer(env.model, height=height, width=width)
        self.camera_names = camera_list if camera_list is not None else []
        self.show_viewer = show_viewer
        self.viewer = None

    def __enter__(self):
        if self.show_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.env.model, self.env.data)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.viewer is not None:
            self.viewer.close()

    def render_step(self, action: np.ndarray) -> tuple[np.ndarray, bool, dict]:
        obs, terminated = self.env.step(action)

        frames = {}
        for cam_name in self.camera_names:
            self.renderer.update_scene(self.env.data, camera=cam_name)
            rgb_frame = self.renderer.render()
            frames[cam_name] = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

        return obs, terminated, frames
