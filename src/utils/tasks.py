from env.base import Environment
from env.pick_env import PickEnvironment
from env.reach_env import ReachEnvironment

ENV_CLASSES: dict[str, type[Environment]] = {
    "reach": ReachEnvironment,
    "pick": PickEnvironment,
}
