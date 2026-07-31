from env.base import Environment
from env.pick_and_place_env import PickAndPlaceEnvironment
from env.reach_env import ReachEnvironment

ENV_CLASSES: dict[str, type[Environment]] = {
    "reach": ReachEnvironment,
    "pick_and_place": PickAndPlaceEnvironment,
}
