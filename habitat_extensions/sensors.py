from typing import Any, Dict

import numpy as np
from gym import spaces
from habitat.config import Config
from habitat.core.registry import registry
from habitat.core.simulator import Observations, Sensor, SensorTypes, Simulator
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

from habitat_extensions.shortest_path_follower import (
    ShortestPathFollowerCompat,
)
from habitat_extensions.task import VLNExtendedEpisode
from habitat.tasks.nav.nav import HeadingSensor
from habitat.tasks.vln.vln import VLNEpisode

@registry.register_sensor(name="GlobalGPSSensor")
class GlobalGPSSensor(Sensor):
    r"""The agents current location in the global coordinate frame

    Args:
        sim: reference to the simulator for calculating task observations.
        config: Contains the DIMENSIONALITY field for the number of dimensions
                to express the agents position
    Attributes:
        _dimensionality: number of dimensions used to specify the agents position
    """

    cls_uuid: str = "globalgps"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._dimensionality = getattr(config, "DIMENSIONALITY", 2)
        assert self._dimensionality in [2, 3]
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(self._dimensionality,),
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, **kwargs: Any):
        return self._sim.get_agent_state().position.astype(np.float32)


@registry.register_sensor(name="OrienSensor")
class OrienSensor(HeadingSensor):
    cls_uuid: str = "orientation"
    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        rotation_world_agent = agent_state.rotation
        res = np.array([*(rotation_world_agent.imag),rotation_world_agent.real])
        return res


@registry.register_sensor
class ShortestPathSensor(Sensor):
    r"""Sensor for observing the action to take that follows the shortest path
    to the goal.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """

    cls_uuid: str = "shortest_path_sensor"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        super().__init__(config=config)
        if config.USE_ORIGINAL_FOLLOWER:
            self.follower = ShortestPathFollowerCompat(
                sim, config.GOAL_RADIUS, return_one_hot=False
            )
            self.follower.mode = "geodesic_path"
        else:
            self.follower = ShortestPathFollower(
                sim, config.GOAL_RADIUS, return_one_hot=False
            )
        # self._sim = sim
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0.0, high=100, shape=(1,), dtype=np.float)

    def get_observation(self, *args: Any, episode, **kwargs: Any):
        best_action = self.follower.get_next_action(episode.goals[0].position)
        return np.array(
            [
                best_action
                if best_action is not None
                else HabitatSimActions.STOP
            ]
        )


@registry.register_sensor
class VLNOracleProgressSensor(Sensor):
    r"""Sensor for observing how much progress has been made towards the goal.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """

    cls_uuid: str = "progress"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float)

    def get_observation(
        self, observations, *args: Any, episode, **kwargs: Any
    ):
        current_position = self._sim.get_agent_state().position.tolist()

        distance_to_target = self._sim.geodesic_distance(
            current_position, episode.goals[0].position
        )


        if "geodesic_distance" not in episode.info.keys():
            distance_from_start = self._sim.geodesic_distance(
                episode.start_position, episode.goals[0].position
            )
            episode.info["geodesic_distance"] = distance_from_start

        distance_from_start = episode.info["geodesic_distance"]

        progress =  (distance_from_start - distance_to_target) / distance_from_start

        return np.array(progress, dtype = np.float32)


@registry.register_sensor
class RxRInstructionSensor(Sensor):

    cls_uuid: str = "rxr_instruction"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        # self.max_text_len = config.max_text_len
        # self.features_path = config.features_path
        # super().__init__(config=config)
        self.uuid = "instruction"
        self.observation_space = spaces.Discrete(0)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.MEASUREMENT

    # def _get_observation_space(self, *args: Any, **kwargs: Any):
    #     return spaces.Box(
    #         low=np.finfo(np.float32).min,
    #         high=np.finfo(np.float32).max,
    #         shape=(512, 768),
    #         dtype=np.float32,
    #     )

    def get_observation(
        self,
        observations: Dict[str, "Observations"],
        episode: VLNExtendedEpisode,
        **kwargs,
    ):
        return {
            "text": episode.instruction.instruction_text,
            "tokens": episode.instruction.instruction_tokens,
            "trajectory_id": episode.trajectory_id
        }
        # features = np.load(
        #     self.features_path.format(
        #         split=episode.instruction.split,
        #         id=int(episode.instruction.instruction_id),
        #         lang=episode.instruction.language.split("-")[0],
        #     ),
        # )
        # feats = np.zeros((self.max_text_len, 768), dtype=np.float32)
        # s = features["features"].shape
        # feats[: s[0], : s[1]] = features["features"][:self.max_text_len,:768]
        # return feats





##########################################################################################
##########################################################################################
######
###### CUSTOM SENSOR FOR R2RIE-CE
######
##########################################################################################
##########################################################################################
@registry.register_sensor(name="ErrorSensorContainsError")
class ErrorSensorContainsError(Sensor):
    '''
    responsible to handle the error data that we created
    '''
    def __init__(self, **kwargs):
        self.uuid = "error_sensor_contains_error"
        self.observation_space = spaces.Discrete(2)
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid
    def _get_observation_space(self, *args: Any, **kwargs: Any) -> spaces.Box:
        return self.observation_space

    def _get_observation(
        self,
        observations: Dict[str, Observations],
        episode: VLNEpisode,
        **kwargs,
    ):
        return episode.error_information.episode_contains_error,
    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)


@registry.register_sensor(name="ErrorSensorTokenSwappedError")
class ErrorSensorTokenSwappedError(Sensor):
    '''
    responsible to handle the error data that we created
    '''
    def __init__(self, **kwargs):
        self.uuid = "error_sensor_token_swapped_error"
        self.observation_space = spaces.Box(low=0, high=4, shape=(2, 2), dtype=np.int64)
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid
    def _get_observation_space(self, *args: Any, **kwargs: Any) -> spaces.Box:
        return self.observation_space

    def _get_observation(
        self,
        observations: Dict[str, Observations],
        episode: VLNEpisode,
        **kwargs,
    ):
        '''
        we could have the following cases: shape of the tensor (num_errors, tokens=2)
        -1 no error,
        -2 error in the direction token, so create a np array of shape (1,2)
        -3 error in the scene awareness token, so create a np array of shape (1,2)
        -4 error in both, so create a np array of shape (2,2)
        -5 all type of errors, so create a np array of shape (3,2)


        Given that we have to stack the tensor (for habitat batch operation), broadcast the error type to the shape of the max tensor (3,2)
        '''
        
        result = []
        token_swapped =  episode.error_information.token_swapped
        for data in token_swapped:
            token_id = data['token_id']
            token_id_string_position = data['token_id_position']
            # could happen that a word is composed by two token. If it is the case, take only the first
            if isinstance(token_id, list):
                token_id = token_id[0]
                token_id_string_position = token_id_string_position[0]

            result.append([token_id, token_id_string_position])
        if len(result) == 0:
            # case 1
            return np.array([[-1,-1],[-1,-1],[-1,-1]])
        elif len(result) == 1:
            # case 2 or 3
            return np.array([result[0], [-1,-1],[-1,-1]])
        elif len(result) == 2:
            return np.array([result[0], result[1],[-1,-1]])
        elif len(result) == 3:
            return np.array([result[0], result[1], result[2]])
        else: raise Exception("ErrorSensorTokenSwappedError should contains 1/2/3 error")
    
    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)

@registry.register_sensor(name="ErrorSensorErrorType")
class ErrorSensorErrorType(Sensor):
    '''
    responsible to handle the error data that we created
    '''
    def __init__(self, **kwargs):
        self.uuid = "error_sensor_error_type"
        self.observation_space = spaces.Discrete(4)
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid
    def _get_observation_space(self, *args: Any, **kwargs: Any) -> spaces.Box:
        return self.observation_space

    def _get_observation(
        self,
        observations: Dict[str, Observations],
        episode: VLNEpisode,
        **kwargs,
    ):
        return episode.error_information.error_type,
    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)
    
@registry.register_sensor(name="EpisodeIdSensor")
class EpisodeIdSensor(Sensor):
    '''
    for debugging purposes
    '''
    def __init__(self, **kwargs):
        self.uuid = "episode_id_UUID"
        self.observation_space = spaces.Discrete(1)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid
    def _get_observation_space(self, *args: Any, **kwargs: Any) -> spaces.Box:
        return self.observation_space

    def _get_observation(
        self,
        observations: Dict[str, Observations],
        episode: VLNEpisode,
        **kwargs,
    ):
        return episode.episode_id
    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)
