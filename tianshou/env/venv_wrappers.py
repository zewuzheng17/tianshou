from typing import Any, List, Optional, Tuple, Union

import numpy
import numpy as np
import gym

from tianshou.env.venvs import GYM_RESERVED_KEYS, BaseVectorEnv
from tianshou.utils import RunningMeanStd


class VectorEnvWrapper(BaseVectorEnv):
    """Base class for vectorized environments wrapper."""

    def __init__(self, venv: BaseVectorEnv) -> None:
        self.venv = venv
        self.is_async = venv.is_async

    def __len__(self) -> int:
        return len(self.venv)

    def __getattribute__(self, key: str) -> Any:
        if key in GYM_RESERVED_KEYS:  # reserved keys in gym.Env
            return getattr(self.venv, key)
        else:
            return super().__getattribute__(key)

    def get_env_attr(
        self,
        key: str,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ) -> List[Any]:
        return self.venv.get_env_attr(key, id)

    def set_env_attr(
        self,
        key: str,
        value: Any,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ) -> None:
        return self.venv.set_env_attr(key, value, id)

    def reset(
        self,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[dict]]]:
        return self.venv.reset(id, **kwargs)

    def step(
        self,
        action: np.ndarray,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.venv.step(action, id)

    def seed(
        self,
        seed: Optional[Union[int, List[int]]] = None,
    ) -> List[Optional[List[int]]]:
        return self.venv.seed(seed)

    def render(self, **kwargs: Any) -> List[Any]:
        return self.venv.render(**kwargs)

    def close(self) -> None:
        self.venv.close()


class VectorEnvNormObs(VectorEnvWrapper):
    """An observation normalization wrapper for vectorized environments.

    :param bool update_obs_rms: whether to update obs_rms. Default to True.
    :param float clip_obs: the maximum absolute value for observation. Default to
        10.0.
    :param float epsilon: To avoid division by zero.
    """

    def __init__(
        self,
        venv: BaseVectorEnv,
        update_obs_rms: bool = True,
        clip_obs: float = 10.0,
        epsilon: float = np.finfo(np.float32).eps.item(),
    ) -> None:
        super().__init__(venv)
        # initialize observation running mean/std
        self.update_obs_rms = update_obs_rms
        self.obs_rms = RunningMeanStd()
        self.clip_max = clip_obs
        self.eps = epsilon

    def reset(
        self,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[dict]]]:
        retval = self.venv.reset(id, **kwargs)
        reset_returns_info = isinstance(
            retval, (tuple, list)
        ) and len(retval) == 2 and isinstance(retval[1], dict)
        if reset_returns_info:
            obs, info = retval
        else:
            obs = retval

        if isinstance(obs, tuple):
            raise TypeError(
                "Tuple observation space is not supported. ",
                "Please change it to array or dict space",
            )

        if self.obs_rms and self.update_obs_rms:
            self.obs_rms.update(obs)
        obs = self._norm_obs(obs)
        if reset_returns_info:
            return obs, info
        else:
            return obs

    def step(
        self,
        action: np.ndarray,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obs, rew, done, info = self.venv.step(action, id)
        if self.obs_rms and self.update_obs_rms:
            self.obs_rms.update(obs)
        return self._norm_obs(obs), rew, done, info

    def _norm_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_rms:
            obs = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.eps)
            obs = np.clip(obs, -self.clip_max, self.clip_max)
        return obs

    def set_obs_rms(self, obs_rms: RunningMeanStd) -> None:
        """Set with given observation running mean/std."""
        self.obs_rms = obs_rms

    def get_obs_rms(self) -> RunningMeanStd:
        """Return observation running mean/std."""
        return self.obs_rms


class VectorEnvVecObs(VectorEnvWrapper):
    """An observation vectorization wrapper for vectorized environments.
    :param float clip_obs: the maximum absolute value for observation. Default to
        10.0.
    """

    def __init__(
        self,
        venv: BaseVectorEnv,
    ) -> None:
        super().__init__(venv)
        size = 0
        low = np.array([])
        high = np.array([])
        for k, v in venv.observation_space.items():
            if isinstance(v, gym.spaces.Box):
                assert len(v.shape) in [0, 1], "the observation space it not vector, but matrix!"
                if len(v.shape) == 0:
                    size += 1
                    low = np.concatenate((low, np.expand_dims(v.low, 0)), axis = None)
                    high = np.concatenate((high, np.expand_dims(v.high, 0)), axis = None)
                else:
                    size += v.shape[0]
                    low = np.concatenate((low, v.low), axis=None)
                    high = np.concatenate((high, v.high), axis=None)
                dtype = v.dtype
            else:
                raise ValueError("Envs contains discrete observations, pls check!")

        self.observation_spaces = gym.spaces.Box(
            low=low,
            high=high,
            shape=(size,),
            dtype=dtype
        )

    def reset(
        self,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[dict]]]:
        retval = self.venv.reset(id, **kwargs)
        reset_returns_info = isinstance(
            retval, (tuple, list)
        ) and len(retval) == 2 and isinstance(retval[1], dict)
        if reset_returns_info:
            obs, info = retval
        else:
            obs = retval

        if isinstance(obs, tuple):
            raise TypeError(
                "Tuple observation space is not supported. ",
                "Please change it to array or dict space",
            )
        elif isinstance(obs, numpy.ndarray):
            raise  TypeError(
                "This is specific for dmc control suite, not for gym envs",
                "Please use ordinary envpool gym envs"
            )

        obs = self._vec_obs(obs)
        if reset_returns_info:
            return obs, info
        else:
            return obs

    def step(
        self,
        action: np.ndarray,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obs, rew, done, info = self.venv.step(action, id)
        return self._vec_obs(obs), rew, done, info

    def _vec_obs(self, obs) -> np.ndarray:
        for v in obs.values():
            paralled_num = v.shape[0]
            break

        obss = np.empty([paralled_num, 0])
        iter = 0
        for k, v in obs.items():
            iter += 1
            # assert len(v.shape) in [1,2], "the observation shape is not valid, pls check!"
            if len(v.shape) == 1:
                v = np.expand_dims(v, 1)
            obss = np.concatenate((obss, v), axis=1)
        return obss



