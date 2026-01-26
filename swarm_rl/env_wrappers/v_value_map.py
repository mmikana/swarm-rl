import copy

import gymnasium as gym
import numpy as np
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs

from gym_art.quadrotor_multi.tests.plot_v_value_2d import plot_v_value_2d


class V_ValueMapWrapper(gym.Wrapper):
    def __init__(self, env, model, render_mode=None):
        """A wrapper that visualize V-value map at each time step"""
        gym.Wrapper.__init__(self, env)
        self._render_mode = render_mode
        self.curr_obs = None
        self.model = model

    def __getstate__(self):
        # Exclude the model from pickling since it contains unpickleable objects
        state = self.__dict__.copy()
        state['_has_model'] = self.model is not None
        if '_has_model':
            del state['model']
        return state

    def __setstate__(self, state):
        # Restore the state without the model
        has_model = state.pop('_has_model', False)
        self.__dict__.update(state)
        # The model needs to be restored separately after unpickling
        self.model = None

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        self.curr_obs = obs
        return obs, info

    def step(self, action):
        obs, reward, info, terminated, truncated = self.env.step(action)
        self.curr_obs = obs
        return obs, reward, info, terminated, truncated

    def render(self):
        # Check if we should render the V-value map
        if self._render_mode == 'rgb_array' and self.model is not None:
            # Render the environment first
            frame = self.env.render(mode='rgb_array')  # Explicitly request rgb_array mode
            if frame is not None and len(frame.shape) >= 3:  # Ensure frame is valid
                # Get the V-value map
                height, width = frame.shape[0], frame.shape[1]
                v_value_map_2d = self.get_v_value_map_2d(width=width, height=height)

                # Ensure both frames have the same height before concatenating
                if v_value_map_2d.shape[0] != frame.shape[0]:
                    # Resize v_value_map_2d to match frame height
                    import cv2
                    v_value_map_2d = cv2.resize(v_value_map_2d, (v_value_map_2d.shape[1], frame.shape[0]))

                # Concatenate the environment frame with the V-value map
                combined_frame = np.concatenate((frame, v_value_map_2d), axis=1)
                return combined_frame
            else:
                # If environment frame is invalid, just return the V-value map
                v_value_map_2d = self.get_v_value_map_2d()
                return v_value_map_2d
        else:
            # If not in rgb_array mode or model is not available, just render the environment
            return self.env.render()

    def get_v_value_map_2d(self, width=None, height=None):
        if self.model is None:
            # Return empty frame if model is not available
            w = width if width else 400
            h = height if height else 400
            return np.zeros((h, w, 3), dtype=np.uint8)

        tmp_score = []
        idx = []
        idy = []
        rnn_states = None
        obs = dict(obs=np.array(self.curr_obs))
        normalized_obs = prepare_and_normalize_obs(self.model, obs)

        # Check if normalized_obs has the right shape
        if normalized_obs['obs'].ndim < 2 or normalized_obs['obs'].shape[0] == 0:
            # Return empty frame if obs is invalid
            w = width if width else 400
            h = height if height else 400
            return np.zeros((h, w, 3), dtype=np.uint8)

        init_x, init_y = copy.deepcopy(normalized_obs['obs'][0][0]), copy.deepcopy(normalized_obs['obs'][0][1])
        for i in range(-10, 11):
            ti_score = []
            for j in range(-10, 11):
                normalized_obs['obs'][0][0] = init_x + i * 0.2
                normalized_obs['obs'][0][1] = init_y + j * 0.2

                # x = self.model.forward_head(self.curr_obs)
                # x, new_rnn_states = self.model.forward_core(x, rnn_states)
                # result = self.model.forward_tail(x, values_only=True, sample_actions=True)
                result = self.model.forward(normalized_obs, rnn_states, values_only=True)

                ti_score.append(result['values'].item())
                idx.append(i * 0.2)
                idy.append(j * 0.2)

            tmp_score.append(ti_score)

        idx, idy, tmp_score = np.array(idx), np.array(idy), np.array(tmp_score)
        v_value_map_2d = plot_v_value_2d(idx, idy, tmp_score, width=width, height=height)

        return v_value_map_2d
