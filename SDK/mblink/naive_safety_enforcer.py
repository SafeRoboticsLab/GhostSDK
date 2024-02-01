from typing import Optional
import numpy as np
import os

from RARL.sac_mini import SAC_mini
from utils import load_config

class NaiveSafetyEnforcer:
    def __init__(self, epsilon: float=0.0, imaginary_horizon: int=100, shield_type: Optional[str]="value", parent_dir: Optional[str]="", version=0) -> None:
        """
        Naive Safety Enforcer that is trained using RL and Domain Randomization

        Args:
            epsilon (float, optional): The epsilon value to be used for value shielding, determining the conservativeness of safety enforcer. Defaults to 0.0.
            imaginary_horizon (int, optional): The horizon to be used for rollout-based shielding. Defaults to 100.
            shield_type (Optional[str], optional): The shielding type to be used, choose from ["value", "rollout"]. Defaults to "value".
        """
        #! TODO: Apply rollout-based shielding with the simulator
        if shield_type != "value":
            raise NotImplementedError

        self.epsilon = epsilon
        self.imaginary_horizon = imaginary_horizon
        self.version = version

        if version == 0:
            # binary naive
            training_dir = "train_result/spirit_naive_binary_f5_newStateDef/spirit_naive_binary_f5_newStateDef_05"
            load_dict = {
                "ctrl": 1_100_000
            }
        elif version == 1:
            # ISAACS reachavoid naive f0
            training_dir = "train_result/spirit_naive_reachavoid_f0_failure_newStateDef2/spirit_naive_reachavoid_f0_failure_newStateDef2_00"
            load_dict = {
                "ctrl": 4_400_000
            }
        model_path = os.path.join(parent_dir, training_dir, "model")        
        model_config_path = os.path.join(parent_dir, training_dir, "config.yaml")

        config_file = os.path.join(parent_dir, model_config_path)
        
        if not os.path.exists(config_file):
            raise ValueError("Cannot find config file for the model, terminated")

        config = load_config(config_file)
        config_arch = config['arch']
        config_update = config['update']

        self.policy = SAC_mini(config_update, config_arch)
        self.policy.build_network(verbose=False)
        print("Loading frozen weights of model at {} with load_dict {}".format(model_path, load_dict))
        self.policy.restore(load_dict["ctrl"], model_path)
        print("-> Done")

        self.is_shielded = None
        self.prev_q = None

    def get_action(self, state:np.ndarray, action:np.ndarray) -> np.ndarray:
        if self.version >= 1:
            # change from 36D to 33D (ignore x, y, yaw: 0, 1, 8 index)
            state = np.concatenate((state[2:8], state[9:]), axis=0)

        critic_q = max(
            self.policy.critic(
                state, action
            )
        )

        if critic_q > self.epsilon:
            action = self.policy.actor(state)
            self.is_shielded = True
        else:
            self.is_shielded = False
        
        self.prev_q = critic_q.reshape(-1)[0]

        return action

    def get_q(self, state:np.ndarray=None, action:np.ndarray=None) -> float:
        if state is not None and action is not None:
            if self.version >= 1:
                # change from 36D to 33D (ignore x, y, yaw: 0, 1, 8 index)
                state = np.concatenate((state[2:8], state[9:]), axis=0)

            critic_q = max(
                self.policy.critic(
                    state, action
                )
            )

            return critic_q.reshape(-1)[0]
        else:
            return self.prev_q
    
    def get_shielding_status(self):
        return self.is_shielded
