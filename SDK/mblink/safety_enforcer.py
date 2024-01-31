from typing import Optional
import numpy as np
import os
import torch

from RARL.sac_adv import SAC_adv
from utils import load_config

class SafetyEnforcer:
    def __init__(self, epsilon: float=0.0, imaginary_horizon: int=100, shield_type: Optional[str]="value", parent_dir: Optional[str]="", version=4) -> None:
        """_summary_

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
            training_dir = "train_result/spirit_isaacs_avoidonly_f5_newStateDef_pretrained/spirit_isaacs_avoidonly_f5_newStateDef_pretrained_05"
            # isaacs pretrained
            load_dict = {
                "ctrl": 3_980_000,
                "dstb": 8_000_001
            }
        elif version == 1:
            training_dir = "train_result/spirit_isaacs_avoidonly_f5_newStateDef/spirit_isaacs_avoidonly_f5_newStateDef_05"
            # isaacs no pretrained
            load_dict = {
                "ctrl": 3_920_000,
                "dstb": 5_480_000
            }
        elif version == 2:
            training_dir = "train_result/spirit_isaacs_reachavoid_f5_newStateDef_pretrained/spirit_isaacs_reachavoid_f5_newStateDef_pretrained_05"
            # old reach-avoid
            load_dict = {
                "ctrl": 4_100_000,
                "dstb": 7_520_000
            }
        elif version == 3:
            training_dir = "train_result/spirit_isaacs_reachavoid_f5_pretrained_newStateDef2/spirit_isaacs_reachavoid_f5_pretrained_newStateDef2_05"
            # new reach-avoid (chiara)
            load_dict = {
                "ctrl": 5_900_000,
                "dstb": 8_000_001
            }
        elif version == 4:
            training_dir = "train_result/spirit_isaacs_reachavoid_f5_pretrained_newStateDef2_mirror/spirit_isaacs_reachavoid_f5_pretrained_newStateDef2_mirror_05"
            # mirrored dstb reach-avoid
            load_dict = {
                "ctrl": 4_700_000,
                "dstb": 8_000_001
            }
        else:
            raise NotImplementedError

        model_path = os.path.join(parent_dir, training_dir, "model")
        model_config_path = os.path.join(parent_dir, training_dir, "config.yaml")

        config_file = os.path.join(parent_dir, model_config_path)
        
        if not os.path.exists(config_file):
            raise ValueError("Cannot find config file for the model, terminated")
        
        config = load_config(config_file)
        config_arch = config['arch']
        config_update = config['update']

        self.policy = SAC_adv(config_update, config_arch)
        self.policy.build_network(verbose=True)
        print("Loading frozen weights of model at {} with load_dict {}".format(model_path, load_dict))
        self.policy.restore(None, model_path, load_dict=load_dict)
        print("-> Done")

        self.is_shielded = None
        self.prev_q = None

    def get_action(self, state:np.ndarray, action:np.ndarray) -> np.ndarray:
        if self.version >= 3: 
            # change from 36D to 33D (ignore x, y, yaw: 0, 1, 8 index)
            state = np.concatenate((state[2:8], state[9:]), axis=0)
        s_dstb = np.concatenate((state, action), axis=0)
        dstb = self.policy.dstb(s_dstb)

        critic_q = max(
            self.policy.adv_critic(
                state, action, dstb
            )
        )

        if critic_q > self.epsilon:
            action = self.policy.ctrl(state)
            self.is_shielded = True
        else:
            self.is_shielded = False
        
        self.prev_q = critic_q.reshape(-1)[0]

        return action

    def get_q(self, state:np.ndarray, action:np.ndarray):
        if self.version >= 3:
            state = np.concatenate((state[2:8], state[9:]), axis=0)
        s_dstb = np.concatenate((state, action), axis=0)
        dstb = self.policy.dstb(s_dstb)

        critic_q = max(
            self.policy.adv_critic(
                state, action, dstb
            )
        )
        
        return critic_q
    
    def get_safety_action(self, state):
        return self.policy.ctrl(state)
    
    def get_shielding_status(self):
        return self.is_shielded
