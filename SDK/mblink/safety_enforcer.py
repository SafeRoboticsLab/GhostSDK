from typing import Optional
import numpy as np
import os
import torch

from RARL.sac_adv import SAC_adv
from utils import load_config

class SafetyEnforcer:
    def __init__(self, epsilon: float=0.0, imaginary_horizon: int=100, shield_type: Optional[str]="value", parent_dir: Optional[str]="") -> None:
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

        training_dir = "train_result/spirit_isaacs_avoidonly_f5_newStateDef/spirit_isaacs_avoidonly_f5_newStateDef_05"
        model_path = os.path.join(parent_dir, training_dir, "model")
        model_config_path = os.path.join(parent_dir, training_dir, "config.yaml")

        config_file = os.path.join(parent_dir, model_config_path)
        
        if not os.path.exists(config_file):
            raise ValueError("Cannot find config file for the model, terminated")
        
        config = load_config(config_file)
        config_arch = config['arch']
        config_update = config['update']

        load_dict = {
            "ctrl": 3_920_000,
            "dstb": 5_480_000
        }

        self.policy = SAC_adv(config_update, config_arch)
        self.policy.build_network(verbose=True)
        print("Loading frozen weights of model at {} with load_dict {}".format(model_path, load_dict))
        self.policy.restore(None, model_path, load_dict=load_dict)
        print("-> Done")

    def get_action(self, state:np.ndarray, action:np.ndarray) -> np.ndarray:
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
        
        return action

    def get_q(self, state:np.ndarray, action:np.ndarray):
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