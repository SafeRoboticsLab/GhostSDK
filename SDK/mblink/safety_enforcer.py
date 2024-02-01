from typing import Optional
import numpy as np
import os

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
        elif version == 2.1:
            # old binary
            training_dir = "train_result/spirit_isaacs_binary_f5_softmax_newStateDef_pretrained/spirit_isaacs_binary_f5_softmax_newStateDef_pretrained_05"
            load_dict = {
                "ctrl": 7_580_000,
                "dstb": 8_000_001
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
        if state is not None and action is not None:
            if self.version >= 3:
                state = np.concatenate((state[2:8], state[9:]), axis=0)
            s_dstb = np.concatenate((state, action), axis=0)
            dstb = self.policy.dstb(s_dstb)

            critic_q = max(
                self.policy.adv_critic(
                    state, action, dstb
                )
            )
            
            return critic_q.reshape(-1)[0]
        else:
            return self.prev_q

    def target_margin(self, state):
        """ (36) and 33D state
            (x, y), z, 
            x_dot, y_dot, z_dot,
            roll, pitch, (yaw)
            w_x, w_y, w_z,
            joint_pos x 12,
            joint_vel x 12
        """
        # this is not the correct target margin, missing corner pos and toe pos, replacing corner pos with height, assuming that toes always touch ground
        # l(x) < 0 --> x \in T
        if self.version >= 3: 
            # change from 36D to 33D (ignore x, y, yaw: 0, 1, 8 index)
            if len(state) == 36:
                state = np.concatenate((state[2:8], state[9:]), axis=0)
            
            spirit_joint_pos = state[9:21]
            return {
                "height": state[0] - 0.4,
                "roll": abs(state[4]) - 0.20,
                "pitch": abs(state[5]) - 0.20
                # "w_x": abs(state[6]) - 0.17444,
                # "w_y": abs(state[7]) - 0.17444,
                # "w_z": abs(state[8]) - 0.17444,
                # "x_dot": abs(state[1]) - 0.2,
                # "y_dot": abs(state[2]) - 0.2,
                # "z_dot": abs(state[3]) - 0.2
            }
        else:
            if len(state) == 33:
                print("ERROR: asking for 36D state when there is only 33D state")
                return {"": np.inf}
            
            spirit_joint_pos = state[12:24]
            return {
                "height": state[2] - 0.4,
                "w_x": abs(state[9]) - 0.17444,
                "w_y": abs(state[10]) - 0.17444,
                "w_z": abs(state[11]) - 0.17444,
                "x_dot": abs(state[3]) - 0.2,
                "y_dot": abs(state[4]) - 0.2,
                "z_dot": abs(state[5]) - 0.2
            }
    
    def get_safety_action(self, state, target=True, threshold=0.0):
        # stable_stance = np.array([
        #     0.3, 0.75, 1.45,
        #     0.3, 0.75, 1.45,
        #     -0.3, 0.75, 1.45,
        #     -0.3, 0.75, 1.45
        # ])
        stable_stance = np.array([
            0.1, 0.75, 1.45, 
            0.4, 0.6, 1.9, 
            -0.1, 0.75, 1.45, 
            -0.4, 0.6, 1.9
        ])

        if not target:
            return self.policy.ctrl(state)
        else:
            # switch between fallback and target stable stance, depending on the current state
            margin = self.target_margin(state)
            lx = max(margin.values())
            # print(margin, "vx", abs(state[1]) - 0.2, "vy", abs(state[2]) - 0.2, "vz", abs(state[3]) - 0.2)
            if len(state) == 33:
                spirit_joint_pos = state[9:21]
            elif len(state) == 36:
                spirit_joint_pos = state[12:24]
            else:
                raise ValueError
            if lx <= threshold: # account for sensor noise
                # in target set, just output stable stance
                #! TODO: enforce stable stance instead of just outputting zero changes to the current stance
                return np.clip(stable_stance - spirit_joint_pos, -np.ones(12)*0.1, np.ones(12)*0.1)
            else:
                return self.policy.ctrl(state)
    
    def get_shielding_status(self):
        return self.is_shielded
