import sys
import os

sys.path.append(os.getcwd())

from types import SimpleNamespace
import argparse
import numpy as np
import torch
import math
from RARL.sac_adv import SAC_adv
from RARL.sac_mini import SAC_mini

from config.utils import load_config

parent_dir = os.getcwd()
# training_dir = "train_result/spirit_isaacs_avoidonly_f5_newStateDef/spirit_isaacs_avoidonly_f5_newStateDef_05"
# training_dir = "train_result/spirit_isaacs_reachavoid_f5_newStateDef_pretrained/spirit_isaacs_reachavoid_f5_newStateDef_pretrained_05"
# training_dir = "train_result/spirit_isaacs_reachavoid_f5_pretrained_newStateDef2/spirit_isaacs_reachavoid_f5_pretrained_newStateDef2_05"
# training_dir = "train_result/spirit_isaacs_reachavoid_f5_pretrained_newStateDef2_mirror/spirit_isaacs_reachavoid_f5_pretrained_newStateDef2_mirror_05"
# training_dir = "train_result/spirit_naive_reachavoid_f0_failure_newStateDef2/spirit_naive_reachavoid_f0_failure_newStateDef2_00"
# training_dir = "train_result/spirit_naive_binary_f5_newStateDef/spirit_naive_binary_f5_newStateDef_05"
training_dir = "train_result/spirit_isaacs_binary_f5_softmax_newStateDef_pretrained/spirit_isaacs_binary_f5_softmax_newStateDef_pretrained_05"
model_path = os.path.join(parent_dir, training_dir, "model")
model_config_path = os.path.join(parent_dir, training_dir, "config.yaml")

config_file = os.path.join(parent_dir, model_config_path)

if not os.path.exists(config_file):
    raise ValueError("Cannot find config file for the model, terminated")

config = load_config(config_file)
config_arch = config['arch']
config_update = config['update']
config_env = config['environment']

# load_dict = {
#     "ctrl": 3_920_000,
#     "dstb": 5_480_000
# }

# load_dict = {
#     "ctrl": 4_100_000,
#     "dstb": 7_520_000
# }

# load_dict = {
#     "ctrl": 5_900_000,
#     "dstb": 8_000_001
# }

# load_dict = {
#     "ctrl": 4_700_000,
#     "dstb": 8_000_001
# }

# load_dict = {
#     "ctrl": 4_400_000
# }

# load_dict = {
#     "ctrl": 1_100_000
# }

load_dict = {"ctrl": 7_580_000, "dstb": 8_000_001}

if config_env.NUM_AGENTS == 2:
    policy = SAC_adv(config_update, config_arch)
    policy.build_network(verbose=True)
    policy.restore(None, model_path, load_dict=load_dict)

    # RESAVE THE MODEL FOR LOWER TORCH VERSION - TO RUN ON SPIRIT
    ## _use_new_zipfile_serialization = False
    torch.save(policy.ctrl.state_dict(),
               os.path.join(
                   model_path, "ctrl",
                   "actor-{}".format(load_dict["ctrl"]) + "-downgrade.pth"),
               _use_new_zipfile_serialization=False)
    torch.save(policy.dstb.state_dict(),
               os.path.join(
                   model_path, "dstb",
                   "actor-{}".format(load_dict["dstb"]) + "-downgrade.pth"),
               _use_new_zipfile_serialization=False)
    torch.save(policy.adv_critic.state_dict(),
               os.path.join(
                   model_path, "adv_critic",
                   "critic-{}".format(load_dict["ctrl"]) + "-downgrade.pth"),
               _use_new_zipfile_serialization=False)
    torch.save(policy.mean_critic.state_dict(),
               os.path.join(
                   model_path, "mean_critic",
                   "critic-{}".format(load_dict["ctrl"]) + "-downgrade.pth"),
               _use_new_zipfile_serialization=False)
elif config_env.NUM_AGENTS == 1:
    policy = SAC_mini(config_update, config_arch)
    policy.build_network(verbose=False)
    policy.restore(load_dict["ctrl"], model_path)
    torch.save(policy.critic.state_dict(),
               os.path.join(
                   model_path, "critic",
                   "critic-{}".format(load_dict["ctrl"]) + "-downgrade.pth"),
               _use_new_zipfile_serialization=False)
    torch.save(policy.actor.state_dict(),
               os.path.join(
                   model_path, "actor",
                   "actor-{}".format(load_dict["ctrl"]) + "-downgrade.pth"),
               _use_new_zipfile_serialization=False)
