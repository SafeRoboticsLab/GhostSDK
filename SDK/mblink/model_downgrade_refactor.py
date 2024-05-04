import os
import argparse
from omegaconf import OmegaConf
from agent import ISAACS
from utils.utils import get_model_index
import torch

from agent.base_block import Actor
from agent.sac import SAC
from agent.sac_br import SACBestResponse


def main(config_file):
    config_file = args.config_file
    cfg = OmegaConf.load(config_file)
    if cfg.solver.num_actors == 2 and cfg.agent.dyn == "SpiritPybullet":
        solver = ISAACS(cfg.solver, cfg.arch, cfg.environment.seed)
        dstb_step, model_path = get_model_index(cfg.solver.out_folder,
                                                cfg.eval.model_type[1],
                                                cfg.eval.step[1],
                                                type="dstb",
                                                autocutoff=0.9)

        ctrl_step, model_path = get_model_index(cfg.solver.out_folder,
                                                cfg.eval.model_type[0],
                                                cfg.eval.step[0],
                                                type="ctrl",
                                                autocutoff=0.9)

        solver.ctrl.restore(ctrl_step, model_path)
        solver.dstb.restore(dstb_step, model_path)
        solver.critic.restore(ctrl_step, model_path)

        # RESAVE THE MODEL FOR LOWER TORCH VERSION - TO RUN ON SPIRIT
        ## _use_new_zipfile_serialization = False
        torch.save(solver.ctrl.net.state_dict(),
                   os.path.join(
                       model_path, "ctrl",
                       "ctrl-{}".format(cfg.eval.step[0]) + "-downgrade.pth"),
                   _use_new_zipfile_serialization=False)
        torch.save(solver.dstb.net.state_dict(),
                   os.path.join(
                       model_path, "dstb",
                       "dstb-{}".format(cfg.eval.step[1]) + "-downgrade.pth"),
                   _use_new_zipfile_serialization=False)
        torch.save(
            solver.critic.net.state_dict(),
            os.path.join(
                model_path, "central",
                "central-{}".format(cfg.eval.step[0]) + "-downgrade.pth"),
            _use_new_zipfile_serialization=False)
    elif cfg.solver.num_actors == 1 and cfg.agent.dyn == "SpiritPybullet":
        # SAC for ctrl
        solver = SAC(cfg.solver, cfg.arch, cfg.environment.seed)
        ctrl_step, model_path = get_model_index(cfg.solver.out_folder,
                                                cfg.eval.model_type,
                                                cfg.eval.step,
                                                type="ctrl",
                                                autocutoff=0.9)
        solver.actor.restore(ctrl_step, model_path)
        solver.critic.restore(ctrl_step, model_path)

        torch.save(solver.actor.net.state_dict(),
                   os.path.join(
                       model_path, "ctrl",
                       "ctrl-{}".format(cfg.eval.step) + "-downgrade.pth"),
                   _use_new_zipfile_serialization=False)
        torch.save(solver.critic.net.state_dict(),
                   os.path.join(
                       model_path, "central",
                       "central-{}".format(cfg.eval.step) + "-downgrade.pth"),
                   _use_new_zipfile_serialization=False)
    elif cfg.solver.num_actors == 1 and cfg.agent.dyn == "SpiritDstbPybullet":
        # SAC for dstb
        ctrl_policy = Actor(cfg=cfg.solver.ctrl,
                            cfg_arch=cfg.arch.ctrl,
                            verbose=False,
                            device=cfg.solver.device,
                            obsrv_list=cfg.agent.obsrv_list.ctrl)
        ctrl_policy.restore(step=cfg.solver.ctrl.step,
                            model_folder=cfg.solver.ctrl.model_folder)
        solver = SACBestResponse(cfg.solver,
                                 cfg.arch,
                                 cfg.environment.seed,
                                 ctrl_policy=ctrl_policy)
        dstb_step, model_path = get_model_index(cfg.solver.out_folder,
                                                cfg.eval.model_type,
                                                cfg.eval.step,
                                                type="dstb",
                                                autocutoff=0.9)

        solver.actor.restore(dstb_step, model_path)
        solver.critic.restore(dstb_step, model_path)

        torch.save(solver.actor.net.state_dict(),
                   os.path.join(
                       model_path, "dstb",
                       "dstb-{}".format(cfg.eval.step) + "-downgrade.pth"),
                   _use_new_zipfile_serialization=False)
        torch.save(solver.critic.net.state_dict(),
                   os.path.join(
                       model_path, "central",
                       "central-{}".format(cfg.eval.step) + "-downgrade.pth"),
                   _use_new_zipfile_serialization=False)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf",
                        "--config_file",
                        help="config file path",
                        type=str)
    args = parser.parse_args()
    main(args)
