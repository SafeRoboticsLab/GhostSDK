# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A class for soft actor-critic being a best response to a fixed target.

This file implements a soft actor-critic (SAC) agent that is a best response
to a fixed target. Currently it only supports targets as neural-network-based
policies (actors).
"""

from typing import Optional, Union, Tuple, List, Dict
import os
import copy
import shutil
import warnings
import torch
import numpy as np
import wandb

from agent.base_training import BaseTraining
from agent.replay_memory import Batch
from agent.base_block import Actor
from simulators import BaseEnv
from utils.dstb import DummyPolicy
from utils.utils import combine_obsrv

from simulators.policy.random_policy import RandomPolicy
from simulators.vec_env.vec_env import VecEnvBase
from simulators.base_zs_env import BaseZeroSumEnv


class ISAACS(BaseTraining):
  leaderboard: np.ndarray

  def __init__(self, cfg_solver, cfg_arch, seed: int):
    super().__init__(cfg_solver, cfg_arch, seed)
    self.critic = self.critics['central']  # alias
    self.ctrl = self.actors['ctrl']  # alias
    self.dstb = self.actors['dstb']  # alias
    self.ctrl.obsrv_list = cfg_solver.obsrv_list.ctrl  # It should be None.
    self.dstb.obsrv_list = cfg_solver.obsrv_list.dstb  # It can be ['ctrl'] or None.

    # Checkpoints.
    self.save_top_k_ctrl = int(cfg_solver.save_top_k.ctrl)
    self.save_top_k_dstb = int(cfg_solver.save_top_k.dstb)
    # Always keeps the dummy dstb (no dstb) and has placeholder for the current.
    self.aux_metric = cfg_solver.eval.aux_metric
    self.leaderboard = np.full(
        shape=(self.save_top_k_ctrl + 1, self.save_top_k_dstb + 2, 1 + len(self.aux_metric)), dtype=float,
        fill_value=None
    )
    self.dummy_dstb_policy = DummyPolicy(id='dummy', action_dim=self.dstb.action_dim)

    self.ctrl_ckpts: List[int] = []
    self.dstb_ckpts: List[int] = []
    self.ctrl_eval = copy.deepcopy(self.ctrl)  # Loads checkpoint weights separately to prevent pollution.
    self.dstb_eval = copy.deepcopy(self.dstb)  # Loads checkpoint weights separately to prevent pollution.

    self.rnd_ctrl_policy = RandomPolicy(
        id='rnd_ctrl', action_range=np.array(cfg_solver.warmup_action_range.ctrl, dtype=np.float32), seed=seed
    )
    self.rnd_dstb_policy = RandomPolicy(
        id='rnd_dstb', action_range=np.array(cfg_solver.warmup_action_range.dstb, dtype=np.float32), seed=seed
    )

    self.dstb_sampler_list: List[Union[DummyPolicy, RandomPolicy,
                                       Actor]] = [self.rnd_dstb_policy for _ in range(self.num_envs)]
    self.softmax_rationality = float(cfg_solver.softmax_rationality)
    self.ctrl_update_ratio = int(cfg_solver.ctrl_update_ratio)

    # Copy ckpts from previous stages.
    if cfg_arch.actor_0.pretrained_path is not None:
      tmp_folder = os.path.join(self.model_folder, cfg_solver.actor_0.net_name)
      os.makedirs(tmp_folder, exist_ok=True)
      shutil.copyfile(
          cfg_arch.actor_0.pretrained_path, os.path.join(tmp_folder, f"{cfg_solver.actor_0.net_name}-0.pth")
      )
    if cfg_arch.actor_1.pretrained_path is not None:
      tmp_folder = os.path.join(self.model_folder, cfg_solver.actor_1.net_name)
      os.makedirs(tmp_folder, exist_ok=True)
      shutil.copyfile(
          cfg_arch.actor_1.pretrained_path, os.path.join(tmp_folder, f"{cfg_solver.actor_1.net_name}-0.pth")
      )
    if cfg_arch.critic_0.pretrained_path is not None:
      tmp_folder = os.path.join(self.model_folder, cfg_solver.critic_0.net_name)
      os.makedirs(tmp_folder, exist_ok=True)
      shutil.copyfile(
          cfg_arch.critic_0.pretrained_path, os.path.join(tmp_folder, f"{cfg_solver.critic_0.net_name}-0.pth")
      )

  @staticmethod
  def combine_action(ctrl_action: torch.Tensor, dstb_action: torch.Tensor) -> torch.Tensor:
    return torch.cat([ctrl_action, dstb_action], dim=-1)

  def get_dstb_sampler(self) -> Union[DummyPolicy, RandomPolicy, Actor]:
    choices = np.append(np.arange(len(self.dstb_ckpts)), -1)  # Dummy dstb.
    logit = np.mean(self.leaderboard[:len(self.ctrl_ckpts), choices, 0], axis=0)

    prob_un = np.exp(-self.softmax_rationality * logit)  # negative here since dstb minimizes.
    prob = prob_un / np.sum(prob_un)
    dstb_ckpt_idx = self.rng.choice(choices, p=prob)

    if dstb_ckpt_idx == -1:
      return self.dummy_dstb_policy
    else:
      dstb_sampler = copy.deepcopy(self.dstb_eval)
      dstb_sampler.restore(self.dstb_ckpts[dstb_ckpt_idx], self.model_folder, verbose=False)
      return dstb_sampler

  def sample(self, obsrv_all: torch.Tensor) -> List[Dict[str, np.ndarray]]:
    """Samples actions given the current observations.

    Args:
        obsrv_all (torch.Tensor): current observaions of all environments.

    Returns:
        List[Dict[str, np.ndarray]]: actions to execute.
    """
    obsrv_all = obsrv_all.float().to(self.device)

    # Gets control actions.
    if self.cnt_step < self.warmup_steps:  # Warms up with random actions.
      ctrl_action_all, _ = self.rnd_ctrl_policy.get_action(obsrv_all)
    else:
      with torch.no_grad():
        if self.ctrl.is_stochastic:
          ctrl_action_all, _ = self.ctrl.sample(obsrv_all, append=None, latent=None)
        else:
          ctrl_action_all = self.ctrl.net(obsrv_all, append=None, latent=None)
      ctrl_action_all = ctrl_action_all.cpu().numpy()  # (num_envs, ctrl_action_dim)

    action_all = []
    for i in range(self.num_envs):
      dstb_sampler = self.dstb_sampler_list[i]
      # Although we pass `agents_action` to the sampler, it is used only if dstb_sampler.obsrv_list contains 'ctrl'.
      # Also, `RandomPolicy` and `DummyPolicy` do not use `agents_action`.
      with torch.no_grad():
        if dstb_sampler.is_stochastic:
          assert not isinstance(dstb_sampler, DummyPolicy), "Dummy policy cannot be stochastic."
          dstb_action, _ = dstb_sampler.sample(
              obsrv_all[i], agents_action={"ctrl": ctrl_action_all[i]}, append=None, latent=None
          )  # (dstb_action_dim,)
        else:
          dstb_action, _ = dstb_sampler.get_action(
              obsrv_all[i], agents_action={"ctrl": ctrl_action_all[i]}, append=None, latent=None
          )  # (dstb_action_dim,)
      if isinstance(dstb_action, torch.Tensor):
        dstb_action = dstb_action.cpu().numpy()
      action_all.append({'ctrl': ctrl_action_all[i], 'dstb': dstb_action})

    return action_all

  def interact(
      self, rollout_env: Union[BaseZeroSumEnv, VecEnvBase], obsrv_all: torch.Tensor, action_all: List[Dict[str,
                                                                                                           np.ndarray]]
  ):
    # TODO: better compatiability with single rollout env.
    # * Here, we overload variables with outputs from a single env.
    if self.num_envs == 1:
      obsrv_nxt, r, done, info = rollout_env.step(action_all[0], cast_torch=True)
      obsrv_nxt_all = obsrv_nxt[None]
      r_all = np.array([r])
      done_all = np.array([done])
      info_all = np.array([info])
    else:
      obsrv_nxt_all, r_all, done_all, info_all = rollout_env.step(action_all)

    for env_idx, (done, info) in enumerate(zip(done_all, info_all)):
      # Stores the transition in memory. Note that `obsrv` and `action` are cpu tensors.
      action = {k: torch.FloatTensor(v[None]) for k, v in action_all[env_idx].items()}
      self.store_transition(
          obsrv_all[[env_idx]].cpu(), action, r_all[env_idx], obsrv_nxt_all[[env_idx]].cpu(), done, info
      )

      if done:
        if self.num_envs == 1:
          obsrv_nxt_all = rollout_env.reset(cast_torch=True)[None]
        else:
          obsrv_nxt_all[env_idx] = rollout_env.reset_one(index=env_idx)
        g_x = info['g_x']
        if g_x < 0:
          self.cnt_safety_violation += 1
        self.cnt_num_episode += 1
        self.dstb_sampler_list[env_idx] = self.get_dstb_sampler()

    # Updates records.
    self.violation_record.append(self.cnt_safety_violation)
    self.episode_record.append(self.cnt_num_episode)

    # Updates counter.
    self.cnt_step += self.num_envs
    self.cnt_opt_period += self.num_envs
    self.cnt_eval_period += self.num_envs
    return obsrv_nxt_all

  def update_one(self, batch: Batch, timer: int, update_ctrl: bool,
                 update_dstb: bool = True) -> Tuple[float, float, float, float, float, float, float]:
    """Updates the critic and actor networks with one batch.

    Args:
        batch (Batch): a batch of transitions.
    """
    ctrl_action = batch.action['ctrl']
    dstb_action = batch.action['dstb']

    # Updates the critic.
    self.critic.net.train()
    self.critic.target.train()
    self.ctrl.net.eval()
    self.dstb.net.eval()

    with torch.no_grad():
      ctrl_action_nxt, _ = self.ctrl.sample(batch.non_final_obsrv_nxt)
      dstb_action_nxt, _ = self.dstb.sample(
          batch.non_final_obsrv_nxt, agents_action={"ctrl": ctrl_action_nxt.cpu().numpy()}
      )
    action = self.combine_action(ctrl_action, dstb_action)
    action_nxt = self.combine_action(ctrl_action_nxt, dstb_action_nxt)

    q1, q2 = self.critic.net(batch.obsrv, action)  # Gets Q(s, a).
    q1_nxt, q2_nxt = self.critic.target(batch.non_final_obsrv_nxt, action_nxt)  # Gets Q(s', a').
    # TODO: what is the correct usage of entropy motives in zero-sum games?
    loss_q = self.critic.update(
        q1=q1, q2=q2, q1_nxt=q1_nxt, q2_nxt=q2_nxt, non_final_mask=batch.non_final_mask, reward=batch.reward,
        g_x=batch.info['g_x'], l_x=batch.info['l_x'], binary_cost=batch.info['binary_cost'], entropy_motives=0
    )

    # Updates the ctrl actor.
    if update_ctrl and timer % self.ctrl.update_period == 0:
      if self.cnt_step < self.warmup_steps:
        update_alpha = False
      else:
        update_alpha = True
      self.ctrl.net.train()
      self.dstb.net.eval()
      self.critic.net.eval()
      ctrl_action_sample, log_prob = self.ctrl.sample(obsrv=batch.obsrv)
      with torch.no_grad():
        if self.dstb.obsrv_list is None:
          dstb_action_aux = self.dstb.net(batch.obsrv)
        else:
          dstb_action_aux = self.dstb.net(batch.obsrv, action=ctrl_action_sample)
      action_sample = self.combine_action(ctrl_action_sample, dstb_action_aux)

      q1_sample, q2_sample = self.critic.net(batch.obsrv, action_sample)
      loss_ctrl, loss_ent_ctrl, loss_alpha_ctrl = self.ctrl.update(
          q1=q1_sample, q2=q2_sample, log_prob=log_prob, update_alpha=update_alpha
      )
    else:
      loss_ctrl = loss_ent_ctrl = loss_alpha_ctrl = 0.

    # Updates the dstb actor.
    if update_dstb and timer % self.dstb.update_period == 0:
      if self.cnt_step < self.warmup_steps:
        update_alpha = False
      else:
        update_alpha = True
      self.dstb.net.train()
      self.ctrl.net.eval()
      self.critic.net.eval()
      with torch.no_grad():
        ctrl_action_aux = self.ctrl.net(batch.obsrv)
      if self.dstb.obsrv_list is None:
        dstb_action_sample, log_prob = self.dstb.net.sample(obsrv=batch.obsrv)
      else:
        dstb_action_sample, log_prob = self.dstb.net.sample(obsrv=batch.obsrv, action=ctrl_action_aux)
      action_sample = self.combine_action(ctrl_action_aux, dstb_action_sample)

      q1_sample, q2_sample = self.critic.net(batch.obsrv, action_sample)
      loss_dstb, loss_ent_dstb, loss_alpha_dstb = self.dstb.update(
          q1=q1_sample, q2=q2_sample, log_prob=log_prob, update_alpha=update_alpha
      )
    else:
      loss_dstb = loss_ent_dstb = loss_alpha_dstb = 0.

    if timer % self.critic.update_target_period == 0:  # Updates the target networks.
      self.critic.update_target()

    self.critic.net.eval()
    self.ctrl.net.eval()
    self.dstb.net.eval()

    return loss_q, loss_ctrl, loss_ent_ctrl, loss_alpha_ctrl, loss_dstb, loss_ent_dstb, loss_alpha_dstb

  def update(self):
    if (self.cnt_step >= self.min_steps_b4_opt and self.cnt_opt_period >= self.opt_period):
      if self.cnt_dstb_updates == self.ctrl_update_ratio:
        update_ctrl = True
        self.cnt_dstb_updates = 0
        loss_ctrl_all = []
        loss_ent_ctrl_all = []
        loss_alpha_ctrl_all = []
      else:
        update_ctrl = False
      print(f"Updates at sample step {self.cnt_step}: ctrl ({update_ctrl}).")
      self.cnt_opt_period = 0
      loss_q_all = []
      loss_dstb_all = []
      loss_ent_dstb_all = []
      loss_alpha_dstb_all = []

      for timer in range(self.num_updates_per_opt):
        sample = True
        cnt = 0
        while sample:
          batch = self.sample_batch()
          sample = torch.logical_not(torch.any(batch.non_final_mask))
          cnt += 1
          if cnt >= 10:
            break
        if sample:
          warnings.warn("Cannot get a valid batch!!", UserWarning)
          continue

        loss_q, loss_ctrl, loss_ent_ctrl, loss_alpha_ctrl, loss_dstb, loss_ent_dstb, loss_alpha_dstb = self.update_one(
            batch, timer, update_ctrl=update_ctrl
        )
        loss_q_all.append(loss_q)
        if update_ctrl and timer % self.ctrl.update_period == 0:
          loss_ctrl_all.append(loss_ctrl)
          loss_ent_ctrl_all.append(loss_ent_ctrl)
          loss_alpha_ctrl_all.append(loss_alpha_ctrl)
        if timer % self.dstb.update_period == 0:
          loss_dstb_all.append(loss_dstb)
          loss_ent_dstb_all.append(loss_ent_dstb)
          loss_alpha_dstb_all.append(loss_alpha_dstb)

      loss_q_mean = np.array(loss_q_all).mean()
      loss_dstb_mean = np.array(loss_dstb_all).mean()
      loss_ent_dstb_mean = np.array(loss_ent_dstb_all).mean()
      loss_alpha_dstb_mean = np.array(loss_alpha_dstb_all).mean()
      if update_ctrl:
        loss_ctrl_mean = np.array(loss_ctrl_all).mean()
        loss_ent_ctrl_mean = np.array(loss_ent_ctrl_all).mean()
        loss_alpha_ctrl_mean = np.array(loss_alpha_ctrl_all).mean()
      else:
        loss_ctrl_mean = loss_ent_ctrl_mean = loss_alpha_ctrl_mean = None
      self.loss_record.append([
          loss_q_mean, loss_ctrl_mean, loss_ent_ctrl_mean, loss_alpha_ctrl_mean, loss_dstb_mean, loss_ent_dstb_mean,
          loss_alpha_dstb_mean
      ])

      if self.use_wandb:
        log_dict = {
            "loss/critic": loss_q_mean,
            "loss/dstb": loss_dstb_mean,
            "loss/entropy_dstb": loss_ent_dstb_mean,
            "loss/alpha_dstb": loss_alpha_dstb_mean,
            "metrics/cnt_safety_violation": self.cnt_safety_violation,
            "metrics/cnt_num_episode": self.cnt_num_episode,
            "hyper_parameters/alpha_ctrl": self.ctrl.alpha,
            "hyper_parameters/alpha_dstb": self.dstb.alpha,
            "hyper_parameters/gamma": self.critic.gamma,
        }
        if update_ctrl:
          log_dict['loss/ctrl'] = loss_ctrl_mean
          log_dict['loss/entropy_ctrl'] = loss_ent_ctrl_mean
          log_dict['loss/alpha_ctrl'] = loss_alpha_ctrl_mean
        wandb.log(log_dict, step=self.cnt_step, commit=False)

      self.cnt_dstb_updates += 1

  def update_ctrl_agent(
      self, env: BaseZeroSumEnv, rollout_env: Union[BaseZeroSumEnv, VecEnvBase], ctrl_ckpt_step: int
  ):
    """Updates the agent(s) in the environment with the latest actor.

    Args:
        env (BaseEnv): _description_
        rollout_env (Union[BaseEnv, VecEnvBase]): _description_
    """
    if ctrl_ckpt_step == self.cnt_step:
      self.ctrl_eval.update_policy(self.ctrl)
    else:
      self.ctrl_eval.restore(ctrl_ckpt_step, self.model_folder, verbose=False)

    env_policy: Actor = env.agent.policy
    env_policy.update_policy(self.ctrl_eval)

    if self.num_envs > 1:
      for agent in self.agent_copy_list:
        agent_policy: Actor = agent.policy
        agent_policy.update_policy(self.ctrl_eval)
      rollout_env.set_attr("agent", self.agent_copy_list, value_batch=True)
    else:
      rollout_env_policy: Actor = rollout_env.agent.policy
      rollout_env_policy.update_policy(self.ctrl_eval)

  def update_dstb_agent(self, dstb_ckpt_step: int):
    if dstb_ckpt_step == self.cnt_step:
      self.dstb_eval.update_policy(self.dstb)
    else:
      self.dstb_eval.restore(dstb_ckpt_step, self.model_folder, verbose=False)

  def update_hyper_param(self):
    self.ctrl.update_hyper_param()  # lr_pi, lr_alpha
    self.dstb.update_hyper_param()  # lr_pi, lr_alpha
    flag_rst_alpha = self.critic.update_hyper_param()  # lr_q, gamma
    if flag_rst_alpha:
      self.ctrl.reset_alpha()
      self.dstb.reset_alpha()

  def eval(
      self, env: BaseZeroSumEnv, rollout_env: Union[BaseZeroSumEnv, VecEnvBase], eval_callback, init_eval: bool = False
  ) -> bool:
    if self.cnt_eval_period >= self.eval_period or init_eval:
      print(f"\nChecks at sample step {self.cnt_step}:")
      self.cnt_eval_period = 0  # Resets counter.
      cur_ctrl_idx = len(self.ctrl_ckpts)
      cur_dstb_idx = len(self.dstb_ckpts)

      # * `adversary` cannot be under object function, which results in complaints about thread.lock object.
      # Evaluates the current dstb vs. all ctrl ckpts.
      self.update_dstb_agent(dstb_ckpt_step=self.cnt_step)
      for ctrl_idx, ctrl_ckpt_step in enumerate(self.ctrl_ckpts):
        self.update_ctrl_agent(env, rollout_env, ctrl_ckpt_step=ctrl_ckpt_step)
        eval_results: dict = eval_callback(
            env=env, rollout_env=rollout_env, value_fn=self.value, adversary=self.dstb_eval,
            fig_path=os.path.join(self.figure_folder, f"{ctrl_ckpt_step}_{self.cnt_step}.png")
        )
        self.update_leaderboard(eval_results, ctrl_idx, cur_dstb_idx)

      # Evaluates the current ctrl vs. all dstb ckpts
      self.update_ctrl_agent(env, rollout_env, ctrl_ckpt_step=self.cnt_step)
      for dstb_idx, dstb_ckpt_step in enumerate(self.dstb_ckpts):
        self.update_dstb_agent(dstb_ckpt_step=dstb_ckpt_step)
        eval_results: dict = eval_callback(
            env=env, rollout_env=rollout_env, value_fn=self.value, adversary=self.dstb_eval,
            fig_path=os.path.join(self.figure_folder, f"{self.cnt_step}_{dstb_ckpt_step}.png")
        )
        self.update_leaderboard(eval_results, cur_ctrl_idx, dstb_idx)

      # Evaluates the current ctrl vs. current dstb.
      self.update_dstb_agent(dstb_ckpt_step=self.cnt_step)
      eval_results: dict = eval_callback(
          env=env, rollout_env=rollout_env, value_fn=self.value, adversary=self.dstb_eval,
          fig_path=os.path.join(self.figure_folder, f"{self.cnt_step}_{self.cnt_step}.png")
      )
      self.update_leaderboard(eval_results, cur_ctrl_idx, cur_dstb_idx)

      # Evaluates the current ctrl vs. dummy dstb.
      eval_results: dict = eval_callback(
          env=env, rollout_env=rollout_env, value_fn=self.value, adversary=self.dummy_dstb_policy,
          fig_path=os.path.join(self.figure_folder, f"{self.cnt_step}_dummy.png")
      )
      self.update_leaderboard(eval_results, cur_ctrl_idx, -1)

      log_dict = {}
      log_dict[f'eval/{self.eval_metric}_ctrl'] = np.nanmean(self.leaderboard[cur_ctrl_idx, :, 0])
      log_dict[f'eval/{self.eval_metric}_dstb'] = np.nanmean(self.leaderboard[:, cur_dstb_idx, 0])
      for metric_idx, aux_metric_name in enumerate(self.aux_metric):
        log_dict[f'eval/{aux_metric_name}_ctrl'] = np.nanmean(self.leaderboard[cur_ctrl_idx, :, metric_idx + 1])
        log_dict[f'eval/{aux_metric_name}_dstb'] = np.nanmean(self.leaderboard[:, cur_dstb_idx, metric_idx + 1])
      # print(log_dict)
      self.eval_record.append(list(log_dict.values()))

      # Prunes the leaderboard.
      self.prune_leaderboard()

      if self.use_wandb:
        wandb.log(log_dict, step=self.cnt_step, commit=True)
      return True
    else:
      return False

  def update_leaderboard(self, eval_results: dict, ctrl_idx: int, dstb_idx: int):
    """Updates leaderboard.
    """
    self.leaderboard[ctrl_idx, dstb_idx, 0] = eval_results[self.eval_metric]
    for metric_idx, aux_metric_name in enumerate(self.aux_metric):
      self.leaderboard[ctrl_idx, dstb_idx, 1 + metric_idx] = eval_results[aux_metric_name]

  def prune_leaderboard(self):
    self.critic.save(self.cnt_step, self.model_folder)  # Always saves critic checkpoints.
    with np.printoptions(precision=3, suppress=False):
      print(self.leaderboard[..., 0])
    if len(self.ctrl_ckpts) == self.save_top_k_ctrl:
      ctrl_avg_metric = np.nanmean(self.leaderboard[..., 0], axis=1)
      ctrl_idx = np.argmin(ctrl_avg_metric)  # Removes the ctrl ckpt that has the minimum average metric.
      with np.printoptions(precision=3, suppress=False):
        print("ctrl results", ctrl_avg_metric)
      if ctrl_idx != self.save_top_k_ctrl:
        print(f'Saving current ctrl by removing {ctrl_idx}')
        self.ctrl.remove(self.ctrl_ckpts[ctrl_idx], self.model_folder)
        self.ctrl_ckpts[ctrl_idx] = self.cnt_step
        self.leaderboard[ctrl_idx] = self.leaderboard[-1]
        self.ctrl.save(self.cnt_step, self.model_folder)
    else:
      self.ctrl_ckpts.append(self.cnt_step)
      self.ctrl.save(self.cnt_step, self.model_folder)

    if len(self.dstb_ckpts) == self.save_top_k_dstb:
      dstb_avg_metric = np.nanmean(self.leaderboard[:, :-1, 0], axis=0)
      dstb_idx = np.argmax(dstb_avg_metric)  # Removes the dstb ckpt that has the maximum average metric.
      with np.printoptions(precision=3, suppress=False):
        print("dstb results", dstb_avg_metric)
      if dstb_idx != self.save_top_k_dstb:
        print(f'Saving current dstb by removin {dstb_idx}')
        self.dstb.remove(self.dstb_ckpts[dstb_idx], self.model_folder)
        self.dstb_ckpts[dstb_idx] = self.cnt_step
        self.leaderboard[:, dstb_idx] = self.leaderboard[:, -2]
        self.dstb.save(self.cnt_step, self.model_folder)
    else:
      self.dstb_ckpts.append(self.cnt_step)
      self.dstb.save(self.cnt_step, self.model_folder)
    print()

  def save(self, max_model: Optional[int] = None):
    self.ctrl.save(self.cnt_step, self.model_folder, max_model)
    self.dstb.save(self.cnt_step, self.model_folder, max_model)
    self.critic.save(self.cnt_step, self.model_folder, max_model)

  def value(self, obsrv: np.ndarray, append: Optional[np.ndarray] = None) -> np.ndarray:
    obsrv_tensor = torch.FloatTensor(obsrv).to(self.device)
    with torch.no_grad():
      ctrl_action = self.ctrl.net(obsrv_tensor, append=append)
      if self.dstb.obsrv_list is None:
        dstb_obsrv = obsrv_tensor
      else:
        dstb_obsrv = combine_obsrv(obsrv_tensor, ctrl_action)
      dstb_action = self.dstb.net(dstb_obsrv, append=append)
      action = self.combine_action(ctrl_action, dstb_action)
    return self.critic.value(obsrv_tensor, action, append=append)

  def init_learn(self, env: BaseEnv) -> Union[BaseEnv, VecEnvBase]:
    self.cnt_dstb_updates = 0
    return super().init_learn(env)
