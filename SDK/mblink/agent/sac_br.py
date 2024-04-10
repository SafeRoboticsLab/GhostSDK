# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A class for soft actor-critic being a best response to a fixed target.

This file implements a soft actor-critic (SAC) agent that is a best response to a fixed target. We use "ego" for the
policy trained by SAC and "adv" for the fixed target policy.
"""

from typing import Optional, Union, Tuple, List, Dict
import os
import warnings
import torch
import numpy as np
from queue import PriorityQueue
import wandb

from agent.base_training import BaseTraining
from agent.replay_memory import Batch
from agent.base_block import Actor
from utils.utils import combine_obsrv

from simulators.policy.base_policy import BasePolicy
from simulators.policy.random_policy import RandomPolicy
from simulators.vec_env.vec_env import VecEnvBase
from simulators.base_zs_env import BaseZeroSumEnv


class SACBestResponse(BaseTraining):

  def __init__(
      self, cfg_solver, cfg_arch, seed: int, ctrl_policy: Optional[BasePolicy] = None,
      dstb_policy: Optional[BasePolicy] = None
  ):
    super().__init__(cfg_solver, cfg_arch, seed)

    # Checkpoints.
    self.save_top_k = cfg_solver.save_top_k
    self.leaderboard = PriorityQueue()

    self.critic = self.critics['central']  # alias
    self.actor = list(self.actors.values())[0]  # alias

    self.rnd_ego_policy = RandomPolicy(
        id='rnd_ego', action_range=np.array(cfg_solver.warmup_action_range, dtype=np.float32), seed=seed
    )

    if self.actor.net_name == 'ctrl':
      assert dstb_policy is not None
      self.adv_policy = dstb_policy
      self.fix_ctrl = False
      self.fix_dstb = True
      self.actor.obsrv_list = cfg_solver.obsrv_list.ctrl
      self.ego_info_adv = cfg_solver.obsrv_list.ctrl is not None
      self.adv_info_adv = cfg_solver.obsrv_list.dstb is not None
    elif self.actor.net_name == 'dstb':
      assert ctrl_policy is not None
      self.adv_policy = ctrl_policy
      self.fix_ctrl = True
      self.fix_dstb = False
      self.actor.obsrv_list = cfg_solver.obsrv_list.dstb
      self.ego_info_adv = cfg_solver.obsrv_list.dstb is not None
      self.adv_info_adv = cfg_solver.obsrv_list.ctrl is not None
    else:
      raise ValueError('Unsupported actor name: {}'.format(self.actor.net_name))
    if self.ego_info_adv and self.adv_info_adv:
      raise ValueError("Only one policy can have information advantage.")

    self.eval_max = bool(cfg_solver.eval.to_max)

  @property
  def ego_name(self) -> str:
    return self.actor.net_name

  @property
  def adv_name(self) -> str:
    if self.fix_ctrl:
      return 'ctrl'
    else:  # `fix_dstb` is True.
      return 'dstb'

  def combine_action(self, ego_action: torch.Tensor, adv_action: torch.Tensor) -> torch.Tensor:
    if self.fix_ctrl:  # `adv_name`: 'ctrl' & `ego_name`: 'dstb'
      return torch.cat([adv_action, ego_action], dim=-1)
    else:
      return torch.cat([ego_action, adv_action], dim=-1)

  def sample_ego_action(
      self, ego_obsrv_all: torch.Tensor, agents_action: Optional[Dict[str, np.ndarray]] = None
  ) -> np.ndarray:
    if self.cnt_step < self.warmup_steps:  # Warms up with random actions.
      ego_action_all = self.rnd_ego_policy.get_action(ego_obsrv_all)[0]
    else:
      with torch.no_grad():
        if self.actor.is_stochastic:
          ego_action_all = self.actor.sample(ego_obsrv_all, agents_action=agents_action)[0]
        else:
          ego_action_all = self.actor.get_action(ego_obsrv_all, agents_action=agents_action)[0]

    if isinstance(ego_action_all, torch.Tensor):
      ego_action_all = ego_action_all.cpu().numpy()  # (num_envs, ego_action_dim)
    return ego_action_all

  def sample(self, obsrv_all: torch.Tensor) -> List[Dict[str, np.ndarray]]:
    """Samples actions given the current observations.

    Args:
        obsrv_all (torch.Tensor): current observaions of all environments.

    Returns:
        List[Dict[str, np.ndarray]]: actions to execute.
    """
    obsrv_all = obsrv_all.float().to(self.device)

    if self.ego_info_adv:
      adv_action_all = self.adv_policy.get_action(obsrv_all)[0]
      ego_action_all = self.sample_ego_action(obsrv_all, agents_action={self.adv_name: adv_action_all})
    elif self.adv_info_adv:
      ego_action_all = self.sample_ego_action(obsrv_all)
      adv_action_all = self.adv_policy.get_action(obsrv_all, agents_action={self.ego_name: ego_action_all})[0]
    else:
      ego_action_all = self.sample_ego_action(obsrv_all)
      adv_action_all = self.adv_policy.get_action(obsrv_all)[0]

    action_all = [{self.ego_name: ego_action_all[i], self.adv_name: adv_action_all[i]} for i in range(self.num_envs)]
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

    # Updates records.
    self.violation_record.append(self.cnt_safety_violation)
    self.episode_record.append(self.cnt_num_episode)

    # Updates counter.
    self.cnt_step += self.num_envs
    self.cnt_opt_period += self.num_envs
    self.cnt_eval_period += self.num_envs
    return obsrv_nxt_all

  def update_one(self, batch: Batch, timer: int) -> Tuple[float, float, float, float]:
    """Updates the critic and actor networks with one batch.

    Args:
        batch (Batch): a batch of transitions.
    """
    ego_action = batch.action[self.actor.net_name]
    adv_action = batch.action[self.adv_name]

    # Updates the critic.
    self.critic.net.train()
    self.critic.target.train()
    self.actor.net.eval()
    with torch.no_grad():
      if self.ego_info_adv:
        adv_action_nxt, _ = self.adv_policy.get_action(batch.non_final_obsrv_nxt)
        ego_action_nxt, log_prob_nxt = self.actor.sample(
            batch.non_final_obsrv_nxt, agents_action={self.adv_name: adv_action_nxt}
        )
        adv_action_nxt = torch.FloatTensor(adv_action_nxt).to(self.device)
      elif self.adv_info_adv:
        ego_action_nxt, log_prob_nxt = self.actor.sample(batch.non_final_obsrv_nxt)
        adv_action_nxt, _ = self.adv_policy.get_action(
            batch.non_final_obsrv_nxt, agents_action={self.ego_name: ego_action_nxt.cpu().numpy()}
        )
        adv_action_nxt = torch.FloatTensor(adv_action_nxt).to(self.device)
      else:
        ego_action_nxt, log_prob_nxt = self.actor.sample(batch.non_final_obsrv_nxt)
        adv_action_nxt, _ = self.adv_policy.get_action(batch.non_final_obsrv_nxt)
        adv_action_nxt = torch.FloatTensor(adv_action_nxt).to(self.device)
      entropy_motives = self.actor.alpha * log_prob_nxt.view(-1)

    action = self.combine_action(ego_action, adv_action)
    action_nxt = self.combine_action(ego_action_nxt, adv_action_nxt)

    q1, q2 = self.critic.net(batch.obsrv, action)  # Gets Q(s, a).
    q1_nxt, q2_nxt = self.critic.target(batch.non_final_obsrv_nxt, action_nxt)  # Gets Q(s', a').
    loss_q = self.critic.update(
        q1=q1, q2=q2, q1_nxt=q1_nxt, q2_nxt=q2_nxt, non_final_mask=batch.non_final_mask, reward=batch.reward,
        g_x=batch.info['g_x'], l_x=batch.info['l_x'], binary_cost=batch.info['binary_cost'],
        entropy_motives=entropy_motives
    )

    if timer % self.actor.update_period == 0:  # Updates the actor.
      if self.cnt_step < self.warmup_steps:
        update_alpha = False
      else:
        update_alpha = True
      self.actor.net.train()
      self.critic.net.eval()

      if self.ego_info_adv:
        adv_action_sample, _ = self.adv_policy.get_action(batch.obsrv)
        ego_action_sample, log_prob = self.actor.net.sample(batch.obsrv, action=adv_action_sample)
        adv_action_sample = torch.FloatTensor(adv_action_sample).to(self.device)
      elif self.adv_info_adv:
        ego_action_sample, log_prob = self.actor.net.sample(batch.obsrv)
        adv_action_sample, _ = self.adv_policy.get_action(
            batch.obsrv, agents_action={self.ego_name: ego_action_sample.detach().cpu().numpy()}
        )
        adv_action_sample = torch.FloatTensor(adv_action_sample).to(self.device)
      else:
        ego_action_sample, log_prob = self.actor.net.sample(obsrv=batch.obsrv)
        adv_action_sample, _ = self.adv_policy.get_action(batch.obsrv)
        adv_action_sample = torch.FloatTensor(adv_action_sample).to(self.device)
      action_sample = self.combine_action(ego_action_sample, adv_action_sample)

      q1_sample, q2_sample = self.critic.net(batch.obsrv, action_sample)
      loss_pi, loss_ent, loss_alpha = self.actor.update(
          q1=q1_sample, q2=q2_sample, log_prob=log_prob, update_alpha=update_alpha
      )
    else:
      loss_pi = loss_ent = loss_alpha = 0.

    if timer % self.critic.update_target_period == 0:  # Updates the target networks.
      self.critic.update_target()

    self.critic.net.eval()
    self.actor.net.eval()
    return loss_q, loss_pi, loss_ent, loss_alpha

  def update(self):
    if (self.cnt_step >= self.min_steps_b4_opt and self.cnt_opt_period >= self.opt_period):
      print(f"Updates at sample step {self.cnt_step}")
      self.cnt_opt_period = 0
      loss_q_all = []
      loss_pi_all = []
      loss_ent_all = []
      loss_alpha_all = []

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

        loss_q, loss_pi, loss_ent, loss_alpha = self.update_one(batch, timer)
        loss_q_all.append(loss_q)
        if timer % self.actor.update_period == 0:
          loss_pi_all.append(loss_pi)
          loss_ent_all.append(loss_ent)
          loss_alpha_all.append(loss_alpha)

      loss_q_mean = np.array(loss_q_all).mean()
      loss_pi_mean = np.array(loss_pi_all).mean()
      loss_ent_mean = np.array(loss_ent_all).mean()
      loss_alpha_mean = np.array(loss_alpha_all).mean()
      self.loss_record.append([loss_q_mean, loss_pi_mean, loss_ent_mean, loss_alpha_mean])

      if self.use_wandb:
        log_dict = {
            "loss/critic": loss_q_mean,
            f"loss/policy_{self.ego_name}": loss_pi_mean,
            f"loss/entropy_{self.ego_name}": loss_ent_mean,
            f"loss/alpha_{self.ego_name}": loss_alpha_mean,
            "metrics/cnt_safety_violation": self.cnt_safety_violation,
            "metrics/cnt_num_episode": self.cnt_num_episode,
            "hyper_parameters/alpha": self.actor.alpha,
            "hyper_parameters/gamma": self.critic.gamma,
        }
        wandb.log(log_dict, step=self.cnt_step, commit=False)

  def update_eval_agent(self, env: BaseZeroSumEnv, rollout_env: Union[BaseZeroSumEnv, VecEnvBase]):
    """Updates the agent(s) in the environment with the latest actor.

    Args:
        env (BaseEnv): _description_
        rollout_env (Union[BaseEnv, VecEnvBase]): _description_
    """
    if self.ego_name == 'ctrl':
      env_policy: Actor = env.agent.policy
      env_policy.update_policy(self.actor)

      if self.num_envs > 1:
        for agent in self.agent_copy_list:
          agent_policy: Actor = agent.policy
          agent_policy.update_policy(self.actor)
        rollout_env.set_attr("agent", self.agent_copy_list, value_batch=True)
      else:
        rollout_env_policy: Actor = rollout_env.agent.policy
        rollout_env_policy.update_policy(self.actor)

  def update_hyper_param(self):
    self.actor.update_hyper_param()  # lr_pi, lr_alpha
    flag_rst_alpha = self.critic.update_hyper_param()  # lr_q, gamma
    if flag_rst_alpha:
      self.actor.reset_alpha()

  def eval(
      self, env: BaseZeroSumEnv, rollout_env: Union[BaseZeroSumEnv, VecEnvBase], eval_callback, init_eval: bool = False
  ) -> bool:
    if self.cnt_eval_period >= self.eval_period or init_eval:
      print(f"Checks at sample step {self.cnt_step}:")
      self.update_eval_agent(env, rollout_env)
      self.cnt_eval_period = 0  # Resets counter.

      # * This adversary cannot be under object function, which results in complaints about thread.lock object.
      adversary = self.adv_policy if self.ego_name == 'ctrl' else self.actor
      eval_results: dict = eval_callback(
          env=env, rollout_env=rollout_env, value_fn=self.value, adversary=adversary,
          fig_path=os.path.join(self.figure_folder, f"{self.cnt_step}.png")
      )
      self.eval_record.append(list(eval_results.values()))
      if not init_eval:
        self.update_leaderboard(eval_results)
      if self.use_wandb:
        log_dict = {f"eval/{key}": value for key, value in eval_results.items()}
        wandb.log(log_dict, step=self.cnt_step, commit=True)
      return True
    else:
      return False

  def update_leaderboard(self, eval_results: dict):
    """Updates leaderboard, saves checkpoints, and removes outdated checkpoints.

    Args:
        eval_results (dict): evaluation results.
    """
    metric = eval_results[self.eval_metric] if self.eval_max else -eval_results[self.eval_metric]
    save_current = False
    if self.cnt_step >= self.min_steps_b4_opt:
      if self.leaderboard.qsize() < self.save_top_k:
        self.leaderboard.put((metric, self.cnt_step))
        save_current = True
      elif metric > self.leaderboard.queue[0][0]:  # overwrite
        # Remove the one has the minimum metric.
        _, step_remove = self.leaderboard.get()
        self.actor.remove(step_remove, self.model_folder)
        self.critic.remove(step_remove, self.model_folder)
        self.leaderboard.put((metric, self.cnt_step))
        save_current = True

      if save_current:
        print('Saving current model...')
        self.save(self.max_model)
        print(self.leaderboard.queue)

  def save(self, max_model: Optional[int] = None):
    self.actor.save(self.cnt_step, self.model_folder, max_model)
    self.critic.save(self.cnt_step, self.model_folder, max_model)

  def value(self, obsrv: np.ndarray, append: Optional[np.ndarray] = None) -> np.ndarray:
    obsrv_tensor = torch.FloatTensor(obsrv).to(self.device)
    with torch.no_grad():
      if self.ego_info_adv:
        adv_action, _ = self.adv_policy.get_action(obsrv, append=append)
        ego_action = self.actor.net(obsrv_tensor, action=adv_action, append=append)
      elif self.adv_info_adv:
        ego_action = self.actor.net(obsrv_tensor, append=append)
        adv_action, _ = self.adv_policy.get_action(obsrv, agents_action={self.ego_name: ego_action}, append=append)
      else:
        ego_action = self.actor.net(obsrv_tensor, append=append)
        adv_action, _ = self.adv_policy.get_action(obsrv, append=append)

      adv_action = torch.FloatTensor(adv_action).to(self.device)
      action = self.combine_action(ego_action, adv_action)
    return self.critic.value(obsrv_tensor, action, append=append)
