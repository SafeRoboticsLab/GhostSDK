# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""Classes for basic building blocks in soft actor-critic (SAC).

This file implements `Actor` and `Critic` building blocks. Each block has a
`net` attribute, which is a `torch.Module`. `BaseBlock` implements basic
operators, e.g., build_optimizer, save, restore, and, remove, but requires its
children to implement `build_network()` and `update()`
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, Dict, List
import time
import os
import copy
import numpy as np
import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR
from torch.nn.functional import mse_loss

from agent.model import GaussianPolicy, TwinnedQNetwork
from agent.scheduler import StepLRMargin
from utils.train import soft_update, save_model, get_bellman_update
from simulators.policy.base_policy import BasePolicy


def build_network(cfg, cfg_arch, device: torch.device,
                  verbose: bool = True) -> Tuple[Dict[str, Critic], Dict[str, Actor]]:
  critics: Dict[str, Critic] = {}
  actors: Dict[str, Actor] = {}
  for idx in range(cfg.num_critics):
    cfg_critic = getattr(cfg, f"critic_{idx}")
    cfg_arch_critic = getattr(cfg_arch, f"critic_{idx}")
    critic = Critic(cfg=cfg_critic, cfg_arch=cfg_arch_critic, verbose=verbose, device=device)
    critics[cfg_critic.net_name] = critic
  assert "central" in critics, "Must have a central critic."

  for idx in range(cfg.num_actors):
    cfg_actor = getattr(cfg, f"actor_{idx}")
    cfg_arch_actor = getattr(cfg_arch, f"actor_{idx}")
    actor = Actor(cfg=cfg_actor, cfg_arch=cfg_arch_actor, verbose=verbose, device=device)
    actors[cfg_actor.net_name] = actor
  return critics, actors


class BaseBlock(ABC):
  net: torch.nn.Module

  def __init__(self, cfg, device: torch.device) -> None:
    self.eval = cfg.eval
    self.device = device
    self.net_name: str = cfg.net_name

  @abstractmethod
  def build_network(self, verbose: bool = True):
    raise NotImplementedError

  def build_optimizer(self, cfg):
    if cfg.opt_type == "AdamW":
      self.opt_cls = AdamW
    elif cfg.opt_type == "Adam":
      self.opt_cls = Adam
    else:
      raise ValueError("Not supported optimizer type!")

    # Learning Rate
    self.lr_schedule: bool = cfg.lr_schedule
    if self.lr_schedule:
      self.lr_period = int(cfg.lr_period)
      self.lr_decay = float(cfg.lr_decay)
      self.lr_end = float(cfg.lr_end)
    self.lr = float(cfg.lr)

    # Builds the optimizer.
    self.optimizer = self.opt_cls(self.net.parameters(), lr=self.lr, weight_decay=0.01)
    if self.lr_schedule:
      self.scheduler = StepLR(self.optimizer, step_size=self.lr_period, gamma=self.lr_decay)

  def update_hyper_param(self):
    if not self.eval:
      if self.lr_schedule:
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        if lr <= self.lr_end:
          for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr_end
        else:
          self.scheduler.step()

  def save(self, step: int, model_folder: str, max_model: Optional[int] = None, verbose: bool = True) -> str:
    path = os.path.join(model_folder, self.net_name)
    save_model(self.net, step, path, self.net_name, max_model)
    if verbose:
      print(f"  => Saves {self.net_name} at {path}.")
    return path

  def restore(self, step: int, model_folder: str, verbose: bool = True) -> str:
    path = os.path.join(model_folder, self.net_name, f'{self.net_name}-{step}.pth')
    self.net.load_state_dict(torch.load(path, map_location=self.device))
    self.net.to(self.device)
    if verbose:
      print(f"  => Restores {self.net_name} at {path}.")
    return path

  def remove(self, step: int, model_folder: str, verbose: bool = True) -> str:
    path = os.path.join(model_folder, self.net_name, f'{self.net_name}-{step}.pth')
    if verbose:
      print(f"  => Removes {self.net_name} at {path}.")
    if os.path.exists(path):
      os.remove(path)
    return path


class Actor(BaseBlock, BasePolicy):
  policy_type: str = "NNCS"
  net: GaussianPolicy  # TODO: different policies, e.g., GMM.

  def __init__(
      self, cfg, cfg_arch, device: torch.device, obsrv_list: Optional[List] = None, verbose: bool = True
  ) -> None:
    BaseBlock.__init__(self, cfg, device)
    BasePolicy.__init__(self, id=self.net_name, obsrv_list=obsrv_list)

    self.action_dim = int(cfg_arch.action_dim)
    self.action_range = np.array(cfg_arch.action_range, dtype=np.float32)
    self.actor_type: str = cfg.actor_type

    if not self.eval:
      self.update_period = int(cfg.update_period)

    self.build_network(cfg, cfg_arch, verbose=verbose)

  @property
  def is_stochastic(self) -> bool:
    return self.net.is_stochastic

  @property
  def alpha(self):
    return self.log_alpha.exp()

  def build_network(self, cfg, cfg_arch, verbose: bool = True):
    # TODO: different policies, e.g., GMM.
    self.net = GaussianPolicy(
        obsrv_dim=cfg_arch.obsrv_dim, mlp_dim=cfg_arch.mlp_dim, action_dim=self.action_dim,
        action_range=self.action_range, append_dim=cfg_arch.append_dim, latent_dim=cfg_arch.latent_dim,
        activation_type=cfg_arch.activation, device=self.device, verbose=verbose
    )

    # Loads model if specified.
    if hasattr(cfg_arch, "pretrained_path"):
      if cfg_arch.pretrained_path is not None:
        pretrained_path = cfg_arch.pretrained_path
        # We do not want to load log_std.
        net_copy = copy.deepcopy(self.net)
        self.net.load_state_dict(torch.load(pretrained_path, map_location=self.device))
        self.net.log_std.load_state_dict(net_copy.log_std.state_dict())
        del net_copy
        print(f"--> Loads {self.net_name} from {pretrained_path}.")

    if self.eval:
      self.net.eval()
      for _, param in self.net.named_parameters():
        param.requires_grad = False
      self.log_alpha = torch.log(torch.FloatTensor([1e-8])).to(self.device)
    else:
      self.build_optimizer(cfg)

  def build_optimizer(self, cfg):
    super().build_optimizer(cfg)

    # entropy-related parameters
    self.init_alpha = torch.log(torch.FloatTensor([cfg.alpha])).to(self.device)
    self.min_alpha = torch.log(torch.FloatTensor([cfg.min_alpha])).to(self.device)
    self.learn_alpha: bool = cfg.learn_alpha
    self.log_alpha = self.init_alpha.detach().clone()
    self.target_entropy = torch.tensor(-self.action_dim).to(self.device)
    if self.learn_alpha:
      self.log_alpha.requires_grad = True
      self.lr_al: float = cfg.lr_al
      self.lr_al_schedule: bool = cfg.lr_al_schedule
      self.log_alpha_optimizer = self.opt_cls([self.log_alpha], lr=self.lr_al, weight_decay=0.01)
      if self.lr_al_schedule:
        self.lr_al_period: int = cfg.lr_al_period
        self.lr_al_decay: float = cfg.lr_al_decay
        self.lr_al_end: float = cfg.lr_al_end
        self.log_alpha_scheduler = StepLR(
            self.log_alpha_optimizer, step_size=self.lr_al_period, gamma=self.lr_al_decay
        )

  def update_hyper_param(self):
    if not self.eval:
      super().update_hyper_param()

      if self.learn_alpha and self.lr_al_schedule:
        lr = self.log_alpha_optimizer.state_dict()['param_groups'][0]['lr']
        if lr <= self.lr_al_end:
          for param_group in self.log_alpha_optimizer.param_groups:
            param_group['lr'] = self.lr_al_end
        else:
          self.log_alpha_scheduler.step()

  def reset_alpha(self):
    self.log_alpha = self.init_alpha.detach().clone()
    if self.learn_alpha:
      self.log_alpha.requires_grad = True
      self.log_alpha_optimizer = self.opt_cls([self.log_alpha], lr=self.lr_al, weight_decay=0.01)
      if self.lr_al_schedule:
        self.log_alpha_scheduler = StepLR(
            self.log_alpha_optimizer, step_size=self.lr_al_period, gamma=self.lr_al_decay
        )

  def update(self, q1: torch.Tensor, q2: torch.Tensor, log_prob: torch.Tensor,
             update_alpha: bool) -> Tuple[float, float, float]:
    """Updates actor network with Q values (policy gradient).

    Args:
        q1 (torch.Tensor): _description_
        q2 (torch.Tensor): _description_
        log_prob (torch.Tensor): _description_

    Returns:
        Tuple[float, float, float]: _description_
    """
    if self.actor_type == "min":
      q_pi = torch.max(q1, q2)
    elif self.actor_type == "max":
      q_pi = torch.min(q1, q2)

    # cost: min_theta E[ Q + alpha * (log pi + H)]
    # loss_pi = Q + alpha * log pi
    # reward: max_theta E[ Q - alpha * (log pi + H)]
    # loss_pi = -Q + alpha * log pi
    loss_entropy = self.alpha * log_prob.view(-1).mean()
    if self.actor_type == "min":
      loss_q_eval = q_pi.mean()
    elif self.actor_type == "max":
      loss_q_eval = -q_pi.mean()

    if update_alpha:
      loss_pi = loss_q_eval + loss_entropy
    else:
      loss_pi = loss_q_eval

    self.optimizer.zero_grad()
    loss_pi.backward()
    self.optimizer.step()

    # Automatic temperature tuning
    loss_alpha = (self.alpha * (-log_prob.detach() - self.target_entropy)).mean()
    if self.learn_alpha and update_alpha:
      self.log_alpha_optimizer.zero_grad()
      loss_alpha.backward()
      self.log_alpha_optimizer.step()
      self.log_alpha.data = torch.max(torch.min(self.log_alpha.data, self.init_alpha.data), self.min_alpha.data)

    return loss_q_eval.item(), loss_entropy.item(), loss_alpha.item()

  def update_policy(self, actor: Actor):
    self.net.load_state_dict(actor.net.state_dict())

  def get_action(
      self,
      obsrv: Union[np.ndarray, torch.Tensor],
      agents_action: Optional[Dict[str, np.ndarray]] = None,
      append: Optional[Union[np.ndarray, torch.Tensor]] = None,
      latent: Optional[Union[np.ndarray, torch.Tensor]] = None,
      **kwargs,
  ) -> Tuple[np.ndarray, Dict]:
    time0 = time.time()
    if self.obsrv_list is not None:
      assert agents_action is not None, "Must provide agents' actions."
      action_input = self.combine_actions(self.obsrv_list, agents_action)
    else:
      action_input = None

    with torch.no_grad():
      action = self.net(
          obsrv, action=action_input, append=append, latent=latent
      )  # Note that `action` is the same data format as `obsrv`.
    if isinstance(action, torch.Tensor):
      action = action.cpu().numpy()
    t_process = time.time() - time0
    status = 1
    return action, dict(t_process=t_process, status=status)

  def sample(
      self,
      obsrv: Union[np.ndarray, torch.Tensor],
      agents_action: Optional[Dict[str, np.ndarray]] = None,
      append: Optional[Union[np.ndarray, torch.Tensor]] = None,
      latent: Optional[Union[np.ndarray, torch.Tensor]] = None,
      **kwargs,
  ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    if self.obsrv_list is not None:
      assert agents_action is not None, "Must provide agents' actions."
      action_input = self.combine_actions(self.obsrv_list, agents_action)
    else:
      action_input = None

    if self.is_stochastic:
      return self.net.sample(obsrv, action=action_input, append=append, latent=latent)
    else:
      raise ValueError("Not a stochastic policy!")


class Critic(BaseBlock):
  net: TwinnedQNetwork

  def __init__(self, cfg, cfg_arch, device: torch.device, verbose: bool = True) -> None:
    super().__init__(cfg, device)

    if not self.eval:
      self.mode: str = cfg.mode
      self.update_target_period = int(cfg.update_target_period)

    self.build_network(cfg, cfg_arch, verbose=verbose)

  def build_network(self, cfg, cfg_arch, verbose: bool = True):

    self.net = TwinnedQNetwork(
        obsrv_dim=cfg_arch.obsrv_dim, mlp_dim=cfg_arch.mlp_dim, action_dim=cfg_arch.action_dim,
        append_dim=cfg_arch.append_dim, latent_dim=cfg_arch.latent_dim, activation_type=cfg_arch.activation,
        device=self.device, verbose=verbose
    )

    # Loads model if specified.
    if hasattr(cfg_arch, "pretrained_path"):
      if cfg_arch.pretrained_path is not None:
        pretrained_path = cfg_arch.pretrained_path
        self.net.load_state_dict(torch.load(pretrained_path, map_location=self.device))
        print(f"--> Loads {self.net_name} from {pretrained_path}.")

    if self.eval:
      self.net.eval()
      for _, param in self.net.named_parameters():
        param.requires_grad = False
      self.target = self.net  # alias
    else:
      self.target = copy.deepcopy(self.net)
      self.build_optimizer(cfg)

  def build_optimizer(self, cfg):
    super().build_optimizer(cfg)
    self.terminal_type: str = cfg.terminal_type
    self.tau = float(cfg.tau)

    # Discount factor
    self.gamma_schedule: bool = cfg.gamma_schedule
    if self.gamma_schedule:
      self.gamma_scheduler = StepLRMargin(
          init_value=cfg.gamma, period=cfg.gamma_period, decay=cfg.gamma_decay, end_value=cfg.gamma_end, goal_value=1.
      )
      self.gamma: float = self.gamma_scheduler.get_variable()
    else:
      self.gamma: float = cfg.gamma

  def update_hyper_param(self) -> bool:
    """Updates the hyper-parameters of the critic, e.g., discount factor.

    Returns:
        bool: True if the discount factor (gamma) is updated.
    """
    if self.eval:
      return False
    else:
      super().update_hyper_param()
      if self.gamma_schedule:
        old_gamma = self.gamma_scheduler.get_variable()
        self.gamma_scheduler.step()
        self.gamma = self.gamma_scheduler.get_variable()
        if self.gamma != old_gamma:
          return True
      return False

  def update(
      self, q1: torch.Tensor, q2: torch.Tensor, q1_nxt: torch.Tensor, q2_nxt: torch.Tensor,
      non_final_mask: torch.Tensor, reward: torch.Tensor, g_x: torch.Tensor, l_x: torch.Tensor,
      binary_cost: torch.Tensor, entropy_motives: torch.Tensor
  ) -> float:
    """Updates critic network with next Q values (target).

    Args:
        q1 (torch.Tensor):
        q2 (torch.Tensor):
        q1_nxt (torch.Tensor):
        q2_nxt (torch.Tensor):
        non_final_mask (torch.Tensor):
        reward (torch.Tensor):
        g_x (torch.Tensor):
        l_x (torch.Tensor):
        binary_cost (torch.Tensor):
        entropy_motives (torch.Tensor):

    Returns:
        float: critic loss.
    """

    # Gets Bellman update.
    y = get_bellman_update(
        mode=self.mode, batch_size=q1.shape[0], q1_nxt=q1_nxt, q2_nxt=q2_nxt, non_final_mask=non_final_mask,
        reward=reward, g_x=g_x, l_x=l_x, binary_cost=binary_cost, gamma=self.gamma, terminal_type=self.terminal_type
    )
    if self.mode == 'performance':
      y[non_final_mask] += self.gamma * entropy_motives

    # Regresses MSE loss for both Q1 and Q2.
    loss_q1 = mse_loss(input=q1.view(-1), target=y)
    loss_q2 = mse_loss(input=q2.view(-1), target=y)
    loss_q = loss_q1 + loss_q2

    # Backpropagates.
    self.optimizer.zero_grad()
    loss_q.backward()
    self.optimizer.step()
    return loss_q.item()

  def update_target(self):
    soft_update(self.target, self.net, self.tau)

  def restore(self, step: int, model_folder: str, verbose: bool = True):
    path = super().restore(step, model_folder, verbose=verbose)
    if not self.eval:
      self.target.load_state_dict(torch.load(path, map_location=self.device))
      self.target.to(self.device)

  def value(
      self,
      obsrv: Union[np.ndarray, torch.Tensor],
      action: Union[np.ndarray, torch.Tensor],
      append: Optional[Union[np.ndarray, torch.Tensor]] = None,
      latent: Optional[Union[np.ndarray, torch.Tensor]] = None,
  ) -> np.ndarray:
    with torch.no_grad():
      q_pi_1, q_pi_2 = self.net(obsrv, action, append=append, latent=latent)
    value = (q_pi_1+q_pi_2) / 2
    if isinstance(value, torch.Tensor):
      value = value.cpu().numpy()
    return value
