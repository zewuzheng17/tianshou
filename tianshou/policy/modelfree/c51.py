from typing import Any, Dict, Optional
from copy import deepcopy
import numpy as np
import torch
from torch.functional import F
from tianshou.data import Batch, ReplayBuffer
from tianshou.policy import DQNPolicy


class C51Policy(DQNPolicy):
    """Implementation of Categorical Deep Q-Network. arXiv:1707.06887.

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param float discount_factor: in [0, 1].
    :param int num_atoms: the number of atoms in the support set of the
        value distribution. Default to 51.
    :param float v_min: the value of the smallest atom in the support set.
        Default to -10.0.
    :param float v_max: the value of the largest atom in the support set.
        Default to 10.0.
    :param int estimation_step: the number of steps to look ahead. Default to 1.
    :param int target_update_freq: the target network update frequency (0 if
        you do not use the target network). Default to 0.
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).

    .. seealso::

        Please refer to :class:`~tianshou.policy.DQNPolicy` for more detailed
        explanation.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            optim: torch.optim.Optimizer,
            discount_factor: float = 0.99,
            num_atoms: int = 51,
            v_min: float = -10.0,
            v_max: float = 10.0,
            estimation_step: int = 1,
            gradnorm: bool = False,
            target_update_freq: int = 0,
            add_infer: bool = False,
            infer_gradient_scale: float = 0.1,
            infer_target_scale: float = 100.,
            reward_normalization: bool = False,
            global_grad_norm: bool = False,
            reset_policy: bool = False,
            reset_policy_interval: int = 50000,
            **kwargs: Any,
    ) -> None:
        super().__init__(
            model, optim, discount_factor, estimation_step, gradnorm, target_update_freq,
            reward_normalization, **kwargs
        )
        assert num_atoms > 1, "num_atoms should be greater than 1"
        assert v_min < v_max, "v_max should be larger than v_min"
        self._num_atoms = num_atoms
        self._v_min = v_min
        self._v_max = v_max
        self.support = torch.nn.Parameter(
            torch.linspace(self._v_min, self._v_max, self._num_atoms),
            requires_grad=False,
        )
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        self._add_infer = add_infer
        self._reset_policy = reset_policy
        if self._add_infer:
            self.model_infer = deepcopy(self.model)
            self.model_infer.train(False)
        if self._reset_policy:
            self.model_reset = deepcopy(self.model)
            self.reset_optim = torch.optim.Adam(self.model_reset.parameters(), lr=optim.param_groups[0]['lr'])
            self.model_init = deepcopy(self.model)

        self._infer_target_scale = infer_target_scale
        self._infer_gradient_scale = infer_gradient_scale
        self._global_grad_norm = global_grad_norm
        self.gradnorm_idx = gradnorm
        self._reset_policy_interval = reset_policy_interval
        self._ini = True
        self.return_dict = {
                                  "loss": True,
                                  "grad_norm": self.gradnorm_idx,
                                  "add_infer": self._add_infer,
                                  "reset_policy": self._reset_policy
                                    }


    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        return self.support.repeat(len(indices), 1)  # shape: [bsz, num_atoms]

    def compute_q_value(
            self, logits: torch.Tensor, mask: Optional[np.ndarray]
    ) -> torch.Tensor:
        return super().compute_q_value((logits * self.support).sum(2), mask)

    def _target_dist(self, batch: Batch) -> torch.Tensor:
        # get q(s', a') for all a'
        if self._target:
            act = self(batch, input="obs_next").act
            next_dist = self(batch, model="model_old", input="obs_next").logits
        else:
            next_batch = self(batch, input="obs_next")
            act = next_batch.act
            next_dist = next_batch.logits
        # select the specific distribution of action a'
        next_dist = next_dist[np.arange(len(act)), act, :]
        target_support = batch.returns.clamp(self._v_min, self._v_max)
        # An amazing trick for calculating the projection gracefully.
        # ref: https://github.com/ShangtongZhang/DeepRL
        target_dist = (
                              1 - (target_support.unsqueeze(1) - self.support.view(1, -1, 1)).abs() /
                              self.delta_z
                      ).clamp(0, 1) * next_dist.unsqueeze(1)  # add the dimension back
        return target_dist.sum(-1)

    def _reset_policys(self):
        self.model_reset = deepcopy(self.model_init)
        self.reset_optim = torch.optim.Adam(self.model_reset.parameters(), lr=self.optim.param_groups[0]['lr'])

    def _learn_reset_policy(self, batch, target_dist):
        self.reset_optim.zero_grad()
        obs = batch['obs']
        obs_next = obs.obs if hasattr(obs, "obs") else obs
        logits, hidden, representation, multi_head_output_q, multi_head_output_v = self.model_reset(obs_next, state=None, info=batch.info)
        act = batch.act
        curr_dist = logits[np.arange(len(act)), act, :]
        weight = batch.pop("weight", 1.0)
        cross_entropy = -(target_dist * torch.log(curr_dist + 1e-8)).sum(1)
        origin_loss = (cross_entropy * weight).mean()
        origin_loss.backward()
        if self._global_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model_reset.parameters(), max_norm=10, norm_type=2)
        self.reset_optim.step()
        return origin_loss.item()

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._target and self._iter % self._freq == 0:
            self.sync_weight()
        self.optim.zero_grad()

        with torch.no_grad():
            target_dist = self._target_dist(batch)
            if self._add_infer:
                _, _, _, infer_target_q, infer_target_v = self.model_infer(batch.obs)
        # print(infer_target)
        weight = batch.pop("weight", 1.0)
        outs = self(batch)
        curr_dist = outs.logits
        curr_infer_target_q = outs.m_h_output_q
        curr_infer_target_v = outs.m_h_output_v
        act = batch.act
        # act is the indice of action that has maximun q value
        curr_dist = curr_dist[np.arange(len(act)), act, :]
        cross_entropy = -(target_dist * torch.log(curr_dist + 1e-8)).sum(1)
        origin_loss = (cross_entropy * weight).mean()

        if curr_infer_target_q is not None and curr_infer_target_v is not None:
            multi_head_loss = self._infer_gradient_scale * F.mse_loss(infer_target_q * self._infer_target_scale, \
                                                                      curr_infer_target_q) + \
                              self._infer_gradient_scale * F.mse_loss(infer_target_v * self._infer_target_scale, \
                                                                      curr_infer_target_v)
            # out0 = F.mse_loss(infer_target * self._infer_target_scale, curr_infer_target, reduction='none')
            # out1 = torch.mean(out0, dim=1)
            # out2 = torch.sum(out1, dim=1)
            # multi_head_loss = self._infer_gradient_scale * torch.mean(out2)
        else:
            multi_head_loss = torch.tensor(0).detach()

        if self._ini == True:
            print("multi_head loss: ", multi_head_loss, "target scale:", self._infer_target_scale)
            self._ini = False

        if self.gradnorm_idx and curr_infer_target is not None:
            self.gradnorm(origin_loss, multi_head_loss)
        else:
            loss = origin_loss + multi_head_loss
            loss.backward()

        if self._global_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10, norm_type=2)

        self.optim.step()

        if self._reset_policy and self._iter % self._reset_policy_interval == 0:
            print("policy reseted!!!")
            self._reset_policys()
            reset_loss = self._learn_reset_policy(batch, target_dist)
        elif self._reset_policy:
            reset_loss = self._learn_reset_policy(batch, target_dist)
        else:
            reset_loss = 0

        batch.weight = cross_entropy.detach()  # prio-buffer
        self._iter += 1

        # parse for return
        parse_return_dict = {
            "grad_norm":{
                "weight0": getattr(self, "multitask_weights", torch.tensor([0, 0]))[0].detach().item(),
                "weight1": getattr(self, "multitask_weights", torch.tensor([0, 0]))[1].detach().item()
            },
            "add_infer":{
                "multi_head_loss": multi_head_loss.item()
            },
            "loss": {
                "loss": loss.item()
            },
            "reset_policy":{
                "reset_policy_loss": reset_loss
            }
        }
        output_dict = {}
        for k in self.return_dict.keys():
            if self.return_dict[k]:
                output_dict.update(parse_return_dict[k])

        return output_dict
        # if self.gradnorm_idx and curr_infer_target is not None:
        #     return {"weight0": self.multitask_weights[0].detach().item(), \
        #             "weight1": self.multitask_weights[1].detach().item(), \
        #             "origin_loss": origin_loss.item(), "multi_head_loss": multi_head_loss.item()}
        # elif self._add_infer:
        #     return {"origin_loss": origin_loss.item(), "multi_head_loss": multi_head_loss.item()}
        # else:
        #     return {"loss": loss.item()}
