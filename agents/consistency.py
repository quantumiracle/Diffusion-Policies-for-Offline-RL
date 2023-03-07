# Implementation of Consistency Model
# https://arxiv.org/pdf/2303.01469.pdf

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

class Consistency(nn.Module):
    """
    This is ridiculous Unet structure, hey but it works!
    """

    def __init__(self, state_dim, action_dim, model, max_action, 
                 beta_schedule='linear', n_timesteps=100,
                 loss_type='l2', clip_denoised=True, predict_epsilon=True,
                 eps: float = 0.002, D: int = 128) -> None:
        super(Consistency, self).__init__()

        self.eps = eps
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.model = model

    def predict_consistency(self, state, action, t) -> torch.Tensor:
        if isinstance(t, float):
            t = (
                torch.tensor([t] * action.shape[0], dtype=torch.float32)
                .to(action.device)
                .unsqueeze(1)
            )  # (batch, 1)

        action_ori = action  # (batch, action_dim)
        action = self.model(action, t, state)  # be careful of the order

        t = t - self.eps
        c_skip_t = 0.25 / (t.pow(2) + 0.25) # (batch, 1)
        c_out_t = 0.25 * t / ((t + self.eps).pow(2) + 0.25).pow(0.5)
        return c_skip_t * action_ori + c_out_t * action

    def loss(self, state, action, z, t1, t2, ema_model):
        x2 = action + z * t2  # x2: (batch, action_dim), t2: (batch, 1)
        x2 = self.predict_consistency(state, x2, t2)

        with torch.no_grad():
            x1 = action + z * t1
            x1 = ema_model.predict_consistency(state, x1, t1)

        return F.mse_loss(x1, x2)

    @torch.no_grad()
    def sample(self, state, ts: List[float]=list(reversed([2.0, 80.0]))):
        action_shape = list(state.shape)
        action_shape[-1] = self.action_dim
        action = torch.randn(action_shape).to(device=state.device) * 80.0  # TODO 80?
        action = self.predict_consistency(state, action, ts[0])

        for t in ts[1:]:
            z = torch.randn_like(action)
            action = action + math.sqrt(t**2 - self.eps**2) * z
            action = self.predict_consistency(state, action, t)

        action.clamp_(-self.max_action, self.max_action)
        return action

    def forward(self, state) -> torch.Tensor:
        action_shape = list(state.shape)
        action_shape[-1] = self.action_dim
        # Sample 2 Steps
        pre_action = self.sample(
            state,
            list(reversed([2.0, 80.0])),
        )
        return pre_action


# original one for image (batch, C, H, W)
# class Consistency(nn.Module):
#     """
#     This is ridiculous Unet structure, hey but it works!
#     """

#     def __init__(self, state_dim, action_dim, model, max_action, 
#                  beta_schedule='linear', n_timesteps=100,
#                  loss_type='l2', clip_denoised=True, predict_epsilon=True,
#                  eps: float = 0.002, D: int = 128) -> None:
#         super(Consistency, self).__init__()

#         self.eps = eps
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.max_action = max_action
#         self.model = model

#     def forward(self, x, t) -> torch.Tensor:
#         if isinstance(t, float):
#             t = (
#                 torch.tensor([t] * x.shape[0], dtype=torch.float32)
#                 .to(x.device)
#                 .unsqueeze(1)
#             )  # (batch, 1)

#         x_ori = x  # (batch, C, H, W)
#         x = self.model(x, t)

#         t = t - self.eps
#         c_skip_t = 0.25 / (t.pow(2) + 0.25) # (batch, 1)
#         c_out_t = 0.25 * t / ((t + self.eps).pow(2) + 0.25).pow(0.5)
#         return c_skip_t[:, :, None, None] * x_ori + c_out_t[:, :, None, None] * x

#     def loss(self, x, z, t1, t2, ema_model):
#         x2 = x + z * t2[:, :, None, None]
#         x2 = self(x2, t2)

#         with torch.no_grad():
#             x1 = x + z * t1[:, :, None, None]
#             x1 = ema_model(x1, t1)

#         return F.mse_loss(x1, x2)

#     @torch.no_grad()
#     def sample(self, x, ts: List[float]):
#         x = self(x, ts[0])

#         for t in ts[1:]:
#             z = torch.randn_like(x)
#             x = x + math.sqrt(t**2 - self.eps**2) * z
#             x = self(x, t)

#         return x
