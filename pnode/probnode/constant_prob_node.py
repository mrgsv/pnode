from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torchdyn.core import NeuralODE


class ConstantProbNODE(nn.Module):
    r"""Probability layer with constant probability of using of NeuralODE"""
    def __init__(self, neural_ode: NeuralODE, prob: float = 0.5):
        """
        Create ConstantProbNODE
        Parameters
        ----------
        neural_ode: NeuralODE
            Neural ODE
        prob: float
            Probability of using NeuralODE
        """
        super(ConstantProbNODE, self).__init__()

        self.neural_ode = neural_ode
        assert 0 <= prob <= 1, (
            f"Expected probability from 0 to 1, but got: {prob}"
        )
        self.prob = prob

    @property
    def nfe(self):
        return self.neural_ode.vf.nfe

    def _use_ode_to_preprocess(self):
        return np.random.binomial(1, self.prob) == 1

    def forward(
        self,
        h: torch.Tensor,
        t_span: Optional[torch.Tensor] = None,
        return_only_last_h_in_ode: bool = True
    ) -> torch.Tensor:
        if self.training:
            if self._use_ode_to_preprocess():
                traj = self.neural_ode(h, t_span)
                h = traj[-1] if return_only_last_h_in_ode else traj
        else:
            traj = (1 - self.prob) * h + self.prob * self.neural_ode(h, t_span)
            h = traj[-1] if return_only_last_h_in_ode else traj

        return h

    def __repr__(self):
        return f"ConstantProbNODE:\n\t- prob: {self.prob}\n" + f"(neural_ode): {self.neural_ode.__repr__()}"
