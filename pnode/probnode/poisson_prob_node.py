from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torchdyn.core import NeuralODE


class PoissonProbNODE(nn.Module):
    r"""Probability layer where probability of using of NeuralODE equal to 1 / amount_of_forward_passes"""

    def __init__(self, neural_ode: NeuralODE):
        r"""
        Create PoissonProbNODE
        Parameters
        ----------
        neural_ode: NeuralODE
            Neural ODE
        """
        super(PoissonProbNODE, self).__init__()

        self.neural_ode = neural_ode
        self.num_forwards = 0
        self.prob = None

    @property
    def nfe(self):
        return self.neural_ode.vf.nfe

    def _use_ode_to_preprocess(self):
        self.prob = 1 / (self.num_forwards + 1)
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
        return f"PoissonProbNODE:\n" + f"(neural_ode): {self.neural_ode.__repr__()}"
