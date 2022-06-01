from typing import Optional

import dgl
import torch
import torch.nn as nn
from torchdyn.core import NeuralODE

from .gcnlayer import GCNLayer
from ..utils import probability_node_factory


class NeuralGCDE(nn.Module):
    r"""Simple implementation of Neural Graph Convolutional Differential Equation"""
    def __init__(
            self,
            graph: dgl.DGLGraph,
            num_layers: int,
            in_feats: int,
            out_feats: int,
            hidden_size: int,
            dropout: float,
            gcn_hidden_size: int,
            gcn_dropout: float,
            solver: str = "rk4",
            return_t_eval: bool = False,
            **kwargs
    ):
        super(NeuralGCDE, self).__init__()
        func = nn.Sequential(
            GCNLayer(
                g=graph,
                in_feats=hidden_size,
                out_feats=gcn_hidden_size,
                activation=nn.Softplus(),
                dropout=gcn_dropout
            ),
            GCNLayer(
                g=graph,
                in_feats=gcn_hidden_size,
                out_feats=hidden_size,
                activation=None,
                dropout=gcn_dropout
            )
        )
        neural_de = NeuralODE(func, solver=solver, return_t_eval=return_t_eval)
        self.neural_de = probability_node_factory(neural_de, **kwargs)

        self.gcn1 = GCNLayer(
            g=graph,
            in_feats=in_feats,
            out_feats=hidden_size,
            activation=None,
            dropout=dropout
        )
        self.other_gcn_layers = nn.Sequential(
            *[
                GCNLayer(
                    g=graph,
                    in_feats=hidden_size,
                    out_feats=hidden_size,
                    activation=nn.ReLU(),
                    dropout=dropout
                )
                for _ in range(num_layers - 2)
            ]
        ) if num_layers - 2 > 0 else None
        self.gcn2 = GCNLayer(
            g=graph,
            in_feats=hidden_size,
            out_feats=out_feats,
            activation=None,
            dropout=0.
        )

    def forward(
        self,
        h: torch.Tensor,
        t_span: Optional[torch.Tensor] = None,
        return_only_last_h_in_ode: bool = True
    ):
        h = self.gcn1(h)
        if self.other_gcn_layers:
            h = self.other_gcn_layers(h)
        h = self.neural_de(h, t_span, return_only_last_h_in_ode)
        h = self.gcn2(h)

        return h
