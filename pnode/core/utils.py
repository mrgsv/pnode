from typing import Union

from torchdyn.core import NeuralODE

from ..probnode import ConstantProbNODE, PoissonProbNODE


def probability_node_factory(
    neural_ode: NeuralODE,
    **kwargs
) -> Union[ConstantProbNODE, PoissonProbNODE]:
    r"""
    Probability neural ode factory
    Parameters
    ----------
    neural_ode: NeuralODE
        Neural ODE for wrapping
    kwargs: Dict
        kwargs
    Returns
    -------
    One of ProbabilityNODE modules
    """
    if "poisson" == kwargs.get("node_type"):
        return PoissonProbNODE(neural_ode)
    elif "constant" == kwargs.get("node_type"):
        prob = kwargs.get("prob")
        return ConstantProbNODE(neural_ode, prob)
    else:
        raise NotImplementedError(
            f"Expected node_type in ('poisson', 'constant'), but got: {kwargs.get('node_type')}"
        )
