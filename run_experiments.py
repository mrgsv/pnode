import os
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Union, Dict
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace

import dgl
import torch
import numpy as np
from tqdm.cli import tqdm
from dgl.data import CoraGraphDataset, CiteseerGraphDataset

from pnode.core.pgcde import NeuralGCDE


DEFAULT_PATH_TO_SAVE_RESULTS = "./results"
DEFAULT_DEVICE = "cpu"
DEFAULT_N_EXPERIMENTS = 25
DEFAULT_N_EPOCHS = 3000
DEFAULT_VERBOSE = 1000
DEFAULT_DATASET = ["Cora"]
DEFAULT_BERNOULLI_PROB = [0.2]
DEFAULT_GAUSSIAN_PARAMS = [0, 1]
DEFAULT_NODE_TYPE_AND_PARAMS = ["constant", "0.5"]
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 5e-4
DEFAULT_SOLVER = "rk4"
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_DROPOUT = 0.4
DEFAULT_GCN_DROPOUT = 0.9

DATASET_BY_NAME = {
    "Cora": CoraGraphDataset,
    "Citeseer": CiteseerGraphDataset,
}

logger = logging.getLogger()


def accuracy(y_hat: torch.Tensor, y: torch.Tensor):
    preds = torch.max(y_hat, 1)[1]
    return torch.mean((y == preds).float())


def _run_experiment(
    n_epochs: int,
    device: str,
    verbose: int,
    features: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    model_params: Dict,
    lr: float,
    weight_decay: float,
    return_only_last_h_in_ode: bool,
) -> Dict:
    model = NeuralGCDE(**model_params).to(device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    criterion = torch.nn.CrossEntropyLoss()
    results = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
        "val_accuracy": [],
        "val_loss": [],
        "forward_time": [],
        "backward_time": [],
        "nfe": []
    }

    for epoch in range(1, n_epochs + 1):
        model.train()
        start_time = time.time()
        outputs = model(features, return_only_last_h_in_ode=return_only_last_h_in_ode)
        f_time = time.time() - start_time
        nfe = int(model._modules["neural_de"].neural_ode.vf.nfe)
        y_pred = outputs
        loss = criterion(y_pred[train_mask], labels[train_mask])
        opt.zero_grad()

        start_time = time.time()
        loss.backward()
        b_time = time.time() - start_time

        opt.step()

        with torch.no_grad():
            model.eval()

            y_pred = model(features, return_only_last_h_in_ode=return_only_last_h_in_ode)
            model._modules["neural_de"].neural_ode.vf.nfe = 0

            train_loss = loss.item()
            train_acc = accuracy(y_pred[train_mask], labels[train_mask]).item()
            test_acc = accuracy(y_pred[test_mask], labels[test_mask]).item()
            test_loss = criterion(y_pred[test_mask], labels[test_mask]).item()
            val_acc = accuracy(y_pred[val_mask], labels[val_mask]).item()
            val_loss = criterion(y_pred[val_mask], labels[val_mask]).item()

            results["train_loss"].append(train_loss)
            results["train_accuracy"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_accuracy"].append(test_acc)
            results["val_loss"].append(val_loss)
            results["val_accuracy"].append(val_acc)
            results["nfe"].append(nfe)
            results["forward_time"].append(f_time)
            results["backward_time"].append(b_time)

        if verbose != 0 and epoch % verbose == 0:
            print(
                f'[{epoch:4d}], Train Loss: {train_loss:3.3f}, Test Loss: {test_loss:3.3f}, '
                f'Train Accuracy: {train_acc:3.3f}, Val Accuracy: {val_acc:3.3f}, '
                f'Test Accuracy: {test_acc:3.3f}, NFE: {nfe:6d}'
            )

    print(f'max test acc: {max(results["test_accuracy"])}')
    print(f"Result prob: {model.neural_de.prob}")

    return results


def run_experiments(
    setup: Namespace,
    n_experiments: int,
    n_epochs: int,
    path_to_save_results: str,
    device: str,
    verbose: int,
    dataset: Union[CoraGraphDataset, CiteseerGraphDataset],
    lr: float,
    weight_decay: float,
    num_layers: int,
    hidden_size: int,
    gcn_hidden_size: int,
    dropout: float,
    gcn_dropout: float,
    solver: str,
    return_t_eval: bool,
    pnode_type_and_params: list,
    return_only_last_h_in_ode: bool,
    noise_rate: float
):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.manual_seed(0)
    np.random.seed(0)

    graph: dgl.DGLGraph = dataset[0]
    features = graph.ndata["feat"].to(device)
    if noise_rate > 0:
        labels = graph.ndata["label"]
        noisy_labels_indices = np.random.choice(
            np.arange(labels.shape[0]),
            size=round(labels.shape[0] * noise_rate),
            replace=False
        )
        unique_labels = labels.unique().numpy()
        for noisy_idx in noisy_labels_indices:
            labels[noisy_idx] = np.random.choice(unique_labels[unique_labels != labels[noisy_idx].item()])
        labels = labels.to(device)
    else:
        labels = graph.ndata["label"].to(device)

    train_mask = graph.ndata["train_mask"]
    val_mask = graph.ndata["val_mask"]
    test_mask = graph.ndata["test_mask"]

    g = graph.add_self_loop().to(device)
    in_feats = features.shape[1]
    out_feats = labels.unique().shape[0]

    # compute diagonal of normalization matrix D according to standard formula
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    # add to dgl.Graph in order for the norm to be accessible at training time
    g.ndata["norm"] = norm.unsqueeze(1)

    neural_gcde_params = {
        "graph": g,
        "num_layers": num_layers,
        "in_feats": in_feats,
        "out_feats": out_feats,
        "hidden_size": hidden_size,
        "dropout": dropout,
        "gcn_hidden_size": gcn_hidden_size,
        "gcn_dropout": gcn_dropout,
        "solver": solver,
        "return_t_eval": return_t_eval,
        "node_type": pnode_type_and_params[0],
    }
    if "constant" == pnode_type_and_params[0]:
        neural_gcde_params["prob"] = float(pnode_type_and_params[1])

    start_time = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    result_path = os.path.join(path_to_save_results, start_time) + f"_{dataset.name}" + f"_{pnode_type_and_params[0]}"
    if "constant" == pnode_type_and_params[0]:
        result_path = result_path + pnode_type_and_params[1]
    result_path = result_path + f"_noise_rate{noise_rate}"
    Path(result_path).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(result_path, "experiment_setup.json"), "w") as f:
        experiment_setup = vars(setup)
        del experiment_setup["callback"]
        json.dump(experiment_setup, f)

    for experiment_idx in tqdm(range(n_experiments)):
        # print(f"Experiment #{experiment_idx} started")
        results = _run_experiment(
            n_epochs=n_epochs,
            device=device,
            verbose=verbose,
            features=features,
            labels=labels,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            model_params=neural_gcde_params,
            lr=lr,
            weight_decay=weight_decay,
            return_only_last_h_in_ode=return_only_last_h_in_ode,
        )

        with open(os.path.join(result_path, str(experiment_idx)) + ".json", "w") as f:
            json.dump(results, f)

        # print(f"Experiment #{experiment_idx} finished\n")


def setup_experiments(arguments: Namespace):
    """
    Args:
        arguments: from parser
    Returns:
        None
    """

    run_experiments(
        setup=arguments,
        n_experiments=arguments.n_experiments,
        n_epochs=arguments.n_epochs,
        path_to_save_results=arguments.path_for_results,
        device=arguments.device,
        verbose=arguments.verbose,
        dataset=DATASET_BY_NAME[arguments.dataset](),
        lr=arguments.lr,
        weight_decay=arguments.weight_decay,
        num_layers=arguments.num_layers,
        hidden_size=arguments.hidden_size,
        gcn_hidden_size=arguments.gcn_hidden_size,
        dropout=arguments.dropout,
        gcn_dropout=arguments.gcn_dropout,
        solver=arguments.solver,
        return_t_eval=arguments.return_t_eval,
        pnode_type_and_params=arguments.pnode_type_and_params,
        return_only_last_h_in_ode=arguments.return_only_last_h_in_ode,
        noise_rate=arguments.noise_rate
    )


def boolean_string(s: str):
    if s not in ("False", "True"):
        raise ValueError("Not a valid boolean string")
    return s == "True"


def setup_parser(parser: ArgumentParser):
    """
    Args:
        parser: ArgumentParser
            Argument parser
    Returns:
        ArgumentParser with adjusted arguments
    """
    parser.add_argument(
        "--n-experiments",
        required=True,
        dest="n_experiments",
        help="amount of experiments",
        type=int
    )
    parser.add_argument(
        "--n-epochs",
        required=True,
        dest="n_epochs",
        help="number of epochs",
        type=int
    )
    parser.add_argument(
        "--device",
        required=True,
        dest="device",
        help="device: cpu|cuda"
    )
    parser.add_argument(
        "--verbose",
        required=True,
        dest="verbose",
        help="verbose steps",
        type=int
    )
    parser.add_argument(
        "--dataset",
        required=True,
        dest="dataset",
        help="dataset name: Cora|Citeseer",
    )
    noise_params_parser = parser.add_mutually_exclusive_group(required=False)
    noise_params_parser.add_argument(
        "--noise-bernoulli-prob", default=DEFAULT_BERNOULLI_PROB,
        dest="noise_bernoulli_prob",
        help="noise bernoulli prob",
        type=float
    )
    noise_params_parser.add_argument(
        "--noise-gaussian-params", default=DEFAULT_GAUSSIAN_PARAMS,
        dest="noise_gaussian_params",
        help="noise gaussian params: mu sigma; e.g. --gaussian-params 0 1",
        nargs=2,
        type=float
    )
    parser.add_argument(
        "--pnode-type-and-params",
        required=True,
        dest="pnode_type_and_params",
        help="type of probability neural ode and its params: constant 0.5|poisson|beta",
        nargs="+"
    )
    parser.add_argument(
        "--lr",
        required=True,
        dest="lr",
        help="learning rate",
        type=float
    )
    parser.add_argument(
        "--weight-decay",
        required=True,
        dest="weight_decay",
        help="weight decay",
        type=float
    )
    parser.add_argument(
        "--path-to-save-results", required=True,
        dest="path_for_results",
        help="directory for saving results"
    )
    parser.add_argument(
        "--solver",
        required=True,
        dest="solver",
        help="solver"
    )
    parser.add_argument(
        "--num-layers",
        required=True,
        dest="num_layers",
        type=int
    )
    parser.add_argument(
        "--hidden-size",
        required=True,
        dest="hidden_size",
        type=int
    )
    parser.add_argument(
        "--gcn-hidden-size",
        required=True,
        dest="gcn_hidden_size",
        type=int
    )
    parser.add_argument(
        "--dropout",
        required=True,
        dest="dropout",
        help="dropout for first layer in GCDE",
        type=float
    )
    parser.add_argument(
        "--gcn-dropout",
        required=True,
        dest="gcn_dropout",
        help="dropout for func in neural gde",
        type=float
    )
    parser.add_argument(
        "--return-t-eval",
        required=True,
        dest="return_t_eval",
        type=boolean_string
    )
    parser.add_argument(
        "--return-only-last-h-in-ode",
        default=True,
        dest="return_only_last_h_in_ode",
        type=boolean_string
    )

    parser.add_argument(
        "--noise-rate",
        required=True,
        dest="noise_rate",
        type=float
    )
    parser.set_defaults(callback=setup_experiments)


def main():
    """Used from CLI"""
    parser = ArgumentParser(
        prog="pnode-experiments",
        description="cli for experiments",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    setup_parser(parser)
    arguments = parser.parse_args()
    arguments.callback(arguments)


if __name__ == "__main__":
    main()
