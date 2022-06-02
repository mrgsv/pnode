# pnode
### Probabilistic Neural Ordinary Differential Equations
| $\mathbb{P}(\hat{\mathbf{H}}(t) = \mathbf{X})$ | $p$                                                                                     | $1 - p$               |
|----------------------------------------|-----------------------------------------------------------------------------------------|-----------------------|
| $\mathbf{X}$                               | $\text{Solver}\left(\mathbf{F}_{GCN}(0, \cdot, \mathbf{\Theta}), \hat{\mathbf{H}}(t - 1), 1\right)$ | $\hat{\mathbf{H}}(t - 1)$ |

where $\hat{\mathbf{H}}(t)$ (discrete random variable) -- hidden state at layer $t$.

In order to run experiments (node classification on Cora or Citeseer):
```shell
mkdir venv
python -m venv venv
source venv/bin/activate

./run.sh
```


