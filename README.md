# pnode
### Probabilistic Neural Ordinary Differential Equations
| $\mathbb{P}(\hat{\bm{H}}(t) = \bm{X})$ | $p$                                                                                     | $1 - p$               |
|----------------------------------------|-----------------------------------------------------------------------------------------|-----------------------|
| $$\bm{X}$                               | $\text{Solver}\left(\bm{F}_{GCN}(0, \cdot, \bm{\Theta}), \hat{\bm{H}}(t - 1), 1\right)$ | $\hat{\bm{H}}(t - 1)$ |


In order to run experiments (node classification on Cora or Citeseer):
```shell
mkdir venv
python -m venv venv
source venv/bin/activate

./run.sh
```


