# svd

This repository provides `python` implementation of a Riemannian Optimization approach to the matrix singular value decomposition.

## Instrallation

`svd` can be installed with

```
pip install git+https://github.com/mirucaaura/svd.git
```

## Example

Example is stored in `example` folder.

```
python ./example/sample.py
```

## Quickstart

First, import packages:

```python=
import numpy as np
from svd import SVD
```

Define problem as:

```python=
np.random.seed(20220630)
m, n, p = 5, 4, 3

Ur, _ = np.linalg.qr(np.random.normal(size=(m, n)))
Vr, _ = np.linalg.qr(np.random.normal(size=(n, n)))
S = np.diag(np.arange(n, 0, -1))
A = Ur @ S @ Vr.T

mu = np.arange(p, 0, -1)
N = np.diag(mu)
```

Initial point which is on $\mathrm{St}(p,m) \times \mathrm{St}(p,n)$ is needed. Here, we generate initial point as follows:

```python=
U0 = np.eye(m, p)
V0 = np.eye(n, p)
```

Then, you can run optimization algorithm. You can choose steepest descent method and conjugate gradient method. Here, we give an example to use steepest descent method. The approximated optimal solution is returned.

```python=
svd = SVD(A, N)
U_star, V_star = svd.steepest_descent(U0, V0, verbose=True)
```

In this case, we can verify the quality of the obtained solution as follows:

```python=
print('Optimal value obtained by steepst descent: {}'.format(svd.F(U_star, V_star)))
print('Optimal value: {}'.format(svd.F_opt(S, N)))
```

```
Optimal value obtained by steepst descent: -20.00000000000001
Optimal value: -20
```