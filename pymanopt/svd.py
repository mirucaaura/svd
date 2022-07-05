from pymanopt.manifolds import Stiefel, Product
from pymanopt.manifolds.manifold import Manifold
from pymanopt import Problem
import autograd.numpy as np
import pymanopt.manifolds
from pymanopt.optimizers import ConjugateGradient

np.random.seed(20220630)

# setting problem
m, n, p = 5, 4, 3
N = np.diag(np.arange(p, 0, -1))
Ur, _ = np.linalg.qr(np.random.normal(size=(m, n)))
Vr, _ = np.linalg.qr(np.random.normal(size=(n, n)))
S = np.diag(np.arange(n, 0, -1))
A = Ur @ S @ Vr.T

# setting manifold
stiefel_1 = Stiefel(m, p)
stiefel_2 = Stiefel(n, p)
manifold = Product([stiefel_1, stiefel_2])

# define cost function
@pymanopt.function.autograd(manifold)
def cost(U, V):
    return -np.trace(U.T @ A @ V @ N)

# define problem and solver
problem = Problem(manifold=manifold, cost=cost)
solver = ConjugateGradient(verbosity=2)

# initial guess
U0, _ = np.linalg.qr(np.random.rand(m, p))
V0, _ = np.linalg.qr(np.random.rand(n, p))
x0 = np.array([U0, V0])
# x0 = manifold.random_point()
# run
res = solver.run(problem, initial_point=x0)
print(res)