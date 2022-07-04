import numpy as np
from svd.svd import SVD
import matplotlib.pyplot as plt

np.random.seed(20220630)
m, n, p = 5, 4, 3
mu = np.arange(p, 0, -1)
N = np.diag(mu)

Ur, _ = np.linalg.qr(np.random.normal(size=(m, n)))
Vr, _ = np.linalg.qr(np.random.normal(size=(n, n)))
S = np.diag(np.arange(n, 0, -1))
A = Ur @ S @ Vr.T

# initialize
U0 = np.eye(m, p)
V0 = np.eye(n, p)

svd = SVD(A, N)
U_star_st, V_star_st = svd.steepest_descent(U0, V0, verbose=True)
U_star_cg, V_star_cg = svd.cg(U0, V0, verbose=True)

print('Optimal value obtained by steepst descent: {}'.format(svd.F(U_star_st, V_star_st)))
print('Optimal value obtained by conjugate descent: {}'.format(svd.F(U_star_cg, V_star_cg)))
print('Optimal value: {}'.format(svd.F_opt(S, N)))