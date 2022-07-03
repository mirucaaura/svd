from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import os


@dataclass
class SVD:
    A: np.ndarray
    N: np.ndarray
    alpha: float = 0.1
    beta: float = 0.9
    sigma: float = 1e-4
    rho: float = 0.9
    itemax: int = 300
    eps: float = 1e-9

    def F(self, U, V):
        return -np.trace(U.T @ self.A @ V @ self.N)

    def F_opt(self, S, N):
        p = N.shape[0]
        S = S[:p, :p]
        return -np.trace(S @ N)
    
    def Sym(self, X):
        return (X + X.T) / 2

    def qf(self, X):
        Q, _ = np.linalg.qr(X)
        return Q

    def calc_xi(self, U, V):
        return self.A @ V @ self.N - U @ self.Sym(U.T @ self.A @ V @ self.N)

    def calc_eta(self, U, V):
        return self.A.T @ U @ self.N - V @ self.Sym(V.T @ self.A.T @ U @ self.N)
    
    def R(self, xi, eta, U, V):
        return (self.qf(U + xi), self.qf(V + eta))

    def gradF(self, U, V):
        return (U @ self.Sym(U.T @ self.A @ V @ self.N) - self.A @ V @ self.N,
                V @ self.Sym(V.T @ self.A.T @ U @ self.N) - self.A.T @ U @ self.N)

    def inner_product_at_UV(self, U, V, xi, eta):
        """Calculation of inner product.

        Calculation the value of <grad f(U, V), (xi, eta)>_(U, V),
        where (U, V) and (xi, eta) are a current point and a search direction, respectively. 

        Args:
            U (numpy.ndarray): The current point
            V (numpy.ndarray): The current point
            xi (numpy.ndarray): The descent vector
            eta (numpy.ndarray): The descent vector

        Returns:
            numpy.ndarray: <grad f(U, V), (xi, eta)>_(U, V)
        """
        gradF_x, gradF_y = self.gradF(U, V)
        return np.trace(gradF_x.T @ xi) + np.trace(gradF_y.T @ eta)

    def vector_transport(self, zeta, chi, xi, eta, U, V):
        return (zeta - self.qf(U + xi) @ self.Sym(self.qf(U + xi).T @ zeta),
                chi - self.qf(V + eta) @ self.Sym(self.qf(V + eta).T @ chi))

    def norm_grad(self, U, V):
        gradF_x, gradF_y = self.gradF(U, V)
        return np.trace(gradF_x.T @ gradF_x) + np.trace(gradF_y.T @ gradF_y)

    def armijo_rule(self, U, V, xi, eta, c):
        R_xi, R_eta = self.R(c * xi, c * eta, U, V)
        left = self.F(U, V) - self.F(R_xi, R_eta)
        right = -self.sigma * self.inner_product_at_UV(U, V, c * xi, c * eta)
        return left >= right

    def wolfe_rule(self, U, V, xi, eta, c):
        R_xi, R_eta = self.R(c * xi, c * eta, U, V)
        a, b = self.vector_transport(R_xi, R_eta, xi, eta, U, V)
        left = c * self.inner_product_at_UV(R_xi, R_eta, a, b)
        right = self.rho * self.inner_product_at_UV(U, V, xi, eta)
        return left >= right

    def backtrack(self, U, V, xi, eta):
        m = 1
        c = pow(self.beta, m) * self.alpha
        while self.armijo_rule(U, V, xi, eta, c) == False or self.wolfe_rule(U, V, xi, eta, c) == False:
            m += 1
            c = pow(self.beta, m) * self.alpha
            if c < 1e-4:
                break
        return c

    def save_F(self, data, filename):
        path = os.path.dirname(__file__) + filename
        with open(path, 'a') as f:
            np.savetxt(data)

    def steepest_descent(self, U0, V0, verbose=False):
        Uk = U0
        Vk = V0
        for i in range(self.itemax):
            # compute the search direction
            xik = self.calc_xi(Uk, Vk)
            etak = self.calc_eta(Uk, Vk)
            # compute stepsize
            tk = self.backtrack(Uk, Vk, xik, etak)
            # compute the next iterate
            U_next = self.qf(Uk + tk * xik)
            V_next = self.qf(Vk + tk * etak)
            if verbose:
                if i % (self.itemax // 10) == 0:
                    print(i, ':\t', self.norm_grad(xik, etak), self.F(U_next, V_next))
            if self.norm_grad(xik, etak) < self.eps:
                break
            Uk = U_next
            Vk = V_next
        return Uk, Vk

    def cg(self, U0, V0, verbose=False):
        xik = self.calc_xi(U0, V0)
        etak = self.calc_eta(U0, V0)
        bar_xik = -xik
        bar_etak = -etak
        Uk = U0
        Vk = V0
        for i in range(self.itemax):
            # 4. compute stepsize
            tk = self.backtrack(Uk, Vk, xik, etak)
            # 4. set
            U_next = self.qf(Uk + tk * xik)
            V_next = self.qf(Vk + tk * etak)
            # 5. compute zeta, chi
            zeta = bar_xik  - self.qf(Uk + tk * xik)  @ self.Sym(self.qf(Uk + tk * xik).T  @ bar_xik)
            chi  = bar_etak - self.qf(Vk + tk * etak) @ self.Sym(self.qf(Vk + tk * etak).T @ bar_etak)
            # 5. compute bar_xik_next, bar_etak_next
            bar_xik_next = U_next @ self.Sym(U_next.T @ self.A @ V_next @ self.N) - self.A @ V_next @ self.N
            bar_etak_next = V_next @ self.Sym(V_next.T @ self.A.T @ U_next @ self.N) - self.A.T @ U_next @ self.N
            # 6. compute beta
            beta = np.trace(bar_xik_next.T @ (bar_xik_next - zeta)) + np.trace(bar_etak_next.T @ (bar_etak_next - chi)) / (np.trace(bar_xik.T @ bar_xik) + np.trace(bar_etak.T @ bar_etak))
            # 7. set
            xik_next = -bar_xik_next + beta * (xik - self.qf(Uk + tk * xik) @ self.Sym(self.qf(Uk + tk * xik).T @ xik))
            etak_next = -bar_etak_next + beta * (etak - self.qf(Vk + tk * etak) @ self.Sym(self.qf(Vk + tk * etak).T @ etak))
            if verbose:
                if i % (self.itemax // 10) == 0:
                    print(i, ':\t', self.norm_grad(xik_next, etak_next), self.F(U_next, V_next))
            if self.norm_grad(xik_next, etak_next) < self.eps:
                break
            # set
            Uk = U_next
            Vk = V_next
            bar_xik = bar_xik_next
            bar_etak = bar_etak_next
            xik = xik_next
            etak = etak_next
        return Uk, Vk


