# Copyright (C) 2018, Anass Al
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>

import warnings
import numpy as np
from planners.ilqr.cost import TreeCost


class iLQR:
    """Finite Horizon Iterative Linear Quadratic Regulator."""

    def __init__(self, dynamics, max_reg=1e10, hessians=False):
        """Constructs an iLQR solver.

        Args:
            dynamics: Plant dynamics.
            max_reg: Maximum regularization term to break early due to
                divergence. This can be disabled by setting it to None.
            hessians: Use the dynamic model's second order derivatives.
                Default: only use first order derivatives. (i.e. iLQR instead
                of DDP).
        """
        self.dynamics = dynamics
        self.cost = None
        self.N = None
        self._use_hessians = hessians and dynamics.has_hessians
        if hessians and not dynamics.has_hessians:
            warnings.warn("hessians requested but are unavailable in dynamics")

        # Regularization terms: Levenberg-Marquardt parameter.
        # See II F. Regularization Schedule.
        self._mu = 1.0
        self._mu_min = 1e-6
        self._mu_max = max_reg
        self._delta_0 = 2.0
        self._delta = self._delta_0
        self._mu = 1.0
        self.rel_tol = 1e-6
        self.abs_tol = 1e-6

        self._k = None
        self._K = None
        self.k = None
        self.K = None

        self._nominal_xs = None
        self._nominal_us = None

        self.xs = None
        self.us = None
        self.F_x = None
        self.F_u = None
        self.F_xx = None
        self.F_ux = None
        self.F_uu = None
        self.L = None
        self.L_x = None
        self.L_u = None
        self.L_xx = None
        self.L_ux = None
        self.L_uu = None
        self.dV = None
        self.V_x = None
        self.V_xx = None

        self.J_opt = None

        super(iLQR, self).__init__()

    def fit(self, us_init, cost: TreeCost = None, n_iterations=100):
        """Computes the optimal controls.

        Args:
            us_init: Initial control path [N, action_size].
            cost: Cost function in tree structure. Default: None.
            n_iterations: Maximum number of interations. Default: 100.
            tol: Tolerance. Default: 1e-6.

        Returns:
            Tuple of
                xs: optimal state path [N+1, state_size].
                us: optimal control path [N, action_size].
        """
        self.cost = cost
        N = len(us_init)
        self.N = N

        # _k和_K是用于存储控制增益的数组，分别对应于每个时间步的增益
        self._k = np.zeros((N, self.dynamics.action_size))
        self._K = np.zeros((N, self.dynamics.action_size, self.dynamics.state_size))

        # xs和us用于存储状态和控制路径，初始为空数组
        self.xs = np.empty((N, self.dynamics.state_size))
        self.us = np.empty((N, self.dynamics.action_size))

        # F_x和F_u存储状态和控制对状态变化的影响
        self.F_x = np.empty((N, self.dynamics.state_size, self.dynamics.state_size))
        self.F_u = np.empty((N, self.dynamics.state_size, self.dynamics.action_size))

        # 如果启用海森矩阵，初始化相应的多维数组，以计算二阶导数
        if self._use_hessians:
            self.F_xx = np.empty((N, self.dynamics.state_size, self.dynamics.state_size, self.dynamics.state_size))
            self.F_ux = np.empty((N, self.dynamics.state_size, self.dynamics.action_size, self.dynamics.state_size))
            self.F_uu = np.empty((N, self.dynamics.state_size, self.dynamics.action_size, self.dynamics.action_size))
        
        # 初始化存储每个时间步成本及其对状态和控制的导数（包括一阶和二阶导数）
        self.L = np.empty(N)
        self.L_x = np.empty((N, self.dynamics.state_size))
        self.L_u = np.empty((N, self.dynamics.action_size))
        self.L_xx = np.empty((N, self.dynamics.state_size, self.dynamics.state_size))
        self.L_ux = np.empty((N, self.dynamics.action_size, self.dynamics.state_size))
        self.L_uu = np.empty((N, self.dynamics.action_size, self.dynamics.action_size))

        # dV存储价值函数的变化，V_x和V_xx分别存储一阶和二阶导数
        self.dV = np.empty(N)
        self.V_x = np.empty((N, self.dynamics.state_size))
        self.V_xx = np.empty((N, self.dynamics.state_size, self.dynamics.state_size))

        # Reset regularization term. 初始化正则化参数_mu和_delta，用于稳定优化过程
        self._mu = 1.0
        self._delta = self._delta_0

        # Backtracking line search candidates 0 < alpha <= 1. 定义候选步长alphas，用于后续的回溯线搜索，以确保控制更新的有效性
        alphas = 1.1 ** (-np.arange(10) ** 2)

        # 将初始控制路径复制到self.us，并将增益数组赋值给self.k和self.K
        self.us = us_init.copy()
        self.k = self._k
        self.K = self._K

        # accepted标记当前更新是否被接受，converged标记是否收敛
        accepted = True
        converged = False

        # 开始循环，直到达到最大迭代次数
        for iteration in range(n_iterations):
            # Forward rollout only if it needs to be recomputed. 如果当前更新被接受，执行前向传播以计算状态路径，并将accepted设置为False，以避免重复计算
            if accepted:
                self._forward_rollout()
                accepted = False
            try:
                # Backward pass. 尝试进行后向传播，以计算梯度和更新控制
                self._backward_pass()

                # Backtracking line search.
                accepted, converged = self._backtrack_line_search(alphas)   # 执行回溯线搜索，根据步长调整更新并检查是否接受。
                if converged:
                    break   # 如果算法收敛，退出迭代循环

            except np.linalg.LinAlgError as e:  # 捕获线性代数错误，通常与矩阵不正定相关
                # Quu was not positive-definite and this diverged.
                # Try again with a higher regularization term.
                warnings.warn(str(e))

            if not accepted:
                # Increase regularization term. 如果未接受更新，增加正则化参数，以便稳定计算。如果超过最大正则化值，则发出警告并退出
                warnings.warn("increasing regularization term")
                self._delta = max(1.0, self._delta) * self._delta_0
                self._mu = max(self._mu_min, self._mu * self._delta)
                if self._mu_max and self._mu >= self._mu_max:
                    warnings.warn("exceeded max regularization term")
                    break


        # Store fit parameters. 保存最终的控制增益和路径
        self._k = self.k
        self._K = self.K
        self._nominal_xs = self.xs
        self._nominal_us = self.us

        return self.xs, self.us

    def _check_convergence(self, J_new):
        """Checks if the algorithm has converged.

        Returns:
            Whether the algorithm has converged.
        """
        if np.abs((self.J_opt - J_new) / self.J_opt) < self.rel_tol:
            return True
        return False


    def _backtrack_line_search(self, alphas):
        converged = False
        accepted = False
        for alpha in alphas:
            xs_new, us_new = self._line_search(alpha)
            J_new = self._trajectory_cost(xs_new, us_new)
            if J_new < self.J_opt:
                if self._check_convergence(J_new):
                    converged = True

                accepted = True
                self.xs = xs_new
                self.us = us_new

                # Decrease regularization term.
                self._delta = min(1.0, self._delta) / self._delta_0
                self._mu *= self._delta
                if self._mu <= self._mu_min:
                    self._mu = 0.0
                return accepted, converged
        warnings.warn("Line search failed")
        return accepted, converged

    def _line_search(self, alpha=1.0):
        """Applies the line search for a given trajectory.

        Args:
            xs: Nominal state path [N, state_size].
            us: Nominal control path [N, action_size].
            k: Feedforward gains [N, action_size].
            K: Feedback gains [N, action_size, state_size].
            alpha: Line search coefficient.

        Returns:
            Tuple of
                xs: state path [N+1, state_size].
                us: control path [N, action_size].
        """
        us_new = np.zeros_like(self.us)
        xs_new = np.zeros_like(self.xs)
        x0 = self.cost.tree.get_root().data

        # Eq (12).
        us_new[0] = self.us[0] + alpha * self.k[0]
        xs_new[0] = self.dynamics.f(x0, us_new[0], 0)

        queue = [0]
        while len(queue) > 0:
            parent_key = queue.pop()

            for child_key in self.cost.tree.get_children_keys(parent_key):
                # Eq (12).
                us_new[child_key] = self.us[child_key] + alpha * self.k[child_key] + self.K[child_key].dot(
                    xs_new[parent_key] - self.xs[parent_key])

                # Eq (8c).
                xs_new[child_key] = self.dynamics.f(xs_new[parent_key], us_new[child_key], child_key)

                if self.cost.tree.has_children(child_key):
                    queue.append(child_key)

        return xs_new, us_new

    def _trajectory_cost(self, xs, us):
        """Computes the given trajectory's cost.

        Args:
            xs: State path [N, state_size].
            us: Control path [N, action_size].

        Returns:
            Trajectory's total cost.
        """
        J = map(lambda args: self.cost.l(*args), zip(xs, us, range(self.N)))
        return sum(J)

    def _forward_rollout(self):
        """Apply the forward dynamics to have a trajectory from the starting
        state x0 by applying the control path us.
        Returns:
            Tuple of:
                xs: State path [N, state_size].
                F_x: Jacobian of state path w.r.t. x
                    [N, state_size, state_size].
                F_u: Jacobian of state path w.r.t. u
                    [N, state_size, action_size].
                L: Cost path [N].
                L_x: Jacobian of cost path w.r.t. x [N, state_size].
                L_u: Jacobian of cost path w.r.t. u [N, action_size].
                L_xx: Hessian of cost path w.r.t. x, x
                    [N, state_size, state_size].
                L_ux: Hessian of cost path w.r.t. u, x
                    [N, action_size, state_size].
                L_uu: Hessian of cost path w.r.t. u, u
                    [N, action_size, action_size].
                F_xx: Hessian of state path w.r.t. x, x if Hessians are used
                    [N, state_size, state_size, state_size].
                F_ux: Hessian of state path w.r.t. u, x if Hessians are used
                    [N, state_size, action_size, state_size].
                F_uu: Hessian of state path w.r.t. u, u if Hessians are used
                    [N, state_size, action_size, action_size].
        """
        # the first state is always the propagated state from the initial state
        # the state[i] is propagated from the state[i-1] using the control[i]
        x0 = self.cost.tree.get_root().data
        u = self.us[0]
        x = self.dynamics.f(x0, u, 0)
        self.xs[0] = x
        self.F_x[0] = self.dynamics.f_x(x, u, 0)
        self.F_u[0] = self.dynamics.f_u(x, u, 0)
        self.L[0] = self.cost.l(x, u, 0, terminal=False)
        self.L_x[0] = self.cost.l_x(x, u, 0, terminal=False)
        self.L_u[0] = self.cost.l_u(x, u, 0, terminal=False)
        self.L_xx[0] = self.cost.l_xx(x, u, 0, terminal=False)
        self.L_ux[0] = self.cost.l_ux(x, u, 0, terminal=False)
        self.L_uu[0] = self.cost.l_uu(x, u, 0, terminal=False)
        if self._use_hessians:
            self.F_xx[0] = self.dynamics.f_xx(x, u, 0)
            self.F_ux[0] = self.dynamics.f_ux(x, u, 0)
            self.F_uu[0] = self.dynamics.f_uu(x, u, 0)

        #  BFS to propagate the tree
        queue = [0]
        while len(queue) > 0:
            parent_key = queue.pop()
            # print("forward rollout from No.", parent_key, " state: ", self.xs[parent_key])
            pre_x = self.xs[parent_key]
            for child_key in self.cost.tree.get_children_keys(parent_key):
                u = self.us[child_key]
                x = self.dynamics.f(pre_x, u, child_key)
                self.xs[child_key] = x
                self.F_x[child_key] = self.dynamics.f_x(x, u, child_key)
                self.F_u[child_key] = self.dynamics.f_u(x, u, child_key)

                is_terminal = not self.cost.tree.has_children(child_key)

                self.L[child_key] = self.cost.l(x, u, child_key, terminal=is_terminal)
                self.L_x[child_key] = self.cost.l_x(x, u, child_key, terminal=is_terminal)
                self.L_u[child_key] = self.cost.l_u(x, u, child_key, terminal=is_terminal)
                self.L_xx[child_key] = self.cost.l_xx(x, u, child_key, terminal=is_terminal)
                self.L_ux[child_key] = self.cost.l_ux(x, u, child_key, terminal=is_terminal)
                self.L_uu[child_key] = self.cost.l_uu(x, u, child_key, terminal=is_terminal)

                if self._use_hessians:
                    self.F_xx[child_key] = self.dynamics.f_xx(x, u, child_key)
                    self.F_ux[child_key] = self.dynamics.f_ux(x, u, child_key)
                    self.F_uu[child_key] = self.dynamics.f_uu(x, u, child_key)

                if not is_terminal:
                    queue.append(child_key)

        self.J_opt = self.L.sum()

    def _backward_pass(self):
        """Computes the feedforward and feedback gains k and K.
        """
        self.dV = np.zeros(self.N)
        self.V_x = np.zeros((self.N, self.dynamics.state_size))
        self.V_xx = np.zeros((self.N, self.dynamics.state_size, self.dynamics.state_size))
        self.k = np.zeros((self.N, self.dynamics.action_size))
        self.K = np.zeros((self.N, self.dynamics.action_size, self.dynamics.state_size))

        # Recursive backward pass.
        self._recursive_backward_pass(self.cost.tree.get_root_key())

    def _recursive_backward_pass(self, key):
        for child_key in self.cost.tree.get_children_keys(key):
            self._recursive_backward_pass(child_key)
            self._get_feedback_gains(child_key)

            self.V_x[key] += self.V_x[child_key]
            self.V_xx[key] += self.V_xx[child_key]

    def _get_feedback_gains(self, key):
        if self._use_hessians:
            Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q(self.F_x[key], self.F_u[key], self.L_x[key],
                                                 self.L_u[key], self.L_xx[key], self.L_ux[key],
                                                 self.L_uu[key], self.V_x[key], self.V_xx[key],
                                                 self.F_xx[key], self.F_ux[key], self.F_uu[key])
        else:
            Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q(self.F_x[key], self.F_u[key], self.L_x[key],
                                                 self.L_u[key], self.L_xx[key], self.L_ux[key],
                                                 self.L_uu[key], self.V_x[key], self.V_xx[key])

        self.k[key] = -np.linalg.solve(Q_uu, Q_u)
        self.K[key] = -np.linalg.solve(Q_uu, Q_ux)

        self.dV[key] = self.k[key].dot(Q_u) + 0.5 * self.k[key].dot(Q_uu).dot(self.k[key])

        self.V_x[key] = Q_x + self.K[key].T.dot(Q_uu).dot(self.k[key])
        self.V_x[key] += self.K[key].T.dot(Q_u) + Q_ux.T.dot(self.k[key])

        self.V_xx[key] = Q_xx + self.K[key].T.dot(Q_uu).dot(self.K[key])
        self.V_xx[key] += self.K[key].T.dot(Q_ux) + Q_ux.T.dot(self.K[key])
        self.V_xx[key] = 0.5 * (self.V_xx[key] + self.V_xx[key].T)  # To maintain symmetry.

    def _Q(self, f_x, f_u, l_x, l_u, l_xx, l_ux, l_uu, V_x, V_xx,
           f_xx=None, f_ux=None, f_uu=None):
        """Computes second order expansion.

        Args:
            F_x: Jacobian of state w.r.t. x [state_size, state_size].
            F_u: Jacobian of state w.r.t. u [state_size, action_size].
            L_x: Jacobian of cost w.r.t. x [state_size].
            L_u: Jacobian of cost w.r.t. u [action_size].
            L_xx: Hessian of cost w.r.t. x, x [state_size, state_size].
            L_ux: Hessian of cost w.r.t. u, x [action_size, state_size].
            L_uu: Hessian of cost w.r.t. u, u [action_size, action_size].
            V_x: Jacobian of the value function at the next time step
                [state_size].
            V_xx: Hessian of the value function at the next time step w.r.t.
                x, x [state_size, state_size].
            F_xx: Hessian of state w.r.t. x, x if Hessians are used
                [state_size, state_size, state_size].
            F_ux: Hessian of state w.r.t. u, x if Hessians are used
                [state_size, action_size, state_size].
            F_uu: Hessian of state w.r.t. u, u if Hessians are used
                [state_size, action_size, action_size].

        Returns:
            Tuple of
                Q_x: [state_size].
                Q_u: [action_size].
                Q_xx: [state_size, state_size].
                Q_ux: [action_size, state_size].
                Q_uu: [action_size, action_size].
        """
        # Eqs (5a), (5b) and (5c).
        Q_x = l_x + f_x.T.dot(V_x)
        Q_u = l_u + f_u.T.dot(V_x)
        Q_xx = l_xx + f_x.T.dot(V_xx).dot(f_x)

        # Eqs (11b) and (11c).
        reg = self._mu * np.eye(self.dynamics.state_size)
        Q_ux = l_ux + f_u.T.dot(V_xx + reg).dot(f_x)
        Q_uu = l_uu + f_u.T.dot(V_xx + reg).dot(f_u)

        if self._use_hessians:
            Q_xx += np.tensordot(V_x, f_xx, axes=1)
            Q_ux += np.tensordot(V_x, f_ux, axes=1)
            Q_uu += np.tensordot(V_x, f_uu, axes=1)

        return Q_x, Q_u, Q_xx, Q_ux, Q_uu
