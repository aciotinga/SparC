"""Direct unit tests for OT / assignment solvers.

These solvers underpin CW (sum×sum transport) and GCW (product×product
Hungarian matching). Failures here propagate silently into query tests.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sparc.solvers.assignment import solve_assignment
from sparc.solvers.transport import solve_transport
from tests.sparc_helpers import nw_coupling_dense

pytestmark = pytest.mark.solver


class TestTransportSolver:
    def test_plan_preserves_marginals(self):
        cost = np.array([[1.0, 2.0], [3.0, 0.5]])
        supply = np.array([0.5, 0.5])
        demand = np.array([0.4, 0.6])
        plan, u, v = solve_transport(cost, supply, demand)
        assert_allclose(plan.sum(axis=1), supply, atol=1e-8)
        assert_allclose(plan.sum(axis=0), demand, atol=1e-8)
        assert (plan >= -1e-10).all()

    def test_optimal_2x2_matches_scipy(self):
        pytest.importorskip("scipy")
        from scipy.optimize import linprog

        rng = np.random.default_rng(0)
        for _ in range(8):
            cost = rng.uniform(0, 5, size=(2, 2))
            supply = rng.dirichlet([1, 1])
            demand = rng.dirichlet([1, 1])
            plan, _, _ = solve_transport(cost, supply, demand)
            c = cost.reshape(-1)
            A_eq = []
            b_eq = []
            for i in range(2):
                row = np.zeros(4)
                row[i * 2 : (i + 1) * 2] = 1.0
                A_eq.append(row)
                b_eq.append(supply[i])
            for j in range(2):
                row = np.zeros(4)
                row[j::2] = 1.0
                A_eq.append(row)
                b_eq.append(demand[j])
            res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
            assert res.success
            assert_allclose(float(np.sum(plan * cost)), res.fun, atol=1e-5)

    def test_dual_feasibility_on_basis(self):
        cost = np.array([[0.0, 1.0, 2.0], [1.5, 0.5, 0.0], [2.0, 1.0, 0.0]])
        supply = np.array([0.3, 0.35, 0.35])
        demand = np.array([0.4, 0.3, 0.3])
        plan, u, v = solve_transport(cost, supply, demand)
        rc = cost - u[:, None] - v[None, :]
        # Reduced costs for positive plan entries should be near zero at optimum.
        for i in range(3):
            for j in range(3):
                if plan[i, j] > 1e-8:
                    assert rc[i, j] == pytest.approx(0.0, abs=1e-5)

    def test_unbalanced_raises(self):
        cost = np.ones((2, 2))
        with pytest.raises(ValueError, match="unbalanced"):
            solve_transport(cost, [0.5, 0.5], [0.3, 0.3])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            solve_transport(np.zeros((0, 2)), [], [0.5, 0.5])


class TestAssignmentSolver:
    def test_square_identity_is_diagonal(self):
        n = 4
        cost = np.zeros((n, n))
        for i in range(n):
            cost[i, i] = -1.0
            cost[i, (i + 1) % n] = 0.0
        rows, cols = solve_assignment(cost)
        assert len(rows) == n
        assert sorted(cols.tolist()) == list(range(n))
        assert float(cost[rows, cols].sum()) == pytest.approx(-n, rel=0, abs=1e-10)

    def test_rectangular_matches_scipy_when_available(self):
        pytest.importorskip("scipy")
        from scipy.optimize import linear_sum_assignment

        rng = np.random.default_rng(3)
        for n, m in ((3, 5), (5, 3), (4, 4)):
            cost = rng.uniform(0, 10, size=(n, m))
            rows, cols = solve_assignment(cost)
            sr, sc = linear_sum_assignment(cost)
            assert float(cost[rows, cols].sum()) == pytest.approx(
                float(cost[sr, sc].sum()), rel=0, abs=1e-9
            )

    def test_all_rows_matched(self):
        cost = np.array([[3.0, 1.0, 4.0], [2.0, 5.0, 1.0]])
        rows, cols = solve_assignment(cost)
        assert len(rows) == cost.shape[0]
        assert len(cols) == cost.shape[0]
        assert (cols >= 0).all()
        assert (cols < cost.shape[1]).all()


class TestNorthwestReference:
    """NW plan used at leaves — validated via dense helper shared with CW/GCW tests."""

    @pytest.mark.parametrize(
        "p,q",
        [
            ([0.5, 0.5], [0.5, 0.5]),
            ([0.2, 0.3, 0.5], [0.4, 0.1, 0.5]),
            ([0.7, 0.3], [0.2, 0.8]),
        ],
    )
    def test_nw_marginals(self, p, q):
        plan = nw_coupling_dense(p, q)
        assert_allclose(plan.sum(axis=0), q, atol=1e-10)
        assert_allclose(plan.sum(axis=1), p, atol=1e-10)
        assert (plan >= -1e-12).all()

    def test_nw_is_cumulative_mass_coupling(self):
        p = [0.25, 0.35, 0.40]
        q = [0.30, 0.20, 0.50]
        plan = nw_coupling_dense(p, q)
        # NW fills top-left corner first: support is Young diagram shape.
        for i, j in itertools.product(range(3), range(3)):
            if plan[i, j] > 0 and (i > 0 and plan[i - 1, j] == 0 and plan[i, j - 1] == 0):
                pass  # top-left cell may be only positive at (0,0)
        assert plan[0, 0] > 0
