"""Robustify a saved PC via DRO using Optimistic Gradient Descent Ascent (OGDA).

Loads a ``gcw-circuit-v1`` JSON from ``examples/example_pcs/`` and solves

    max_theta  min_{Q : CW(P_hat, Q) <= k}  log E_Q[P_theta]

via a Lagrangian saddle point with simultaneous OGDA updates on theta, phi, and
lambda. Each iteration extrapolates player directions as ``2 * g_t - g_{t-1}``
(plain GDA on the first step). The theta player uses a sample-based Monte Carlo
estimate of ``grad_theta E_Q[log P_theta]``.

By default ``eta_phi = 10 * eta_theta``. Override with ``--eta-phi`` if needed.
Use ``--theta-seed`` to stabilize stochastic theta gradients.

    python examples/robustify_ogda.py plants.json
    python examples/robustify_ogda.py plants.json --dataset-k 3 --eval-every 5
    python examples/robustify_ogda.py plants.json --gda  # ablation: disable optimism
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sparc.circuit import Circuit
from sparc.optim import apply_grads, global_grad_norm
from sparc.queries import (
    cw_distance,
    cw_distance_and_grad,
    log_exp_query,
    log_exp_query_and_grad,
)

_EXAMPLES = Path(__file__).resolve().parent
_EXAMPLE_PCS = _EXAMPLES / "example_pcs"
_ORIGINAL_DATASETS = _EXAMPLES / "original_datasets"
_ADVERSARIAL_DATASETS = _EXAMPLES / "adversarial_datasets"

K = 1.0
ETA_THETA = 1e-3
ETA_PHI_RATIO = 3
ETA_LAMBDA = 10.0
LAMBDA_MAX = 1000.0
LAMBDA_LEAK = 1.0
ALPHA = 1.0
METRIC_P = 1.0
SCALE_FACTOR = 1.0
WARM_START_ITERS = 5
THETA_NUM_SAMPLES = 50
THETA_SEED: int | None = None


def default_eta_phi(eta_theta: float) -> float:
    return ETA_PHI_RATIO * eta_theta


def _import_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Live plotting requires matplotlib. Install it with: pip install matplotlib"
        ) from exc
    return plt


def resolve_circuit_path(name: str) -> Path:
    path = Path(name)
    if path.is_file():
        return path.resolve()
    candidate = _EXAMPLE_PCS / name
    if candidate.is_file():
        return candidate.resolve()
    if not name.endswith(".json"):
        candidate = _EXAMPLE_PCS / f"{name}.json"
        if candidate.is_file():
            return candidate.resolve()
    choices = sorted(p.name for p in _EXAMPLE_PCS.glob("*.json"))
    raise FileNotFoundError(
        f"Circuit not found: {name!r}. "
        f"Pass a path or a basename under example_pcs/ ({', '.join(choices)})."
    )


def circuit_path_to_dataset_name(path: Path) -> str:
    stem = path.stem
    if stem.startswith("hclt_"):
        rest = stem[len("hclt_") :]
        for sep in ("_blocksize", "_seed"):
            if sep in rest:
                return rest.split(sep)[0]
        return rest.split("_")[0]
    return stem


def load_csv_data(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", dtype=np.int32)


def resolve_eval_datasets(dataset_name: str, dataset_k: int) -> tuple[np.ndarray, np.ndarray]:
    original_path = _ORIGINAL_DATASETS / dataset_name / f"{dataset_name}.test.data"
    adversarial_path = _ADVERSARIAL_DATASETS / f"{dataset_name}_K{dataset_k}.data"
    missing = [p for p in (original_path, adversarial_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Evaluation dataset(s) not found:\n"
            + "\n".join(f"  {p}" for p in missing)
        )
    return load_csv_data(original_path), load_csv_data(adversarial_path)


def mean_log_likelihood(circuit: Circuit, rows: np.ndarray) -> float:
    return float(circuit.compile().log_likelihood(rows).mean())


class LiveLikelihoodPlot:
    """Live-updating chart of mean log-likelihood on original vs adversarial test data."""

    def __init__(self, dataset_name: str, dataset_k: int) -> None:
        plt = _import_matplotlib()
        self._plt = plt
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        (self._line_orig,) = self.ax.plot(
            [], [], label=f"original test ({dataset_name})"
        )
        (self._line_adv,) = self.ax.plot(
            [], [], label=f"adversarial K={dataset_k} ({dataset_name})"
        )
        self.ax.set_xlabel("iteration")
        self.ax.set_ylabel("mean log-likelihood")
        self.ax.legend(loc="best")
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()

        self._iters: list[int] = []
        self._orig_lls: list[float] = []
        self._adv_lls: list[float] = []

    def update(self, it: int, orig_ll: float, adv_ll: float) -> None:
        self._iters.append(it)
        self._orig_lls.append(orig_ll)
        self._adv_lls.append(adv_ll)
        self._line_orig.set_data(self._iters, self._orig_lls)
        self._line_adv.set_data(self._iters, self._adv_lls)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self._plt.pause(0.05)

    def hold_open(self) -> None:
        self._plt.ioff()
        self._plt.show()


def combine_phi_grads(logexp_grads, cw_grads, lam):
    """Normalized convex combination of the two phi descent directions."""
    n_e = global_grad_norm(logexp_grads)
    n_c = global_grad_norm(cw_grads)
    s_e = 1.0 / n_e if n_e > 0.0 else 0.0
    s_c = 1.0 / n_c if n_c > 0.0 else 0.0
    w = lam / (1.0 + lam)
    c_e = (1.0 - w) * s_e
    c_c = w * s_c

    sum_g, cat_g = {}, {}
    for out, le_d, cw_d in (
        (sum_g, logexp_grads.sum_grads, cw_grads.sum_grads),
        (cat_g, logexp_grads.cat_grads, cw_grads.cat_grads),
    ):
        for nid in set(le_d) | set(cw_d):
            le = np.asarray(le_d.get(nid, 0.0), dtype=np.float64)
            cw = np.asarray(cw_d.get(nid, 0.0), dtype=np.float64)
            out[nid] = c_e * le + c_c * cw
    return (sum_g, cat_g)


def _eval_and_report(
    p_theta: Circuit,
    it: int,
    *,
    original_data: np.ndarray,
    adversarial_data: np.ndarray,
    plotter: LiveLikelihoodPlot | None,
) -> tuple[float, float]:
    orig_ll = mean_log_likelihood(p_theta, original_data)
    adv_ll = mean_log_likelihood(p_theta, adversarial_data)
    print(f"  eval iter {it:3d}: orig_test_ll={orig_ll:.6f}  adv_test_ll={adv_ll:.6f}")
    if plotter is not None:
        plotter.update(it, orig_ll, adv_ll)
    return orig_ll, adv_ll


GradLike = tuple[dict, dict] | object


def _grad_dicts(grads: GradLike) -> tuple[dict, dict]:
    if hasattr(grads, "sum_grads"):
        return grads.sum_grads, grads.cat_grads
    return grads


def _merge_grad_dicts(
    sum_a: dict, cat_a: dict, sum_b: dict, cat_b: dict, *, sign_b: float = 1.0
) -> tuple[dict, dict]:
    sum_out, cat_out = {}, {}
    for out, a, b in (
        (sum_out, sum_a, sum_b),
        (cat_out, cat_a, cat_b),
    ):
        for nid in set(a) | set(b):
            va = np.asarray(a.get(nid, 0.0), dtype=np.float64)
            vb = np.asarray(b.get(nid, 0.0), dtype=np.float64)
            out[nid] = va + sign_b * vb
    return sum_out, cat_out


def grad_scale(grads: GradLike, c: float) -> tuple[dict, dict]:
    sum_g, cat_g = _grad_dicts(grads)
    return (
        {nid: c * np.asarray(v, dtype=np.float64) for nid, v in sum_g.items()},
        {nid: c * np.asarray(v, dtype=np.float64) for nid, v in cat_g.items()},
    )


def grad_sub(g_a: GradLike, g_b: GradLike) -> tuple[dict, dict]:
    sum_a, cat_a = _grad_dicts(g_a)
    sum_b, cat_b = _grad_dicts(g_b)
    return _merge_grad_dicts(sum_a, cat_a, sum_b, cat_b, sign_b=-1.0)


def ogda_direction(g_curr: GradLike, g_prev: GradLike | None, *, use_ogda: bool) -> tuple[dict, dict]:
    if not use_ogda or g_prev is None:
        return _grad_dicts(g_curr)
    scaled_curr = grad_scale(g_curr, 2.0)
    return grad_sub(scaled_curr, g_prev)


def ogda_scalar(curr: float, prev: float | None, *, use_ogda: bool) -> float:
    if not use_ogda or prev is None:
        return curr
    return 2.0 * curr - prev


def compute_player_grads(
    p_theta: Circuit,
    q_phi: Circuit,
    p_hat: Circuit,
    lam: float,
    *,
    k: float,
    cw_kw: dict,
    theta_num_samples: int,
    theta_seed: int | None,
    skip_theta: bool = False,
) -> tuple[object | None, tuple[dict, dict], float, float]:
    """Return (grad_theta, phi_grads, violation, cw_val)."""
    _, _, grad_phi_logexp = log_exp_query_and_grad(p_theta, q_phi)
    cw_val, cw_grads = cw_distance_and_grad(p_hat, q_phi, **cw_kw)
    phi_grads = combine_phi_grads(grad_phi_logexp, cw_grads, lam)
    violation = ALPHA * (cw_val - k)

    grad_theta = None
    if not skip_theta:
        q_samples = q_phi.sample(theta_num_samples, seed=theta_seed)
        _, grad_theta = p_theta.mean_log_likelihood_and_grad(q_samples)

    return grad_theta, phi_grads, violation, cw_val


@dataclass
class OgdaState:
    prev_theta_grad: object | None = None
    prev_phi_grad: tuple[dict, dict] | None = None
    prev_violation: float | None = None


def update_lambda(lam: float, violation: float, *, eta_lambda: float) -> float:
    return min(
        LAMBDA_MAX,
        max(0.0, LAMBDA_LEAK * lam + eta_lambda * violation),
    )


def ogda_step(
    p_theta: Circuit,
    q_phi: Circuit,
    p_hat: Circuit,
    lam: float,
    state: OgdaState,
    *,
    k: float,
    cw_kw: dict,
    eta_theta: float,
    eta_phi: float,
    eta_lambda: float,
    theta_num_samples: int,
    theta_seed: int | None,
    use_ogda: bool,
    update_theta: bool = True,
) -> tuple[float, float, float]:
    """One OGDA step; return (lam, log_e, cw)."""
    grad_theta, phi_grads, violation, cw_val = compute_player_grads(
        p_theta,
        q_phi,
        p_hat,
        lam,
        k=k,
        cw_kw=cw_kw,
        theta_num_samples=theta_num_samples,
        theta_seed=theta_seed,
        skip_theta=not update_theta,
    )

    phi_dir = ogda_direction(phi_grads, state.prev_phi_grad, use_ogda=use_ogda)
    apply_grads(q_phi, phi_dir, eta_phi, ascent=False)

    lam_dir = ogda_scalar(violation, state.prev_violation, use_ogda=use_ogda)
    lam = update_lambda(lam, lam_dir, eta_lambda=eta_lambda)

    if update_theta and grad_theta is not None:
        theta_dir = ogda_direction(grad_theta, state.prev_theta_grad, use_ogda=use_ogda)
        apply_grads(p_theta, theta_dir, eta_theta, ascent=True)
        state.prev_theta_grad = grad_theta

    state.prev_phi_grad = phi_grads
    state.prev_violation = violation

    log_e = log_exp_query(p_theta, q_phi)
    return lam, log_e, cw_val


def run_dro_ogda(
    p_hat,
    *,
    k=K,
    num_iters=20,
    eta_theta=ETA_THETA,
    eta_phi: float | None = None,
    eta_lambda=ETA_LAMBDA,
    warm_start_iters=WARM_START_ITERS,
    theta_num_samples=THETA_NUM_SAMPLES,
    theta_seed=THETA_SEED,
    metric_p=METRIC_P,
    scale_factor=SCALE_FACTOR,
    use_ogda=True,
    eval_every: int | None = None,
    original_data: np.ndarray | None = None,
    adversarial_data: np.ndarray | None = None,
    plotter: LiveLikelihoodPlot | None = None,
):
    if eta_phi is None:
        eta_phi = default_eta_phi(eta_theta)

    cw_kw = dict(metric_p=metric_p, scale_factor=scale_factor)

    p_theta = p_hat.clone()
    q_phi = p_hat.clone()
    lam = 0.0
    state = OgdaState()

    do_eval = original_data is not None and adversarial_data is not None

    mode = "OGDA" if use_ogda else "GDA"
    print(
        f"initial ({mode}): log(E)={log_exp_query(p_theta, q_phi):.6f}  "
        f"CW={cw_distance(p_hat, q_phi, **cw_kw):.6f}  "
        f"eta_theta={eta_theta:g}  eta_phi={eta_phi:g}"
    )

    if do_eval:
        _eval_and_report(
            p_theta,
            0,
            original_data=original_data,
            adversarial_data=adversarial_data,
            plotter=plotter,
        )

    step_kw = dict(
        k=k,
        cw_kw=cw_kw,
        eta_theta=eta_theta,
        eta_phi=eta_phi,
        eta_lambda=eta_lambda,
        theta_num_samples=theta_num_samples,
        theta_seed=theta_seed,
        use_ogda=use_ogda,
    )

    if warm_start_iters > 0:
        print(f"  warm start: {warm_start_iters} Q-only {mode} iteration(s)")
        for w_iter in range(1, warm_start_iters + 1):
            lam, log_e, cw = ogda_step(
                p_theta,
                q_phi,
                p_hat,
                lam,
                state,
                update_theta=False,
                **step_kw,
            )
            print(
                f"  [warm-start {w_iter}/{warm_start_iters}] "
                f"log(E)={log_e:.6f}  CW={cw:.6f}  "
                f"violation={cw - k:+.6f}  lambda={lam:.4f}"
            )

    for it in range(1, num_iters + 1):
        lam, log_e, cw = ogda_step(
            p_theta,
            q_phi,
            p_hat,
            lam,
            state,
            update_theta=True,
            **step_kw,
        )
        print(
            f"  iter {it:3d}: log(E)={log_e:.6f}  CW={cw:.6f}  "
            f"violation={cw - k:+.6f}  lambda={lam:.4f}"
        )

        if do_eval and eval_every is not None and it % eval_every == 0:
            _eval_and_report(
                p_theta,
                it,
                original_data=original_data,
                adversarial_data=adversarial_data,
                plotter=plotter,
            )

    return p_theta, lam


def main():
    parser = argparse.ArgumentParser(
        description="Robustify a PC with sample-based DRO using OGDA."
    )
    parser.add_argument(
        "circuit",
        help="Basename or path to a gcw-circuit-v1 JSON (e.g. plants.json or adult)",
    )
    parser.add_argument("--k", type=float, default=K, help="CW-ball radius")
    parser.add_argument("--iters", type=int, default=20, help="OGDA iterations")
    parser.add_argument(
        "--output", "-o",
        help="Optional path to save the robustified circuit (gcw-circuit-v1 JSON)",
    )
    parser.add_argument(
        "--dataset-k",
        type=int,
        choices=(1, 3, 5),
        default=None,
        help="Adversarial dataset K (1, 3, or 5). When set, evaluate and plot "
        "mean log-likelihood on original and adversarial test data every "
        "--eval-every iterations.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=5,
        metavar="N",
        help="Evaluate test log-likelihood every N iterations (default: 5). "
        "Only used when --dataset-k is set.",
    )
    parser.add_argument(
        "--warm-start-iters",
        type=int,
        default=WARM_START_ITERS,
        help=f"Q-only iterations before theta updates (default: {WARM_START_ITERS}).",
    )
    parser.add_argument(
        "--theta-num-samples",
        type=int,
        default=THETA_NUM_SAMPLES,
        help=f"Samples from Q_phi per theta update (default: {THETA_NUM_SAMPLES}).",
    )
    parser.add_argument(
        "--theta-seed",
        type=int,
        default=None,
        help="RNG seed for theta sampling (omit for fresh MC noise each iter).",
    )
    parser.add_argument(
        "--eta-theta",
        type=float,
        default=ETA_THETA,
        help=f"Theta (P) learning rate (default: {ETA_THETA:g}).",
    )
    parser.add_argument(
        "--eta-phi",
        type=float,
        default=None,
        help=f"Phi (Q) learning rate (default: {ETA_PHI_RATIO:g}x --eta-theta).",
    )
    parser.add_argument(
        "--eta-lambda",
        type=float,
        default=ETA_LAMBDA,
        help=f"Dual (lambda) step size (default: {ETA_LAMBDA:g}).",
    )
    parser.add_argument(
        "--gda",
        action="store_true",
        help="Disable OGDA optimism (plain GDA ablation baseline).",
    )
    args = parser.parse_args()

    if args.dataset_k is not None and args.eval_every <= 0:
        parser.error("--eval-every must be a positive integer")
    if args.warm_start_iters < 0:
        parser.error("--warm-start-iters must be non-negative")
    if args.theta_num_samples < 1:
        parser.error("--theta-num-samples must be at least 1")
    if args.eta_theta <= 0:
        parser.error("--eta-theta must be positive")
    if args.eta_phi is not None and args.eta_phi <= 0:
        parser.error("--eta-phi must be positive")
    if args.eta_lambda <= 0:
        parser.error("--eta-lambda must be positive")

    eta_phi = args.eta_phi if args.eta_phi is not None else default_eta_phi(args.eta_theta)

    path = resolve_circuit_path(args.circuit)
    print(f"loading {path.name} from {path.parent}")
    p_hat = Circuit.load(path)
    print(f"  nodes in scope: {len(p_hat.root.scope_as_list())}")

    original_data = adversarial_data = None
    plotter = None
    if args.dataset_k is not None:
        dataset_name = circuit_path_to_dataset_name(path)
        print(f"loading eval datasets for {dataset_name!r} (K={args.dataset_k})")
        original_data, adversarial_data = resolve_eval_datasets(
            dataset_name, args.dataset_k
        )
        print(
            f"  original test: {len(original_data)} rows, "
            f"adversarial K={args.dataset_k}: {len(adversarial_data)} rows"
        )
        plotter = LiveLikelihoodPlot(dataset_name, args.dataset_k)

    p_theta, lam = run_dro_ogda(
        p_hat,
        k=args.k,
        num_iters=args.iters,
        eta_theta=args.eta_theta,
        eta_phi=eta_phi,
        eta_lambda=args.eta_lambda,
        warm_start_iters=args.warm_start_iters,
        theta_num_samples=args.theta_num_samples,
        theta_seed=args.theta_seed,
        use_ogda=not args.gda,
        eval_every=args.eval_every if args.dataset_k is not None else None,
        original_data=original_data,
        adversarial_data=adversarial_data,
        plotter=plotter,
    )

    if args.output:
        out = Path(args.output)
        p_theta.save(out)
        print(f"\nsaved robustified circuit to {out.resolve()}")

    print(f"\nfinal lambda={lam:.4f}")

    if plotter is not None:
        plotter.hold_open()


if __name__ == "__main__":
    main()
