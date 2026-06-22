"""Robustify a saved PC via distributionally-robust optimization (DRO).



Loads a ``gcw-circuit-v1`` JSON from ``examples/example_pcs/`` and solves



    max_theta  min_{Q : CW(P_hat, Q) <= k}  log E_Q[P_theta]



via a Lagrangian inner loop on Q (dual variable lambda for the CW-ball

constraint) and a sample-based projected gradient ascent on theta.



The inner adversary (phi, lambda) still optimizes ``log E_Q[P]``. The outer

theta update ascends a Monte Carlo estimate of ``E_Q[log P_theta]`` by drawing

samples from Q_phi and differentiating the mean log-likelihood under P_theta

(rather than the analytical gradient of ``log E_Q[P_theta]``).



    python examples/robustify.py plants.json

    python examples/robustify.py example_pcs/adult.json --iters 50 --output robust_plants.json

    python examples/robustify.py plants.json --dataset-k 3 --eval-every 5

"""



from __future__ import annotations



import argparse

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

NUM_Q_ITERS = 5

ETA_THETA = 1e-3

ETA_PHI = 1e-3

ETA_LAMBDA = 20.0

LAMBDA_MAX = 1000.0

LAMBDA_LEAK = 1.0

ALPHA = 1.0

METRIC_P = 1.0

SCALE_FACTOR = 1.0

WARM_START_ITERS = 5

THETA_NUM_SAMPLES = 2000

THETA_SEED: int | None = None





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





def update_q_phi(

    p_theta: Circuit,

    q_phi: Circuit,

    p_hat: Circuit,

    lam: float,

    *,

    k: float,

    cw_kw: dict,

    eta_phi: float,

    eta_lambda: float,

) -> float:

    """Run NUM_Q_ITERS inner phi/lambda updates; leave p_theta unchanged."""

    for _ in range(NUM_Q_ITERS):

        _, _, grad_phi = log_exp_query_and_grad(p_theta, q_phi)

        cw_val, cw_grads = cw_distance_and_grad(p_hat, q_phi, **cw_kw)



        violation = ALPHA * (cw_val - k)

        lam = min(

            LAMBDA_MAX,

            max(0.0, LAMBDA_LEAK * lam + eta_lambda * violation),

        )

        phi_grads = combine_phi_grads(grad_phi, cw_grads, lam)

        apply_grads(q_phi, phi_grads, eta_phi, ascent=False)

    return lam





def update_p_theta(

    p_theta: Circuit,

    q_phi: Circuit,

    *,

    eta_theta: float,

    theta_num_samples: int,

    theta_seed: int | None,

) -> None:

    """One sample-based theta ascent step on E_Q[log P]."""

    q_samples = q_phi.sample(theta_num_samples, seed=theta_seed)

    _, grad_theta = p_theta.mean_log_likelihood_and_grad(q_samples)

    apply_grads(p_theta, grad_theta, eta_theta, ascent=True)





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





def run_dro(

    p_hat,

    *,

    k=K,

    num_iters=20,

    eta_theta=ETA_THETA,

    eta_phi=ETA_PHI,

    eta_lambda=ETA_LAMBDA,

    warm_start_iters=WARM_START_ITERS,

    theta_num_samples=THETA_NUM_SAMPLES,

    theta_seed=THETA_SEED,

    metric_p=METRIC_P,

    scale_factor=SCALE_FACTOR,

    eval_every: int | None = None,

    original_data: np.ndarray | None = None,

    adversarial_data: np.ndarray | None = None,

    plotter: LiveLikelihoodPlot | None = None,

):

    cw_kw = dict(metric_p=metric_p, scale_factor=scale_factor)



    p_theta = p_hat.clone()

    q_phi = p_hat.clone()

    lam = 0.0

    do_eval = original_data is not None and adversarial_data is not None



    print(f"initial: log(E)={log_exp_query(p_theta, q_phi):.6f}  "

          f"CW={cw_distance(p_hat, q_phi, **cw_kw):.6f}")



    if do_eval:

        _eval_and_report(

            p_theta,

            0,

            original_data=original_data,

            adversarial_data=adversarial_data,

            plotter=plotter,

        )



    q_update_kw = dict(

        k=k,

        cw_kw=cw_kw,

        eta_phi=eta_phi,

        eta_lambda=eta_lambda,

    )



    if warm_start_iters > 0:

        print(f"  warm start: {warm_start_iters} Q-only iteration(s)")

        for w_iter in range(1, warm_start_iters + 1):

            lam = update_q_phi(p_theta, q_phi, p_hat, lam, **q_update_kw)

            log_e = log_exp_query(p_theta, q_phi)

            cw = cw_distance(p_hat, q_phi, **cw_kw)

            print(

                f"  [warm-start {w_iter}/{warm_start_iters}] "

                f"log(E)={log_e:.6f}  CW={cw:.6f}  "

                f"violation={cw - k:+.6f}  lambda={lam:.4f}"

            )



    for it in range(1, num_iters + 1):

        lam = update_q_phi(p_theta, q_phi, p_hat, lam, **q_update_kw)

        update_p_theta(

            p_theta,

            q_phi,

            eta_theta=eta_theta,

            theta_num_samples=theta_num_samples,

            theta_seed=theta_seed,

        )



        log_e = log_exp_query(p_theta, q_phi)

        cw = cw_distance(p_hat, q_phi, **cw_kw)

        print(f"  iter {it:3d}: log(E)={log_e:.6f}  CW={cw:.6f}  "

              f"violation={cw - k:+.6f}  lambda={lam:.4f}")



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

    parser = argparse.ArgumentParser(description="Robustify a PC with sample-based DRO.")

    parser.add_argument(

        "circuit",

        help="Basename or path to a gcw-circuit-v1 JSON (e.g. plants.json or adult)",

    )

    parser.add_argument("--k", type=float, default=K, help="CW-ball radius")

    parser.add_argument("--iters", type=int, default=20, help="outer theta steps")

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

        help="RNG seed for theta sampling (omit for fresh MC noise each outer iter).",

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

        default=ETA_PHI,

        help=f"Phi (Q) learning rate (default: {ETA_PHI:g}).",

    )

    parser.add_argument(

        "--eta-lambda",

        type=float,

        default=ETA_LAMBDA,

        help=f"Dual (lambda) step size (default: {ETA_LAMBDA:g}).",

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

    if args.eta_phi <= 0:

        parser.error("--eta-phi must be positive")

    if args.eta_lambda <= 0:

        parser.error("--eta-lambda must be positive")



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



    p_theta, lam = run_dro(

        p_hat,

        k=args.k,

        num_iters=args.iters,

        eta_theta=args.eta_theta,

        eta_phi=args.eta_phi,

        eta_lambda=args.eta_lambda,

        warm_start_iters=args.warm_start_iters,

        theta_num_samples=args.theta_num_samples,

        theta_seed=args.theta_seed,

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


