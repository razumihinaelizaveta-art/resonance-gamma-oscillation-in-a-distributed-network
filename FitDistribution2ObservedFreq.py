import sys, json
import numpy as np
from scipy.integrate import quad
from scipy.optimize import differential_evolution
from scipy.special import gamma as gamma_fn
from math import sqrt, pi

# ------------------------------------------------------------------
# observations: [left_distance, right_distance, mean_connectivity]
# ------------------------------------------------------------------

if len(sys.argv) < 2:
    exit("Need input JSON with experimental data")

try:
    with open(sys.argv[1]) as fd:
        j = json.load(fd)
    data = np.array(j['Observed probabilities'])
except BaseException as e:
    exit(f'Cannot read `Observed probabilities` from the file `{sys.argv[1]}`: {e}')
if not data.shape[1] == 3 or data.shape[0] < 3:
    exit(f'`Observed probabilities` must have 3 columns: left,right distance boundaries, and probability. The table should have at least 3 entrances. Got:{data.shape}')
# data = np.array([
    # [0,   50,  0.35],
    # [50,  100, 0.20],
    # [100, 200, 0.01],
    # [200, 300, 1e-6]
# ], dtype=float)

a, b, y = data.T

# ------------------------------------------------------------------
# candidate connection laws p(r)
# ------------------------------------------------------------------

models = {

    "exponential": (
        lambda r, A, L:
            A * np.exp(-r / L),
        [(1e-12, 1), (1e-5, 1e5)],
        "pmax sigma".split()
    ),

    "stretched_exponential": (
        lambda r, A, L, k:
            A * np.exp(-(r / L) ** k),
        [(1e-12, 1), (1e-3, 1e4), (0.05, 10)],
        "pmax sigma power".split()
    ),

    "hill_powerlaw": (
        lambda r, A, R, k:
            A / (1 + (r / R) ** k),
        [(1e-12, 1), (1e-3, 1e4), (0.05, 20)],
        "pmax L12 power".split()
    ),

    # "weibull_survival": (
        # lambda r, A, L, k:
            # A * np.exp(-(r / L) ** k),
        # [(1e-12, 1), (1e-3, 1e4), (0.05, 10)]
    # ),

    "gamma_tail": (
        lambda r, A, k, th:
            A * np.maximum(
                1e-300,
                1 - np.array([
                    quad(
                        lambda t:
                        t**(k-1)*np.exp(-t/th) /
                        (gamma_fn(k)*th**k),
                        0, x
                    )[0]
                    for x in np.atleast_1d(r)
                ])
            ),
        [(1e-12, 1), (0.1, 20), (1e-3, 1e4)],
        "pmax alpha theta".split()
    ),

    "lognormal_tail": (
        lambda r, A, mu, sigma:
            A * 0.5 * np.vectorize(
                lambda x:
                1 - np.math.erf(
                    (np.log(max(x,1e-12)) - mu)
                    / (sigma*np.sqrt(2))
                )
            )(r),
        [(1e-12, 1), (-10, 10), (0.05, 5)],
        "pmax mu sigma".split()
    ),
}


def bin_mean(f, pars, lo, hi):
    return quad(lambda x: float(np.squeeze(f(x, *pars))),
                lo, hi, limit=200)[0] / (hi - lo)


def fit_model(name, f, bounds, params):

    def loss(pars):
        pred = np.array([
            max(bin_mean(f, pars, lo, hi), 1e-300)
            for lo, hi in zip(a, b)
        ])

        return np.mean(
            (np.log10(pred) - np.log10(y)) ** 2
        )

    res = differential_evolution(
        loss,
        bounds,
        seed=0,
        polish=True
    )

    pars = res.x

    pred = np.array([
        max(bin_mean(f, pars, lo, hi), 1e-300)
        for lo, hi in zip(a, b)
    ])

    rss = np.sum(
        (np.log10(pred) - np.log10(y)) ** 2
    )

    n = len(y)
    k = len(pars)

    aic = n * np.log(rss / n + 1e-300) + 2 * k
    bic = n * np.log(rss / n + 1e-300) + k * np.log(n)

    return {
        "model": name,
        "params": { n:v for n,v in zip(params,pars)},
        "pred": pred,
        "rmse_log10": np.sqrt(rss / n),
        "AIC": aic,
        "BIC": bic,
    }


results = []

for name, (f, bounds,params) in models.items():
    try:
        results.append(fit_model(name, f, bounds,params))
    except Exception:
        pass

results.sort(key=lambda x: x["AIC"])

best = results[0]

print("\nBEST MODEL")
print("----------")
print("name      :", best["model"])
print("params    :", best["params"])
print("RMSE(log) :", best["rmse_log10"])
print("AIC       :", best["AIC"])
print("BIC       :", best["BIC"])

j["The Best Distribution"] = {"type": best["model"] } | { n:float(best["params"][n]) for n in best["params"] }
j["Fitting errors"] ={
    "RMSE(log)"  : float(best["rmse_log10"]),
    "AIC"        : float(best["AIC"]),
    "BIC"        : float(best["BIC"])
}

with open(sys.argv[2] if len(sys.argv) > 2 else sys.argv[1],"w") as fd:
    json.dump(j,fd,indent=4)

print("\nFitted bin probabilities")
for (lo, hi, obs), pred in zip(data, best["pred"]):
    print(
        f"{lo:4.0f} - {hi:4.0f}   "
        f"obs={obs:.3e}   fit={pred:.3e}"
    )

print("\nAll models ranked by AIC:")
for r in results:
    print(
        f"{r['model']:22s}"
        f" AIC={r['AIC']:10.4f}"
        f" RMSE={r['rmse_log10']:.4f}"
        f" Parameters={r['params']}"
    )

# ============================================================
# Monte-Carlo validation of fitted model
# ============================================================

rng = np.random.default_rng(1234)

N_POINTS = 1000
N_PAIR_SAMPLES = 1000000

# random points in square
pts = rng.uniform(0, 300, size=(N_POINTS, 2))

# use fitted best model
f, _, _ = models[best["model"]]
pars = [ best["params"][n] for n in best["params"] ]

F, _, _ = models['exponential']
PARS = [ pars[0], pars[1]/pars[2] ]

# sample random pairs
i = rng.integers(0, N_POINTS, N_PAIR_SAMPLES)
j = rng.integers(0, N_POINTS, N_PAIR_SAMPLES)

mask = i != j
i = i[mask]
j = j[mask]

# distances
d = np.linalg.norm(pts[i] - pts[j], axis=1)

# connection probabilities
p = np.clip(f(d, *pars), 0, 1)
P = np.clip(F(d, *PARS), 0, 1)

# draw connections
connected = rng.random(len(d)) < p
CONNECTED = rng.random(len(d)) < P

print("\nReconstructed table")
print("-------------------")

for lo, hi, observed in data:

    m = (d >= lo) & (d < hi)

    n_pairs = m.sum()

    if n_pairs == 0:
        continue

    estimated = connected[m].mean()
    Estimated = CONNECTED[m].mean()

    print(
        f"{lo:4.0f} - {hi:4.0f}"
        f"  observed={observed:.6g}"
        f"  reconstructed={estimated:.6g}"
        f"  Reconstructed={Estimated:.6g}"
        f"  pairs={n_pairs}"
    )
