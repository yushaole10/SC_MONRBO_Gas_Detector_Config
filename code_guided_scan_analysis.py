"""Code-guided asynchronous scan stress test.

This script does not claim to reproduce physical sensor dynamics. It maps the
11-command, 1-s Qt polling schedule found in the supplied device source onto
the existing synthetic chamber and quantifies the effect of stale, zero-order-
held channels relative to the manuscript's ideal synchronized-vector model.
"""

from pathlib import Path
import numpy as np
import pandas as pd

import gas_optimization_experiment as g


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experiment"

# Query slots recovered from MainWindow::initSensorQueryCommands().  Slot 0
# (address 0x20) updates CO2 plus temperature/RH; slots 1--10 query CO, O2,
# H2S, CO2, NH3, NOx, CH4, HCHO, SO2 and HCN, respectively.
GAS_UPDATES = {
    0: (3,),
    1: (2,),
    2: (0,),
    3: (1,),
    4: (3,),
    5: (4,),
    6: (5,),
    7: (6,),
    8: (9,),
    9: (8,),
    10: (7,),
}
SCAN_LENGTH = 11


def sequential_hold(x: np.ndarray) -> np.ndarray:
    """Apply the observed query schedule to [sequence,time,12-feature] data."""
    held = np.empty_like(x)
    state = x[:, 0, :].copy()
    for t in range(x.shape[1]):
        slot = t % SCAN_LENGTH
        for gas_idx in GAS_UPDATES[slot]:
            state[:, gas_idx] = x[:, t, gas_idx]
        if slot == 0:
            state[:, 10:12] = x[:, t, 10:12]
        held[:, t, :] = state
    return held


def evaluate(z: np.ndarray, firmware: bool, asynchronous: bool) -> dict:
    loglam, alpha, thr, k, tau = g.decode(z)
    w = None if firmware else g.ridge(loglam)
    vals = []
    for _, x, y in g.SCENARIOS:
        source = sequential_hold(x) if asynchronous else x
        xs = source[:, ::tau]
        ys = y[:, ::tau]
        est = xs[..., :10].copy() if firmware else np.maximum(0, xs @ w)
        if alpha > 0:
            for j in range(1, est.shape[1]):
                est[:, j] = alpha * est[:, j - 1] + (1 - alpha) * est[:, j]
        truth = ys >= 1.0
        raw = est >= thr
        alarm = np.zeros_like(raw)
        run = np.zeros((raw.shape[0], raw.shape[2]), dtype=int)
        for j in range(raw.shape[1]):
            run = np.where(raw[:, j], run + 1, 0)
            alarm[:, j] = run >= k
        fnr = np.logical_and(truth, ~alarm).sum() / max(1, truth.sum())
        fpr = np.logical_and(~truth, alarm).sum() / max(1, (~truth).sum())
        delays = []
        for s in range(truth.shape[0]):
            for gas in range(10):
                idx = np.flatnonzero(truth[s, :, gas])
                if idx.size:
                    detected = np.flatnonzero(alarm[s, idx[0] :, gas])
                    delays.append((detected[0] if detected.size else truth.shape[1] - idx[0]) * tau)
        rmse = np.sqrt(np.mean((est - ys) ** 2))
        vals.append((rmse, fnr, fpr, np.mean(delays) if delays else 0.0))
    worst = np.asarray(vals).max(axis=0)
    return {
        "rmse": worst[0],
        "fnr": worst[1],
        "fpr": worst[2],
        "delay_s": worst[3],
        "tau_model_s": tau,
        "query_interval_s": 1,
        "scan_commands": SCAN_LENGTH,
        "nominal_non_co2_refresh_s": SCAN_LENGTH,
        "maximum_channel_staleness_s": SCAN_LENGTH - 1,
    }


def main() -> None:
    firmware_z = np.r_[[-5.0, 0.0], np.ones(10), 1.0, 1.0]
    archive = np.loadtxt(OUT / "pareto_archive.csv", delimiter=",", skiprows=1)
    x, f, v = archive[:, : g.D], archive[:, g.D : g.D + 3], archive[:, -1]
    representative_z = x[g.best_compromise(x, f, v)]

    rows = []
    for name, z, firmware in (
        ("logic-only fixed threshold", firmware_z, True),
        ("representative SC-MONRBO", representative_z, False),
    ):
        for acquisition in ("ideal synchronized vector", "code-guided sequential ZOH"):
            row = {
                "configuration": name,
                "acquisition_model": acquisition,
                **evaluate(z, firmware, acquisition.startswith("code-guided")),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "code_guided_scan_stress.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
