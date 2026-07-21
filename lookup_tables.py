"""
Generate lookup tables (LUTs) for the two *iterative* heat metrics so the
pipeline can replace per-point root-finding with fast interpolation.

Only these two are tabulated:
  * Romps (2022) heat index -- bisection solve, 2D in (air_temp, specific_humid)
  * Wet-bulb temperature    -- Newton solve, 3D in (air_temp, specific_humid, pressure)

The other four metrics (NWS heat index, apparent temperature, humidex, sWBGT)
are closed-form and cheaper to evaluate directly than to look up, so they are
not tabulated.

Each table is saved as a self-describing compressed .npz holding the metric
grid plus its axis coordinates -- exactly what a RegularGridInterpolator needs
at runtime. Axes match the units the runtime functions already take, so the
lookup is a direct interpolation on the raw data variables (no conversion).
"""

import numpy as np
import numba as nb

from metrics import get_aorc_wbt
from romps_heat_index import get_aorc_romps_heat_index

# --- Axis definitions (tune ranges/steps here) ------------------------------
TA_MIN, TA_MAX, TA_STEP = 240.0, 330.0, 0.1        # air temperature [K]
Q_MIN, Q_MAX = 0.0, 0.04                           # specific humidity [kg/kg]
Q_STEP_HI = 1.0e-4                                 #   Romps HI grid step
Q_STEP_WBT = 2.0e-4                                #   WBT grid step (weaker sens.)
P_MIN, P_MAX, P_STEP = 60_000.0, 105_000.0, 1_000.0  # surface pressure [Pa]


def axis(lo: float, hi: float, step: float) -> np.ndarray:
    """Inclusive, evenly spaced float32 axis (regular grid -> O(1) indexing)."""
    n = int(round((hi - lo) / step)) + 1
    return np.linspace(lo, hi, n, dtype=np.float32)


@nb.njit(parallel=True, cache=True, fastmath=True)
def _fill_romps(ta: np.ndarray, q: np.ndarray) -> np.ndarray:
    out = np.empty((ta.size, q.size), dtype=np.float32)
    for i in nb.prange(ta.size):
        for j in range(q.size):
            out[i, j] = get_aorc_romps_heat_index(ta[i], q[j])
    return out


@nb.njit(parallel=True, cache=True, fastmath=True)
def _fill_wbt(ta: np.ndarray, q: np.ndarray, p: np.ndarray) -> np.ndarray:
    out = np.empty((ta.size, q.size, p.size), dtype=np.float32)
    for i in nb.prange(ta.size):
        for j in range(q.size):
            for k in range(p.size):
                out[i, j, k] = get_aorc_wbt(ta[i], q[j], p[k])
    return out


def main() -> None:
    ta = axis(TA_MIN, TA_MAX, TA_STEP)

    # Romps heat index -- 2D (air_temp, specific_humid) -> heat index [degF]
    q_hi = axis(Q_MIN, Q_MAX, Q_STEP_HI)
    romps = _fill_romps(ta, q_hi)
    np.savez_compressed(
        "romps_hi_lut.npz", air_temp=ta, specific_humid=q_hi, heat_index=romps
    )
    print(f"romps_hi_lut.npz  grid={romps.shape}  {romps.nbytes / 1e6:.1f} MB raw")

    # Wet-bulb temp -- 3D (air_temp, specific_humid, pressure) -> wbt [degF]
    q_wbt = axis(Q_MIN, Q_MAX, Q_STEP_WBT)
    p = axis(P_MIN, P_MAX, P_STEP)
    wbt = _fill_wbt(ta, q_wbt, p)
    np.savez_compressed(
        "wbt_lut.npz", air_temp=ta, specific_humid=q_wbt, pressure=p, wet_bulb=wbt
    )
    print(f"wbt_lut.npz       grid={wbt.shape}  {wbt.nbytes / 1e6:.1f} MB raw")


if __name__ == "__main__":
    main()
