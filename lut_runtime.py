"""
Runtime lookup-table loaders for the two iterative heat metrics.

Drop-in replacements for the iterative functions, with identical signatures and
units, backed by the regular-grid LUTs built by `lookup_tables.py`:

    get_aorc_romps_heat_index(air_temp, specific_humid)            -> degF
    get_aorc_wbt(air_temp, specific_humid, surface_pressure)       -> degF

Each replaces a per-point root-find with a (bi/tri)linear interpolation. Tables
are loaded once per process and cached. Inputs outside the grid are clamped to
the nearest edge; NaN inputs pass through as NaN (the pipeline masks with
`.where`, so out-of-region cells must stay NaN rather than clamp to an edge).
"""

import os
import numpy as np
import numba as nb

_DIR = os.path.dirname(os.path.abspath(__file__))
ROMPS_LUT_PATH = os.path.join(_DIR, "romps_hi_lut.npz")
WBT_LUT_PATH = os.path.join(_DIR, "wbt_lut.npz")

# Cached per process: filled on first call so importing never needs the files.
_ROMPS = None  # (ta0, dta, q0, dq, table[ta, q])
_WBT = None    # (ta0, dta, q0, dq, p0, dp, table[ta, q, p])


def _origin_step(a):
    """Origin and (uniform) step of a regular axis, as float32."""
    return np.float32(a[0]), np.float32(a[1] - a[0])


def _load_romps():
    global _ROMPS
    if _ROMPS is None:
        if not os.path.isfile(ROMPS_LUT_PATH):
            raise FileNotFoundError(
                f"Missing {ROMPS_LUT_PATH}; run `python lookup_tables.py` first."
            )
        with np.load(ROMPS_LUT_PATH) as d:
            ta0, dta = _origin_step(d["air_temp"])
            q0, dq = _origin_step(d["specific_humid"])
            table = np.ascontiguousarray(d["heat_index"], np.float32)
        _ROMPS = (ta0, dta, q0, dq, table)
    return _ROMPS


def _load_wbt():
    global _WBT
    if _WBT is None:
        if not os.path.isfile(WBT_LUT_PATH):
            raise FileNotFoundError(
                f"Missing {WBT_LUT_PATH}; run `python lookup_tables.py` first."
            )
        with np.load(WBT_LUT_PATH) as d:
            ta0, dta = _origin_step(d["air_temp"])
            q0, dq = _origin_step(d["specific_humid"])
            p0, dp = _origin_step(d["pressure"])
            table = np.ascontiguousarray(d["wet_bulb"], np.float32)
        _WBT = (ta0, dta, q0, dq, p0, dp, table)
    return _WBT


@nb.njit(parallel=True, fastmath=True, cache=True)
def _interp2d(x, y, x0, dx, y0, dy, t):
    nx, ny = t.shape
    fnx = np.float32(nx - 1)
    fny = np.float32(ny - 1)
    out = np.empty(x.size, np.float32)
    for k in nb.prange(x.size):
        xv = x[k]
        yv = y[k]
        if xv != xv or yv != yv:                 # NaN passthrough
            out[k] = np.nan
            continue
        gx = (xv - x0) / dx
        gy = (yv - y0) / dy
        gx = min(max(gx, np.float32(0.0)), fnx)  # clamp to grid
        gy = min(max(gy, np.float32(0.0)), fny)
        i = min(int(gx), nx - 2)
        j = min(int(gy), ny - 2)
        tx = gx - i
        ty = gy - j
        out[k] = (
            t[i, j] * (1.0 - tx) * (1.0 - ty)
            + t[i + 1, j] * tx * (1.0 - ty)
            + t[i, j + 1] * (1.0 - tx) * ty
            + t[i + 1, j + 1] * tx * ty
        )
    return out


@nb.njit(parallel=True, fastmath=True, cache=True)
def _interp3d(x, y, z, x0, dx, y0, dy, z0, dz, t):
    nx, ny, nz = t.shape
    fnx = np.float32(nx - 1)
    fny = np.float32(ny - 1)
    fnz = np.float32(nz - 1)
    out = np.empty(x.size, np.float32)
    for k in nb.prange(x.size):
        xv = x[k]
        yv = y[k]
        zv = z[k]
        if xv != xv or yv != yv or zv != zv:     # NaN passthrough
            out[k] = np.nan
            continue
        gx = min(max((xv - x0) / dx, np.float32(0.0)), fnx)
        gy = min(max((yv - y0) / dy, np.float32(0.0)), fny)
        gz = min(max((zv - z0) / dz, np.float32(0.0)), fnz)
        i = min(int(gx), nx - 2)
        j = min(int(gy), ny - 2)
        m = min(int(gz), nz - 2)
        tx = gx - i
        ty = gy - j
        tz = gz - m
        c00 = t[i, j, m] * (1.0 - tx) + t[i + 1, j, m] * tx
        c01 = t[i, j, m + 1] * (1.0 - tx) + t[i + 1, j, m + 1] * tx
        c10 = t[i, j + 1, m] * (1.0 - tx) + t[i + 1, j + 1, m] * tx
        c11 = t[i, j + 1, m + 1] * (1.0 - tx) + t[i + 1, j + 1, m + 1] * tx
        c0 = c00 * (1.0 - ty) + c10 * ty
        c1 = c01 * (1.0 - ty) + c11 * ty
        out[k] = c0 * (1.0 - tz) + c1 * tz
    return out


def get_aorc_romps_heat_index(air_temp, specific_humid):
    """LUT version of romps_heat_index.get_aorc_romps_heat_index (degF)."""
    ta0, dta, q0, dq, table = _load_romps()
    shape = np.asarray(air_temp).shape
    x = np.ascontiguousarray(air_temp, np.float32).ravel()
    y = np.ascontiguousarray(specific_humid, np.float32).ravel()
    return _interp2d(x, y, ta0, dta, q0, dq, table).reshape(shape)


def get_aorc_wbt(air_temp, specific_humid, surface_pressure):
    """LUT version of metrics.get_aorc_wbt (degF)."""
    ta0, dta, q0, dq, p0, dp, table = _load_wbt()
    shape = np.asarray(air_temp).shape
    x = np.ascontiguousarray(air_temp, np.float32).ravel()
    y = np.ascontiguousarray(specific_humid, np.float32).ravel()
    z = np.ascontiguousarray(surface_pressure, np.float32).ravel()
    return _interp3d(x, y, z, ta0, dta, q0, dq, p0, dp, table).reshape(shape)
