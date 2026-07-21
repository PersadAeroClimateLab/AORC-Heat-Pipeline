"""Unit tests for the runtime lookup tables in ``lut_runtime.py``.

The LUTs (built by ``lookup_tables.py``) replace the iterative Romps heat-index
bisection and the wet-bulb Newton solve with regular-grid (bi/tri)linear
interpolation. These tests build *small* tables at production resolution over a
focused, physically realistic range -- so they do not depend on the large
prebuilt ``.npz`` files -- and then verify:

  * interpolation reproduces stored grid nodes exactly,
  * LUT values match the direct iterative functions within interpolation
    tolerance across the realistic surface-meteorology envelope,
  * physical sanity: plausible bounds, monotonicity in T and humidity, the
    wet-bulb <= air-temperature inequality, and Tw -> T at saturation,
  * array semantics: shape preservation, NaN passthrough (masked cells), edge
    clamping, and scalar/array agreement.
"""
import numpy as np
import pytest

import metrics as m
import romps_heat_index as rhi
import lookup_tables as lt
import lut_runtime as L

# Test-table extent. Realistic test points sit strictly inside this range so
# they exercise interpolation, not clamping (clamping is tested separately).
TA_LO, TA_HI = 270.0, 322.0          # K
Q_LO, Q_HI = 0.0, 0.03               # kg/kg
P_LO, P_HI = 70_000.0, 103_000.0     # Pa


def _huss_from_vp_hpa(vp_hpa):
    """Inverse of ``vapor_pressure()``: specific humidity for a target vp (hPa)."""
    return 0.622 * vp_hpa / (1013.25 - 0.378 * vp_hpa)


def _huss_from_rh(temp_c, rh):
    """Specific humidity (kg/kg) for a given temperature (degC) and RH (0-1)."""
    return _huss_from_vp_hpa(rh * float(m.saturation_vapor_pressure(temp_c)))


@pytest.fixture(scope="session", autouse=True)
def _build_test_luts(tmp_path_factory):
    """Build small production-resolution LUTs and point ``lut_runtime`` at them."""
    d = tmp_path_factory.mktemp("luts")

    ta = lt.axis(TA_LO, TA_HI, 0.1)
    q_hi = lt.axis(Q_LO, Q_HI, 1.0e-4)
    romps = lt._fill_romps(ta, q_hi)
    romps_path = d / "romps_hi_lut.npz"
    np.savez(romps_path, air_temp=ta, specific_humid=q_hi, heat_index=romps)

    ta_w = lt.axis(TA_LO, TA_HI, 0.2)
    q_w = lt.axis(Q_LO, Q_HI, 2.0e-4)
    p = lt.axis(P_LO, P_HI, 1_000.0)
    wbt = lt._fill_wbt(ta_w, q_w, p)
    wbt_path = d / "wbt_lut.npz"
    np.savez(wbt_path, air_temp=ta_w, specific_humid=q_w, pressure=p, wet_bulb=wbt)

    # Redirect the module to the test tables and clear its per-process cache.
    L.ROMPS_LUT_PATH = str(romps_path)
    L.WBT_LUT_PATH = str(wbt_path)
    L._ROMPS = None
    L._WBT = None
    yield


def _romps(ta, q):
    return float(L.get_aorc_romps_heat_index(np.float32(ta), np.float32(q)))


def _wbt(ta, q, p):
    return float(L.get_aorc_wbt(np.float32(ta), np.float32(q), np.float32(p)))


# ---------------------------------------------------------------------------
# Interpolation fidelity: nodes are exact, interior matches the direct solver
# ---------------------------------------------------------------------------
def test_interp_reproduces_grid_nodes():
    # 300.0 K and 0.0100 kg/kg land (within float32 index rounding) on the grid;
    # interpolation must return essentially the stored (direct) value. The small
    # tolerance absorbs the residual sub-cell fraction from float32 axis spacing.
    ta, q = np.float32(300.0), np.float32(0.0100)
    assert _romps(ta, q) == pytest.approx(float(rhi.get_aorc_romps_heat_index(ta, q)), abs=1e-2)


@pytest.mark.parametrize(
    "temp_c, rh",
    [(28.0, 0.5), (31.0, 0.6), (34.0, 0.45), (37.0, 0.4), (40.0, 0.3), (30.0, 0.7)],
)
def test_romps_lut_matches_direct(temp_c, rh):
    ta = temp_c + 273.15
    q = _huss_from_rh(temp_c, rh)
    assert _romps(ta, q) == pytest.approx(float(rhi.get_aorc_romps_heat_index(np.float32(ta), np.float32(q))), abs=0.3)


@pytest.mark.parametrize(
    "temp_c, rh, p_pa",
    [(25.0, 0.5, 101_325.0), (32.0, 0.6, 101_325.0), (38.0, 0.4, 95_000.0), (30.0, 0.85, 90_000.0)],
)
def test_wbt_lut_matches_direct(temp_c, rh, p_pa):
    ta = temp_c + 273.15
    q = _huss_from_rh(temp_c, rh)
    assert _wbt(ta, q, p_pa) == pytest.approx(float(m.get_aorc_wbt(np.float32(ta), np.float32(q), np.float32(p_pa))), abs=0.5)


# ---------------------------------------------------------------------------
# Physical sanity
# ---------------------------------------------------------------------------
def test_outputs_finite_and_plausible():
    hi = _romps(305.0, 0.012)
    wb = _wbt(305.0, 0.012, 101_325.0)
    assert np.isfinite(hi) and -60.0 < hi < 200.0     # plausible heat index, degF
    assert np.isfinite(wb) and -60.0 < wb < 130.0     # plausible wet-bulb, degF


def test_romps_increases_with_humidity():
    ta = 35.0 + 273.15
    his = [_romps(ta, _huss_from_rh(35.0, rh)) for rh in (0.2, 0.4, 0.6, 0.8)]
    assert his[0] < his[1] < his[2] < his[3]


def test_romps_increases_with_temperature():
    q = _huss_from_rh(30.0, 0.5)
    his = [_romps(tc + 273.15, q) for tc in (28.0, 32.0, 36.0)]
    assert his[0] < his[1] < his[2]


@pytest.mark.parametrize("temp_c", [20.0, 30.0, 38.0])
@pytest.mark.parametrize("rh", [0.2, 0.5, 0.9])
def test_wbt_not_above_air_temperature(temp_c, rh):
    # Wet-bulb temperature cannot exceed the dry-bulb temperature.
    wb = _wbt(temp_c + 273.15, _huss_from_rh(temp_c, rh), 101_325.0)
    assert wb <= m.celsius_to_fahrenheit(temp_c) + 0.5   # small slack for interpolation


def test_wbt_approaches_air_temp_at_saturation():
    temp_c = 30.0
    wb = _wbt(temp_c + 273.15, _huss_from_rh(temp_c, 1.0), 101_325.0)
    assert wb == pytest.approx(m.celsius_to_fahrenheit(temp_c), abs=1.0)


def test_wbt_increases_with_humidity():
    ta = 32.0 + 273.15
    wbs = [_wbt(ta, _huss_from_rh(32.0, rh), 101_325.0) for rh in (0.2, 0.5, 0.8)]
    assert wbs[0] < wbs[1] < wbs[2]


# ---------------------------------------------------------------------------
# Array semantics
# ---------------------------------------------------------------------------
def test_shape_preserved():
    shape = (4, 5, 6)
    ta = np.full(shape, 305.0, np.float32)
    q = np.full(shape, 0.012, np.float32)
    p = np.full(shape, 101_325.0, np.float32)
    assert L.get_aorc_romps_heat_index(ta, q).shape == shape
    assert L.get_aorc_wbt(ta, q, p).shape == shape


def test_array_matches_scalar():
    ta = np.array([300.0, 305.0, 310.0], np.float32)
    q = np.array([0.008, 0.012, 0.016], np.float32)
    p = np.full(3, 101_325.0, np.float32)
    rhi_arr = L.get_aorc_romps_heat_index(ta, q)
    wbt_arr = L.get_aorc_wbt(ta, q, p)
    for i in range(3):
        assert rhi_arr[i] == pytest.approx(_romps(ta[i], q[i]), abs=1e-4)
        assert wbt_arr[i] == pytest.approx(_wbt(ta[i], q[i], p[i]), abs=1e-4)


def test_nan_passthrough():
    # Masked cells (NaN) must stay NaN rather than clamp to a grid edge.
    ta = np.array([300.0, np.nan, 310.0], np.float32)
    q = np.array([0.010, 0.010, np.nan], np.float32)
    p = np.full(3, 101_325.0, np.float32)
    out_hi = L.get_aorc_romps_heat_index(ta, q)
    out_wb = L.get_aorc_wbt(ta, q, p)
    assert np.isfinite(out_hi[0]) and np.isnan(out_hi[1]) and np.isnan(out_hi[2])
    assert np.isfinite(out_wb[0]) and np.isnan(out_wb[1]) and np.isnan(out_wb[2])


def test_out_of_range_inputs_clamp_to_edge():
    # Inputs beyond the grid clamp to the nearest edge and stay finite.
    edge = _wbt(TA_LO, Q_LO, 101_325.0)
    below = _wbt(TA_LO - 40.0, Q_LO - 0.01, 101_325.0)        # below the T and q edges
    assert np.isfinite(below) and below == pytest.approx(edge, abs=1e-3)

    p_edge = _wbt(300.0, 0.012, P_HI)
    p_over = _wbt(300.0, 0.012, P_HI + 20_000.0)              # above the pressure edge
    assert np.isfinite(p_over) and p_over == pytest.approx(p_edge, abs=1e-3)
