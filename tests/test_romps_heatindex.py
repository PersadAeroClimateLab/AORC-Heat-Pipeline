"""Unit tests for the Lu & Romps (2022) extended heat index in ``metrics.py``.

Reference: Lu, Y.-C. and Romps, D. M. (2022), "Extending the Heat Index",
J. Appl. Meteor. Climatol., 61, 1367-1383. The model is a first-principles
thermoregulation model whose defining property is that the heat index equals the
air temperature at the reference vapour pressure ``Pa0`` (1600 Pa); these tests
exercise that invariant, the saturation-vapour-pressure thermodynamics,
monotonicity, and agreement with the independent NWS regression in the range
where both models are valid.
"""
import numpy as np
import pytest
import metrics as m


def _huss_from_vp_hpa(vp_hpa):
    """Inverse of ``vapor_pressure()``: specific humidity for a target vp (hPa)."""
    return 0.622 * vp_hpa / (1013.25 - 0.378 * vp_hpa)


# Triple-point pressure and the reference vapour pressure are function-local
# constants in metrics.py, so they are restated here as literals.
PTRIP = 611.65   # Pa
PA0 = 1.6e3      # Pa, reference air vapour pressure


# ---------------------------------------------------------------------------
# Romps saturation vapour pressure (over liquid/ice, in Pa, temp in K)
# ---------------------------------------------------------------------------
def test_pvstar_equals_ptrip_at_triple_point():
    assert m.romps_pvstar(m.TRIPLE_TEMP) == pytest.approx(PTRIP, abs=1e-6)


def test_pvstar_zero_guard():
    assert m.romps_pvstar(0.0) == 0.0


def test_pvstar_monotonic():
    assert m.romps_pvstar(310.0) > m.romps_pvstar(300.0) > 0.0


def test_pvstar_branch_continuity_at_triple_point():
    # liquid and ice branches must agree across the triple-point temperature.
    assert m.romps_pvstar(m.TRIPLE_TEMP + 1e-3) == pytest.approx(
        m.romps_pvstar(m.TRIPLE_TEMP - 1e-3), abs=1.0
    )


# ---------------------------------------------------------------------------
# Defining invariant: HI == Ta at the reference vapour pressure Pa0
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("Ta", [300.0, 305.0, 312.0])
def test_heatindex_equals_temp_at_reference_vp(Ta):
    rh = PA0 / m.romps_pvstar(Ta)
    assert m.romps_heatindex(Ta, rh) == pytest.approx(Ta, abs=1e-3)   # Kelvin


def test_wrapper_equals_air_temp_at_reference_vp():
    # End-to-end: when Pa == Pa0 the heat index (deg F) equals the air temp (deg F).
    Ta = 305.0
    huss_ref = _huss_from_vp_hpa(PA0 / 100.0)   # Pa0 expressed in hPa
    expected_f = m.celsius_to_fahrenheit(Ta - 273.15)
    assert m.get_aorc_romps_heat_index(Ta, huss_ref) == pytest.approx(expected_f, abs=1e-2)


# ---------------------------------------------------------------------------
# Direction and monotonicity
# ---------------------------------------------------------------------------
def test_heatindex_direction_relative_to_reference_humidity():
    Ta = 305.0
    air_temp_f = m.celsius_to_fahrenheit(Ta - 273.15)
    assert m.get_aorc_romps_heat_index(Ta, _huss_from_vp_hpa(8.0)) < air_temp_f    # Pa < Pa0
    assert m.get_aorc_romps_heat_index(Ta, _huss_from_vp_hpa(25.0)) > air_temp_f   # Pa > Pa0


def test_heatindex_increases_with_humidity():
    Ta = 305.0
    hi = [m.get_aorc_romps_heat_index(Ta, _huss_from_vp_hpa(vp)) for vp in (10.0, 20.0, 30.0)]
    assert hi[0] < hi[1] < hi[2]


def test_heatindex_increases_with_temperature():
    huss = _huss_from_vp_hpa(15.0)
    hi = [m.get_aorc_romps_heat_index(Ta, huss) for Ta in (300.0, 305.0, 310.0)]
    assert hi[0] < hi[1] < hi[2]


# ---------------------------------------------------------------------------
# Output sanity and masked-value guard
# ---------------------------------------------------------------------------
def test_heatindex_output_is_finite_and_plausible():
    hi = m.get_aorc_romps_heat_index(305.0, 0.012)
    assert np.isfinite(hi)
    assert -50.0 < hi < 200.0    # plausible degrees-F bound


def test_heatindex_zero_temperature_guard():
    # Fill/masked values (Ta == 0) propagate as 0 rather than raising.
    assert m.romps_heatindex(0.0, 0.5) == 0.0


# ---------------------------------------------------------------------------
# Cross-validation against the independent NWS regression (heat_index)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "temp_c, rh_pct",
    [(28.0, 50.0), (31.85, 50.0), (35.0, 40.0), (30.0, 70.0)],
)
def test_romps_agrees_with_nws_in_overlap_region(temp_c, rh_pct):
    # Two independent heat-index models should agree to a few deg F in the
    # moderate range where the NWS regression is valid.
    es_hpa = m.saturation_vapor_pressure(temp_c)
    huss = _huss_from_vp_hpa(rh_pct / 100.0 * es_hpa)
    temp_k = temp_c + 273.15
    nws = m.get_aorc_heat_index(temp_k, huss)
    romps = m.get_aorc_romps_heat_index(temp_k, huss)
    assert romps == pytest.approx(nws, abs=3.0)
