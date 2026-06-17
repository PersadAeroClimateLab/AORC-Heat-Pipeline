"""Unit tests for the heat-stress metrics in ``metrics.py``.

The expected values used here come from published references rather than from
re-running the implementation, so that a transcription error in a formula would
be caught:

* Temperature conversions: definitional fixed points (0 C = 32 F, -40 = -40).
* Heat index: the U.S. National Weather Service heat-index chart and the
  Rothfusz regression documented at
  https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml
  (stated accuracy +/- 1.3 F vs. Steadman's tables, so a 2 F tolerance is used
  against the rounded chart values).
* Saturation vapour pressure: standard psychrometric tables for the Tetens
  equation (e.g. es(20 C) ~= 23.4 hPa, es(30 C) ~= 42.4 hPa).
* Apparent temperature / humidex / simplified WBGT: definitional formulae
  (Steadman AT, Canadian humidex, ACSM simplified WBGT) evaluated by hand.
* Wet-bulb temperature: Stull (2011) reference value and the physical
  identity that at saturation Tw == T.
"""
import numpy as np
import pytest

import metrics as m


# ---------------------------------------------------------------------------
# Temperature conversions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "celsius, fahrenheit",
    [
        (0.0, 32.0),       # freezing point of water
        (100.0, 212.0),    # boiling point of water
        (-40.0, -40.0),    # the scales coincide here
        (37.0, 98.6),      # human body temperature
    ],
)
def test_celsius_to_fahrenheit(celsius, fahrenheit):
    assert m.celsius_to_fahrenheit(celsius) == pytest.approx(fahrenheit)


@pytest.mark.parametrize(
    "fahrenheit, celsius",
    [
        (32.0, 0.0),
        (212.0, 100.0),
        (-40.0, -40.0),
        (98.6, 37.0),
    ],
)
def test_fahrenheit_to_celsius(fahrenheit, celsius):
    assert m.fahrenheit_to_celsius(fahrenheit) == pytest.approx(celsius)


def test_conversions_are_inverse():
    for c in np.linspace(-50.0, 50.0, 21):
        round_trip = m.fahrenheit_to_celsius(m.celsius_to_fahrenheit(c))
        assert round_trip == pytest.approx(c, abs=1e-9)


# ---------------------------------------------------------------------------
# Heat index (NWS Rothfusz regression)
# ---------------------------------------------------------------------------
# Values from the NWS heat-index chart, 40% relative-humidity column.
# (temperature_F, relative_humidity_pct, expected_HI_F)
NWS_CHART_40PCT = [
    (80.0, 40.0, 80.0),
    (90.0, 40.0, 91.0),
    (96.0, 40.0, 101.0),
    (100.0, 40.0, 109.0),
    (104.0, 40.0, 119.0),
    (110.0, 40.0, 136.0),
]


@pytest.mark.parametrize("temp, rh, expected", NWS_CHART_40PCT)
def test_heat_index_nws_chart(temp, rh, expected):
    # +/- 2 F absorbs chart rounding plus the regression's stated +/- 1.3 F error.
    assert m.heat_index(temp, rh) == pytest.approx(expected, abs=2.0)


def test_heat_index_high_humidity_adjustment():
    # T in [80, 87] and RH > 85 triggers the high-humidity correction.
    # NWS chart gives ~105 F for 86 F / 90% RH.
    assert m.heat_index(86.0, 90.0) == pytest.approx(105.0, abs=2.0)


def test_heat_index_low_humidity_adjustment():
    # RH < 13 and 80 <= T <= 112 triggers the low-humidity correction.
    # Hand evaluation of the Rothfusz regression + adjustment for 95 F / 10% RH
    # yields ~89.4 F.
    assert m.heat_index(95.0, 10.0) == pytest.approx(89.4, abs=1.0)


def test_heat_index_uses_simple_formula_below_threshold():
    # For cool/dry conditions the simple Steadman formula is returned directly
    # (no Rothfusz regression), and the heat index stays close to the air temp.
    temp, rh = 75.0, 40.0
    expected_simple = 0.5 * (temp + 61.0 + ((temp - 68.0) * 1.2) + (rh * 0.094))
    assert m.heat_index(temp, rh) == pytest.approx(expected_simple)
    assert expected_simple < 80.0


# ---------------------------------------------------------------------------
# Saturation vapour pressure (Tetens equation)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "temp_c, expected_hpa, tol",
    [
        (0.0, 6.11, 0.01),    # triple-point reference, ~6.11 hPa
        (20.0, 23.39, 0.05),  # standard psychrometric table value
        (30.0, 42.43, 0.05),
        (-10.0, 2.60, 0.05),  # over-ice branch (T <= 0)
    ],
)
def test_saturation_vapor_pressure(temp_c, expected_hpa, tol):
    assert m.saturation_vapor_pressure(temp_c) == pytest.approx(expected_hpa, abs=tol)


def test_saturation_vapor_pressure_monotonic():
    temps = np.linspace(-30.0, 45.0, 76)
    es = np.array([m.saturation_vapor_pressure(t) for t in temps])
    assert np.all(np.diff(es) > 0)


# ---------------------------------------------------------------------------
# Vapour pressure from specific humidity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("q", [0.001, 0.005, 0.01, 0.02])
def test_vapor_pressure_roundtrip(q):
    # Inverting the standard relation q = eps*e / (p - (1-eps)*e) must recover q.
    eps = 0.622
    p = 1013.25
    e = m.vapor_pressure(q)
    q_recovered = eps * e / (p - (1.0 - eps) * e)
    assert q_recovered == pytest.approx(q, rel=1e-5)


def test_vapor_pressure_zero_humidity():
    assert m.vapor_pressure(0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Apparent temperature (Steadman), humidex, simplified WBGT
# ---------------------------------------------------------------------------
def test_apparent_temperature():
    # AT = T + 0.33*e - 0.70*ws - 4.00
    assert m.apparent_temperature(25.0, 20.0, 5.0) == pytest.approx(24.1)


def test_apparent_temperature_wind_chill_effect():
    # Higher wind speed lowers the apparent temperature.
    calm = m.apparent_temperature(30.0, 15.0, 0.0)
    windy = m.apparent_temperature(30.0, 15.0, 10.0)
    assert windy < calm


def test_humidex():
    # H = T + (5/9)*(e - 10)
    assert m.humidex(30.0, 30.0) == pytest.approx(41.1111111)


def test_humidex_no_correction_at_reference_vp():
    # At e = 10 hPa the humidex equals the air temperature.
    assert m.humidex(28.0, 10.0) == pytest.approx(28.0)


def test_swbgt():
    # sWBGT = 0.567*T + 0.393*e + 3.94
    assert m.swbgt(25.0, 20.0) == pytest.approx(25.975)


# ---------------------------------------------------------------------------
# Wet-bulb temperature (psychrometric Newton iteration)
# ---------------------------------------------------------------------------
def test_wbt_reference_value():
    # T = 25 C, RH = 50%, p = 1013.25 hPa. Stull (2011) gives Tw ~= 18.0 C.
    p = 1013.25
    es = m.saturation_vapor_pressure(25.0)
    e = 0.5 * es  # 50% RH
    eps = 0.622
    q = eps * e / (p - (1.0 - eps) * e)
    assert m.wbt(25.0, q, p) == pytest.approx(18.0, abs=0.7)


def test_wbt_equals_temp_at_saturation():
    # Saturated air: the wet-bulb temperature equals the dry-bulb temperature.
    p = 1013.25
    eps = 0.622
    for temp in (10.0, 20.0, 30.0):
        es = m.saturation_vapor_pressure(temp)
        q_sat = eps * es / (p - (1.0 - eps) * es)
        assert m.wbt(temp, q_sat, p) == pytest.approx(temp, abs=0.05)


def test_wbt_not_above_temp():
    # The wet-bulb temperature can never exceed the dry-bulb temperature.
    p = 1013.25
    for q in (0.001, 0.005, 0.01, 0.015):
        assert m.wbt(25.0, q, p) <= 25.0 + 1e-6


def test_wbt_drier_air_gives_lower_wetbulb():
    p = 1013.25
    humid = m.wbt(25.0, 0.012, p)
    dry = m.wbt(25.0, 0.004, p)
    assert dry < humid


# ---------------------------------------------------------------------------
# ddt_saturation_vapor_pressure: derivative consistency
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("temp", [-15.0, -5.0, 5.0, 25.0])
def test_ddt_saturation_vapor_pressure_matches_finite_difference(temp):
    h = 1e-3
    analytic = m.ddt_saturation_vapor_pressure(temp, m.saturation_vapor_pressure(temp))
    numeric = (
        m.saturation_vapor_pressure(temp + h) - m.saturation_vapor_pressure(temp - h)
    ) / (2.0 * h)
    assert analytic == pytest.approx(numeric, rel=1.5e-3)


# ---------------------------------------------------------------------------
# AORC wrapper functions (unit handling + composition)
# ---------------------------------------------------------------------------
def test_get_aorc_heat_index_composition():
    air_temp_k = 305.0          # ~31.85 C
    q = 0.012
    air_temp_c = air_temp_k - 273.15
    saturation_vp = m.saturation_vapor_pressure(air_temp_c)
    rel_humid = (m.vapor_pressure(q) / saturation_vp) * 100.0
    expected = m.heat_index(m.celsius_to_fahrenheit(air_temp_c), rel_humid)
    assert m.get_aorc_heat_index(air_temp_k, q) == pytest.approx(expected)


def test_get_aorc_apparent_temp_composition():
    air_temp_k = 300.0
    u, v, q = 3.0, 4.0, 0.01      # wind magnitude = 5 m/s
    air_temp_c = air_temp_k - 273.15
    sfcwind = np.sqrt(u ** 2 + v ** 2)
    expected = m.celsius_to_fahrenheit(
        m.apparent_temperature(air_temp_c, m.vapor_pressure(q), sfcwind)
    )
    assert m.get_aorc_apparent_temp(air_temp_k, u, v, q) == pytest.approx(expected)


def test_get_aorc_humidex_composition():
    air_temp_k = 303.15
    q = 0.015
    air_temp_c = air_temp_k - 273.15
    expected = m.celsius_to_fahrenheit(m.humidex(air_temp_c, m.vapor_pressure(q)))
    assert m.get_aorc_humidex(air_temp_k, q) == pytest.approx(expected)


def test_get_aorc_swbgt_composition():
    air_temp_k = 303.15
    q = 0.015
    air_temp_c = air_temp_k - 273.15
    expected = m.celsius_to_fahrenheit(m.swbgt(air_temp_c, m.vapor_pressure(q)))
    assert m.get_aorc_swbgt(air_temp_k, q) == pytest.approx(expected)


def test_get_aorc_wbt_composition_and_units():
    air_temp_k = 300.0
    q = 0.01
    surface_pressure_pa = 101325.0      # 1 atm in Pa
    expected = m.celsius_to_fahrenheit(
        m.wbt(air_temp_k - 273.15, q, surface_pressure_pa / 100.0)
    )
    assert m.get_aorc_wbt(air_temp_k, q, surface_pressure_pa) == pytest.approx(expected)
