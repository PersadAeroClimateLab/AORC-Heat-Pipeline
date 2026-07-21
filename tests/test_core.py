"""Unit tests for the scalar heat-stress formulations in ``aorc_heat.core``.

Expected values come from published references rather than from re-running the
implementation, so that a transcription error in a formula is caught rather than
reproduced:

* Temperature conversions: definitional fixed points (0 C = 32 F, -40 = -40).
* Saturation vapour pressure: standard psychrometric tables for the Tetens
  equation (es(20 C) ~= 23.4 hPa, es(30 C) ~= 42.4 hPa).
* Heat index: the U.S. National Weather Service heat-index chart and the
  Rothfusz regression at
  https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml
* Apparent temperature / humidex / simplified WBGT: the Steadman, Canadian
  humidex, and ACSM definitional formulae evaluated by hand.
* Wet-bulb temperature: Stull (2011) and the identity Tw == T at saturation.

Tolerances are loose enough to absorb float32 rounding; the kernels compute in
single precision by design.
"""
import numpy as np
import pytest

from aorc_heat import core


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
    assert core.celsius_to_fahrenheit(celsius) == pytest.approx(fahrenheit, abs=1e-4)


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
    assert core.fahrenheit_to_celsius(fahrenheit) == pytest.approx(celsius, abs=1e-4)


def test_conversions_are_inverse():
    for celsius in np.linspace(-50.0, 50.0, 21):
        round_trip = core.fahrenheit_to_celsius(core.celsius_to_fahrenheit(celsius))
        assert round_trip == pytest.approx(celsius, abs=1e-4)


@pytest.mark.parametrize(
    "kelvin, celsius",
    [
        (273.15, 0.0),
        (373.15, 100.0),
        (300.0, 26.85),
    ],
)
def test_kelvin_to_celsius(kelvin, celsius):
    assert core.kelvin_to_celsius(kelvin) == pytest.approx(celsius, abs=1e-3)


# ---------------------------------------------------------------------------
# Saturation vapour pressure (Tetens equation)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "air_temperature_celsius, expected_hpa, tolerance",
    [
        (0.0, 6.11, 0.01),    # triple-point reference
        (20.0, 23.39, 0.05),  # standard psychrometric table value
        (30.0, 42.43, 0.05),
        (-10.0, 2.60, 0.05),  # over-ice branch (T <= 0)
    ],
)
def test_saturation_vapor_pressure(air_temperature_celsius, expected_hpa, tolerance):
    result = core.saturation_vapor_pressure(air_temperature_celsius)
    assert result == pytest.approx(expected_hpa, abs=tolerance)


def test_saturation_vapor_pressure_increases_with_temperature():
    temperatures = np.linspace(-30.0, 45.0, 76)
    pressures = np.array([core.saturation_vapor_pressure(t) for t in temperatures])
    assert np.all(np.diff(pressures) > 0)


@pytest.mark.parametrize("air_temperature_celsius", [-15.0, -5.0, 5.0, 25.0])
def test_saturation_vapor_pressure_slope_matches_finite_difference(air_temperature_celsius):
    delta = 1e-3
    analytic = core.saturation_vapor_pressure_slope(air_temperature_celsius)
    numeric = (
        core.saturation_vapor_pressure(air_temperature_celsius + delta)
        - core.saturation_vapor_pressure(air_temperature_celsius - delta)
    ) / (2.0 * delta)
    assert analytic == pytest.approx(numeric, rel=1.5e-3)


# ---------------------------------------------------------------------------
# Vapour pressure from specific humidity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("specific_humidity", [0.001, 0.005, 0.01, 0.02])
@pytest.mark.parametrize("air_pressure_hpa", [1013.25, 900.0, 850.0])
def test_vapor_pressure_roundtrip(specific_humidity, air_pressure_hpa):
    # Inverting q = eps*e / (p - (1-eps)*e) must recover the input humidity.
    epsilon = 0.622
    vapor = core.vapor_pressure(specific_humidity, air_pressure_hpa)
    recovered = epsilon * vapor / (air_pressure_hpa - (1.0 - epsilon) * vapor)
    assert recovered == pytest.approx(specific_humidity, rel=1e-5)


def test_vapor_pressure_zero_humidity():
    assert core.vapor_pressure(0.0, 1013.25) == pytest.approx(0.0, abs=1e-6)


def test_vapor_pressure_scales_with_ambient_pressure():
    # At fixed specific humidity, lower ambient pressure means lower vapour
    # pressure. This is the behaviour the hardcoded 1013.25 hPa used to hide.
    sea_level = core.vapor_pressure(0.01, 1013.25)
    high_elevation = core.vapor_pressure(0.01, 850.0)
    assert high_elevation < sea_level


# ---------------------------------------------------------------------------
# Relative humidity
# ---------------------------------------------------------------------------
def test_relative_humidity_saturated_air_is_100_percent():
    saturation = core.saturation_vapor_pressure(25.0)
    assert core.relative_humidity(saturation, saturation) == pytest.approx(100.0, rel=1e-5)


def test_relative_humidity_half_saturated():
    saturation = core.saturation_vapor_pressure(25.0)
    assert core.relative_humidity(0.5 * saturation, saturation) == pytest.approx(50.0, rel=1e-5)


# ---------------------------------------------------------------------------
# Heat index (NWS Rothfusz regression)
# ---------------------------------------------------------------------------
# Values read from the NWS heat-index chart, 40% relative-humidity column.
NWS_CHART_40_PERCENT = [
    (80.0, 40.0, 80.0),
    (90.0, 40.0, 91.0),
    (96.0, 40.0, 101.0),
    (100.0, 40.0, 109.0),
    (104.0, 40.0, 119.0),
    (110.0, 40.0, 136.0),
]


@pytest.mark.parametrize(
    "air_temperature_fahrenheit, relative_humidity_percent, expected",
    NWS_CHART_40_PERCENT,
)
def test_heat_index_matches_nws_chart(
    air_temperature_fahrenheit, relative_humidity_percent, expected
):
    # +/- 2 F absorbs chart rounding plus the regression's stated +/- 1.3 F error.
    result = core.heat_index(air_temperature_fahrenheit, relative_humidity_percent)
    assert result == pytest.approx(expected, abs=2.0)


def test_heat_index_high_humidity_adjustment():
    # T in [80, 87] with RH > 85 triggers the high-humidity correction.
    # The NWS chart gives ~105 F at 86 F / 90% RH.
    assert core.heat_index(86.0, 90.0) == pytest.approx(105.0, abs=2.0)


def test_heat_index_low_humidity_adjustment():
    # RH < 13 with 80 <= T <= 112 triggers the low-humidity correction. Hand
    # evaluation of the Rothfusz regression plus adjustment gives ~89.4 F.
    assert core.heat_index(95.0, 10.0) == pytest.approx(89.4, abs=1.0)


def test_heat_index_uses_simple_formula_below_threshold():
    # Below 80 F the simple Steadman form is returned without the regression.
    air_temperature_fahrenheit, relative_humidity_percent = 75.0, 40.0
    expected = 0.5 * (
        air_temperature_fahrenheit
        + 61.0
        + ((air_temperature_fahrenheit - 68.0) * 1.2)
        + (relative_humidity_percent * 0.094)
    )
    assert expected < 80.0
    result = core.heat_index(air_temperature_fahrenheit, relative_humidity_percent)
    assert result == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# Apparent temperature (Steadman), humidex, simplified WBGT
# ---------------------------------------------------------------------------
def test_apparent_temperature():
    # AT = T + 0.33*e - 0.70*ws - 4.00  ->  25 + 6.6 - 3.5 - 4 = 24.1
    assert core.apparent_temperature(25.0, 20.0, 5.0) == pytest.approx(24.1, rel=1e-5)


def test_apparent_temperature_wind_lowers_result():
    calm = core.apparent_temperature(30.0, 15.0, 0.0)
    windy = core.apparent_temperature(30.0, 15.0, 10.0)
    assert windy < calm


def test_humidex():
    # H = T + (5/9)*(e - 10)  ->  30 + (5/9)*20 = 41.1111
    assert core.humidex(30.0, 30.0) == pytest.approx(41.1111111, rel=1e-5)


def test_humidex_equals_air_temperature_at_reference_vapor_pressure():
    assert core.humidex(28.0, 10.0) == pytest.approx(28.0, rel=1e-5)


def test_simplified_wbgt():
    # sWBGT = 0.567*T + 0.393*e + 3.94  ->  14.175 + 7.86 + 3.94 = 25.975
    assert core.simplified_wbgt(25.0, 20.0) == pytest.approx(25.975, rel=1e-5)


# ---------------------------------------------------------------------------
# Wet-bulb temperature (psychrometric Newton iteration)
# ---------------------------------------------------------------------------
def _specific_humidity_at(relative_humidity_fraction, air_temperature_celsius, air_pressure_hpa):
    """Specific humidity for a given RH, using the same relation as core."""
    epsilon = 0.622
    vapor = relative_humidity_fraction * core.saturation_vapor_pressure(air_temperature_celsius)
    return epsilon * vapor / (air_pressure_hpa - (1.0 - epsilon) * vapor)


def test_wet_bulb_temperature_stull_reference():
    # T = 25 C, RH = 50%, p = 1013.25 hPa. Stull (2011) gives Tw ~= 18.0 C.
    pressure = 1013.25
    humidity = _specific_humidity_at(0.5, 25.0, pressure)
    assert core.wet_bulb_temperature(25.0, humidity, pressure) == pytest.approx(18.0, abs=0.7)


@pytest.mark.parametrize("air_temperature_celsius", [10.0, 20.0, 30.0])
def test_wet_bulb_temperature_equals_air_temperature_at_saturation(air_temperature_celsius):
    pressure = 1013.25
    humidity = _specific_humidity_at(1.0, air_temperature_celsius, pressure)
    result = core.wet_bulb_temperature(air_temperature_celsius, humidity, pressure)
    assert result == pytest.approx(air_temperature_celsius, abs=0.05)


@pytest.mark.parametrize("specific_humidity", [0.001, 0.005, 0.01, 0.015])
def test_wet_bulb_temperature_never_exceeds_air_temperature(specific_humidity):
    assert core.wet_bulb_temperature(25.0, specific_humidity, 1013.25) <= 25.0 + 1e-4


def test_wet_bulb_temperature_lower_in_drier_air():
    humid = core.wet_bulb_temperature(25.0, 0.012, 1013.25)
    dry = core.wet_bulb_temperature(25.0, 0.004, 1013.25)
    assert dry < humid
