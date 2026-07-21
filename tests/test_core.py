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
