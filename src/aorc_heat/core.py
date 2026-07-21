"""Scalar heat-stress formulations in literature units.

Every function takes and returns single scalar values in the units named by its
parameters, computes in float32, and is compiled by numba so the pipeline can
apply it elementwise across dask-backed arrays. Nothing here knows anything
about the AORC dataset, its variable names, or its native units -- that
translation lives in `pipeline.py`.

Each function is self-contained and is checked against published values in
`tests/test_core.py` rather than against this implementation.
"""

import numba as nb
import numpy as np

CELSIUS_TO_FAHRENHEIT_SCALE = np.float32(1.8)
CELSIUS_TO_FAHRENHEIT_OFFSET = np.float32(32.0)
KELVIN_TO_CELSIUS_OFFSET = np.float32(273.15)

#: Ratio of the molar masses of water vapour and dry air, dimensionless.
DRY_AIR_TO_VAPOR_MOLAR_MASS_RATIO = np.float32(0.622)

# Tetens (1930) coefficients: over water above 0 C, over ice at or below 0 C.
TETENS_REFERENCE_PRESSURE_HPA = np.float32(6.1078)
TETENS_WATER_NUMERATOR = np.float32(17.27)
TETENS_WATER_DENOMINATOR = np.float32(237.3)
TETENS_ICE_NUMERATOR = np.float32(21.875)
TETENS_ICE_DENOMINATOR = np.float32(265.5)


@nb.vectorize(target="cpu", cache=True, fastmath=True)
def celsius_to_fahrenheit(air_temperature_celsius: float) -> float:
    """Convert a temperature to degrees Fahrenheit.

    :param air_temperature_celsius: Temperature in degrees Celsius
    :return: Temperature in degrees Fahrenheit
    """
    return (
        np.float32(air_temperature_celsius) * CELSIUS_TO_FAHRENHEIT_SCALE
        + CELSIUS_TO_FAHRENHEIT_OFFSET
    )


@nb.vectorize(target="cpu", cache=True, fastmath=True)
def fahrenheit_to_celsius(air_temperature_fahrenheit: float) -> float:
    """Convert a temperature to degrees Celsius.

    :param air_temperature_fahrenheit: Temperature in degrees Fahrenheit
    :return: Temperature in degrees Celsius
    """
    return (
        np.float32(air_temperature_fahrenheit) - CELSIUS_TO_FAHRENHEIT_OFFSET
    ) / CELSIUS_TO_FAHRENHEIT_SCALE


@nb.vectorize(target="cpu", cache=True, fastmath=True)
def kelvin_to_celsius(air_temperature_kelvin: float) -> float:
    """Convert a temperature to degrees Celsius.

    :param air_temperature_kelvin: Temperature in Kelvin
    :return: Temperature in degrees Celsius
    """
    return np.float32(air_temperature_kelvin) - KELVIN_TO_CELSIUS_OFFSET


@nb.vectorize(target="cpu", cache=True, fastmath=True)
def saturation_vapor_pressure(air_temperature_celsius: float) -> float:
    """Saturation vapour pressure from the Tetens equation.

    Uses the over-water coefficients above 0 C and the over-ice coefficients at
    or below it. https://en.wikipedia.org/wiki/Tetens_equation

    :param air_temperature_celsius: Temperature in degrees Celsius
    :return: Saturation vapour pressure in hPa
    """
    temperature = np.float32(air_temperature_celsius)
    if temperature > np.float32(0.0):
        return TETENS_REFERENCE_PRESSURE_HPA * np.exp(
            TETENS_WATER_NUMERATOR * temperature / (temperature + TETENS_WATER_DENOMINATOR)
        )
    return TETENS_REFERENCE_PRESSURE_HPA * np.exp(
        TETENS_ICE_NUMERATOR * temperature / (temperature + TETENS_ICE_DENOMINATOR)
    )


@nb.vectorize(target="cpu", cache=True, fastmath=True)
def saturation_vapor_pressure_slope(air_temperature_celsius: float) -> float:
    """Temperature derivative of `saturation_vapor_pressure`.

    For es(T) = c * exp(a*T / (T + b)), d(es)/dT = es * a*b / (T + b)**2. The
    saturation vapour pressure is recomputed internally rather than accepted as
    an argument, so a caller cannot pass an inconsistent pair.

    :param air_temperature_celsius: Temperature in degrees Celsius
    :return: Rate of change of saturation vapour pressure, hPa per degree Celsius
    """
    temperature = np.float32(air_temperature_celsius)
    if temperature > np.float32(0.0):
        denominator = temperature + TETENS_WATER_DENOMINATOR
        saturation = TETENS_REFERENCE_PRESSURE_HPA * np.exp(
            TETENS_WATER_NUMERATOR * temperature / denominator
        )
        return (
            saturation
            * (TETENS_WATER_NUMERATOR * TETENS_WATER_DENOMINATOR)
            / (denominator * denominator)
        )
    denominator = temperature + TETENS_ICE_DENOMINATOR
    saturation = TETENS_REFERENCE_PRESSURE_HPA * np.exp(
        TETENS_ICE_NUMERATOR * temperature / denominator
    )
    return (
        saturation
        * (TETENS_ICE_NUMERATOR * TETENS_ICE_DENOMINATOR)
        / (denominator * denominator)
    )


@nb.vectorize(target="cpu", cache=True, fastmath=True)
def vapor_pressure(specific_humidity: float, air_pressure_hpa: float) -> float:
    """Vapour pressure from specific humidity and ambient pressure.

    Inverts q = eps*e / (p - (1 - eps)*e) for e.

    :param specific_humidity: Specific humidity in kg/kg
    :param air_pressure_hpa: Total air pressure in hPa
    :return: Vapour pressure in hPa
    """
    humidity = np.float32(specific_humidity)
    pressure = np.float32(air_pressure_hpa)
    epsilon = DRY_AIR_TO_VAPOR_MOLAR_MASS_RATIO
    return humidity * pressure / (epsilon + (np.float32(1.0) - epsilon) * humidity)


@nb.vectorize(target="cpu", cache=True, fastmath=True)
def relative_humidity(vapor_pressure_hpa: float, saturation_vapor_pressure_hpa: float) -> float:
    """Relative humidity as a percentage of saturation.

    :param vapor_pressure_hpa: Vapour pressure in hPa
    :param saturation_vapor_pressure_hpa: Saturation vapour pressure in hPa
    :return: Relative humidity from 0 (dry) to 100 (saturated)
    """
    return (
        np.float32(vapor_pressure_hpa) / np.float32(saturation_vapor_pressure_hpa)
    ) * np.float32(100.0)


@nb.vectorize(target="cpu", cache=True, fastmath=True)
def heat_index(air_temperature_fahrenheit: float, relative_humidity_percent: float) -> float:
    """Heat index from the National Weather Service Rothfusz regression.

    Below roughly 80 F the simple Steadman form is returned directly. Above it
    the full regression applies, with the low- and high-humidity corrections.
    Stated accuracy is +/- 1.3 F.
    https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml

    :param air_temperature_fahrenheit: Temperature in degrees Fahrenheit
    :param relative_humidity_percent: Relative humidity from 0 (dry) to 100 (saturated)
    :return: Heat index in degrees Fahrenheit
    """
    temperature = np.float32(air_temperature_fahrenheit)
    humidity = np.float32(relative_humidity_percent)

    index = np.float32(0.5) * (
        temperature
        + np.float32(61.0)
        + ((temperature - np.float32(68.0)) * np.float32(1.2))
        + (humidity * np.float32(0.094))
    )

    if index > np.float32(80.0):
        temperature_squared = temperature * temperature
        humidity_squared = humidity * humidity
        cross_term = temperature * humidity

        index = np.float32(-42.379)
        index += np.float32(2.04901523) * temperature
        index += np.float32(10.14333127) * humidity
        index += np.float32(-0.22475541) * cross_term
        index += np.float32(-0.00683783) * temperature_squared
        index += np.float32(-0.05481717) * humidity_squared
        index += np.float32(0.00122874) * temperature_squared * humidity
        index += np.float32(0.00085282) * temperature * humidity_squared
        index += np.float32(-0.00000199) * cross_term * cross_term

        if humidity < np.float32(13.0) and np.float32(80.0) <= temperature <= np.float32(112.0):
            span = np.float32(17.0) - np.abs(temperature - np.float32(95.0))
            index -= ((np.float32(13.0) - humidity) / np.float32(4.0)) * np.sqrt(
                np.abs(span / np.float32(17.0))
            )
        elif humidity > np.float32(85.0) and np.float32(80.0) <= temperature <= np.float32(87.0):
            index += ((humidity - np.float32(85.0)) / np.float32(10.0)) * (
                (np.float32(87.0) - temperature) / np.float32(5.0)
            )

    return index


@nb.vectorize(target="cpu", cache=True, fastmath=True)
def apparent_temperature(
    air_temperature_celsius: float, vapor_pressure_hpa: float, wind_speed_ms: float
) -> float:
    """Steadman apparent temperature.

    AT = T + 0.33*e - 0.70*ws - 4.00

    :param air_temperature_celsius: Temperature in degrees Celsius
    :param vapor_pressure_hpa: Vapour pressure in hPa
    :param wind_speed_ms: Wind speed in metres per second
    :return: Apparent temperature in degrees Celsius
    """
    return (
        np.float32(air_temperature_celsius)
        + np.float32(0.33) * np.float32(vapor_pressure_hpa)
        - np.float32(0.7) * np.float32(wind_speed_ms)
        - np.float32(4.0)
    )


@nb.vectorize(target="cpu", cache=True, fastmath=True)
def humidex(air_temperature_celsius: float, vapor_pressure_hpa: float) -> float:
    """Canadian humidex.

    H = T + (5/9)*(e - 10)

    :param air_temperature_celsius: Temperature in degrees Celsius
    :param vapor_pressure_hpa: Vapour pressure in hPa
    :return: Humidex in degrees Celsius
    """
    return np.float32(air_temperature_celsius) + np.float32(5.0) / np.float32(9.0) * (
        np.float32(vapor_pressure_hpa) - np.float32(10.0)
    )


@nb.vectorize(target="cpu", cache=True, fastmath=True)
def simplified_wbgt(air_temperature_celsius: float, vapor_pressure_hpa: float) -> float:
    """ACSM simplified wet-bulb globe temperature.

    sWBGT = 0.567*T + 0.393*e + 3.94

    :param air_temperature_celsius: Temperature in degrees Celsius
    :param vapor_pressure_hpa: Vapour pressure in hPa
    :return: Simplified wet-bulb globe temperature in degrees Celsius
    """
    return (
        np.float32(0.567) * np.float32(air_temperature_celsius)
        + np.float32(0.393) * np.float32(vapor_pressure_hpa)
        + np.float32(3.94)
    )
