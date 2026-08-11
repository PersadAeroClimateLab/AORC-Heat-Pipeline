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

#: Fast-math flags for kernels that branch on a value which may be NaN.
#:
#: `fastmath=NAN_SAFE_FASTMATH` includes `nnan`, which tells LLVM to assume no operand is
#: ever NaN. Under that assumption a `x != x` guard is dead code and gets
#: folded away, and an if/elif chain may be treated as exhaustive. Both
#: matter here: the pipeline masks to a region with `.where`, so most cells
#: arriving at these kernels ARE NaN. Every other fast-math relaxation is
#: kept; only `nnan` and `ninf` are given up.
NAN_SAFE_FASTMATH = {"nsz", "arcp", "contract", "afn", "reassoc"}

CELSIUS_TO_FAHRENHEIT_SCALE = np.float32(1.8)
CELSIUS_TO_FAHRENHEIT_OFFSET = np.float32(32.0)
KELVIN_TO_CELSIUS_OFFSET = np.float32(273.15)

#: Ratio of the molar masses of water vapour and dry air, dimensionless.
DRY_AIR_TO_VAPOR_MOLAR_MASS_RATIO = np.float32(0.622)

# Tetens (1930) over-water coefficients, applied at all temperatures. RH is
# conventionally reported with respect to liquid water even below 0 C (the WMO
# and station-observation convention), so the over-ice branch that used to
# apply at or below 0 C has been removed.
TETENS_REFERENCE_PRESSURE_HPA = np.float32(6.1078)
TETENS_WATER_NUMERATOR = np.float32(17.27)
TETENS_WATER_DENOMINATOR = np.float32(237.3)


@nb.vectorize(target="cpu", cache=True, fastmath=NAN_SAFE_FASTMATH)
def celsius_to_fahrenheit(air_temperature_celsius: float) -> float:
    """Convert a temperature to degrees Fahrenheit.

    :param air_temperature_celsius: Temperature in degrees Celsius
    :return: Temperature in degrees Fahrenheit
    """
    return (
        np.float32(air_temperature_celsius) * CELSIUS_TO_FAHRENHEIT_SCALE
        + CELSIUS_TO_FAHRENHEIT_OFFSET
    )


@nb.vectorize(target="cpu", cache=True, fastmath=NAN_SAFE_FASTMATH)
def fahrenheit_to_celsius(air_temperature_fahrenheit: float) -> float:
    """Convert a temperature to degrees Celsius.

    :param air_temperature_fahrenheit: Temperature in degrees Fahrenheit
    :return: Temperature in degrees Celsius
    """
    return (
        np.float32(air_temperature_fahrenheit) - CELSIUS_TO_FAHRENHEIT_OFFSET
    ) / CELSIUS_TO_FAHRENHEIT_SCALE


@nb.vectorize(target="cpu", cache=True, fastmath=NAN_SAFE_FASTMATH)
def kelvin_to_celsius(air_temperature_kelvin: float) -> float:
    """Convert a temperature to degrees Celsius.

    :param air_temperature_kelvin: Temperature in Kelvin
    :return: Temperature in degrees Celsius
    """
    return np.float32(air_temperature_kelvin) - KELVIN_TO_CELSIUS_OFFSET


@nb.vectorize(target="cpu", cache=True, fastmath=NAN_SAFE_FASTMATH)
def saturation_vapor_pressure(air_temperature_celsius: float) -> float:
    """Saturation vapour pressure from the Tetens equation.

    Uses the over-water coefficients at all temperatures, per the WMO/station
    convention of reporting relative humidity with respect to liquid water even
    below 0 C. https://en.wikipedia.org/wiki/Tetens_equation

    :param air_temperature_celsius: Temperature in degrees Celsius
    :return: Saturation vapour pressure in hPa
    """
    temperature = np.float32(air_temperature_celsius)
    # Masked cells arrive as NaN. The expression below would return NaN anyway,
    # but reaching the ordered comparison with a NaN operand raises the IEEE-754
    # invalid-operation flag, which surfaces as a RuntimeWarning per chunk.
    if temperature != temperature:
        return np.float32(np.nan)
    return TETENS_REFERENCE_PRESSURE_HPA * np.exp(
        TETENS_WATER_NUMERATOR * temperature / (temperature + TETENS_WATER_DENOMINATOR)
    )


@nb.vectorize(target="cpu", cache=True, fastmath=NAN_SAFE_FASTMATH)
def saturation_vapor_pressure_slope(air_temperature_celsius: float) -> float:
    """Temperature derivative of `saturation_vapor_pressure`.

    For es(T) = c * exp(a*T / (T + b)), d(es)/dT = es * a*b / (T + b)**2. The
    saturation vapour pressure is recomputed internally rather than accepted as
    an argument, so a caller cannot pass an inconsistent pair.

    :param air_temperature_celsius: Temperature in degrees Celsius
    :return: Rate of change of saturation vapour pressure, hPa per degree Celsius
    """
    temperature = np.float32(air_temperature_celsius)
    # See `saturation_vapor_pressure`: guards the ordered comparison below
    # against a NaN operand raising the invalid-operation flag.
    if temperature != temperature:
        return np.float32(np.nan)
    denominator = temperature + TETENS_WATER_DENOMINATOR
    saturation = TETENS_REFERENCE_PRESSURE_HPA * np.exp(
        TETENS_WATER_NUMERATOR * temperature / denominator
    )
    return (
        saturation
        * (TETENS_WATER_NUMERATOR * TETENS_WATER_DENOMINATOR)
        / (denominator * denominator)
    )


@nb.vectorize(target="cpu", cache=True, fastmath=NAN_SAFE_FASTMATH)
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


@nb.vectorize(target="cpu", cache=True, fastmath=NAN_SAFE_FASTMATH)
def relative_humidity(vapor_pressure_hpa: float, saturation_vapor_pressure_hpa: float) -> float:
    """Relative humidity as a percentage of saturation.

    :param vapor_pressure_hpa: Vapour pressure in hPa
    :param saturation_vapor_pressure_hpa: Saturation vapour pressure in hPa
    :return: Relative humidity from 0 (dry) to 100 (saturated)
    """
    return (
        np.float32(vapor_pressure_hpa) / np.float32(saturation_vapor_pressure_hpa)
    ) * np.float32(100.0)


@nb.vectorize(target="cpu", cache=True, fastmath=NAN_SAFE_FASTMATH)
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

    # Masked cells arrive as NaN. Every branch below would yield NaN anyway, but
    # the three ordered comparisons would each raise the IEEE-754
    # invalid-operation flag on the way, surfacing as a RuntimeWarning per chunk.
    if temperature != temperature or humidity != humidity:
        return np.float32(np.nan)

    index = np.float32(0.5) * (
        temperature
        + np.float32(61.0)
        + ((temperature - np.float32(68.0)) * np.float32(1.2))
        + (humidity * np.float32(0.094))
    )

    if (index + temperature) * np.float32(0.5) >= np.float32(80.0):
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


@nb.vectorize(target="cpu", cache=True, fastmath=NAN_SAFE_FASTMATH)
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


@nb.vectorize(target="cpu", cache=True, fastmath=NAN_SAFE_FASTMATH)
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


@nb.vectorize(target="cpu", cache=True, fastmath=NAN_SAFE_FASTMATH)
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


WET_BULB_MAX_ITERATIONS = 50
WET_BULB_TOLERANCE_CELSIUS = np.float32(1.0e-4)

#: Largest wet-bulb depression the solver will accept, in degrees Celsius. Acts
#: as a guard rail so a Newton step cannot run away on pathological input.
WET_BULB_MAX_DEPRESSION_CELSIUS = np.float32(40.0)

SPECIFIC_HEAT_DRY_AIR = np.float32(1005.7)    # J/kg/K
SPECIFIC_HEAT_WATER_VAPOR = np.float32(1875.0)  # J/kg/K
LATENT_HEAT_AT_ZERO_CELSIUS = np.float32(2.501e6)  # J/kg
LATENT_HEAT_SLOPE = np.float32(-2370.0)       # J/kg per degree Celsius


#: Fast-math flags for `wet_bulb_temperature`, deliberately omitting `nnan` and
#: `ninf`. With plain `fastmath=NAN_SAFE_FASTMATH` (which sets `nnan`), LLVM is permitted to
#: assume no operand is ever NaN; that license lets it collapse the branch that
#: clamps the iterate against the depression bound so that a NaN iterate takes
#: the "clamp to bound" arm unconditionally, silently replacing NaN with a
#: finite number. The pipeline relies on NaN passing through this function
#: untouched (out-of-region cells arrive as NaN via xarray's `.where` mask), so
#: `nnan` cannot be set here. `ninf` is dropped too, since the division by
#: `residual_slope` can legitimately produce infinities. The remaining flags
#: still give most of the fastmath speedup without licensing this collapse.
@nb.vectorize(target="cpu", cache=True, fastmath=NAN_SAFE_FASTMATH)
def wet_bulb_temperature(
    air_temperature_celsius: float, specific_humidity: float, air_pressure_hpa: float
) -> float:
    """Thermodynamic (isobaric) wet-bulb temperature by Newton iteration.

    Solves the psychrometric energy balance

        cp_moist * (T - Tw) = L(Tw) * (qs(Tw, p) - q)

    for Tw, i.e. the root of f(Tw) = qs(Tw) - q - (cp/L(Tw)) * (T - Tw), with
    f'(Tw) = dqs/dTw + cp/L + (T - Tw) * cp * (dL/dT) / L**2.

    Converges in roughly three to five iterations from the initial guess
    Tw = T; f is monotonic in Tw and f(T) >= 0 for sub-saturated air. For
    saturated air the solution is Tw == T.

    :param air_temperature_celsius: Air temperature in degrees Celsius
    :param specific_humidity: Specific humidity in kg/kg
    :param air_pressure_hpa: Total air pressure in hPa
    :return: Wet-bulb temperature in degrees Celsius
    """
    temperature = np.float32(air_temperature_celsius)
    humidity = np.float32(specific_humidity)
    pressure = np.float32(air_pressure_hpa)

    # A NaN input can never converge: every Newton step is NaN, so neither the
    # tolerance test nor the stall test can ever fire and the loop runs its full
    # iteration cap. Measured, that makes a NaN cell ~14x more expensive than a
    # real one -- and since the pipeline masks to a region with `.where`, most
    # cells in the bounding box are NaN. Returning NaN up front is both the
    # correct answer and far cheaper.
    #
    # This check is only reliable because NAN_SAFE_FASTMATH omits `nnan`.
    # Under `fastmath=NAN_SAFE_FASTMATH` LLVM assumes no operand is ever NaN and is free to
    # fold the comparison away.
    if (
        temperature != temperature
        or humidity != humidity
        or pressure != pressure
    ):
        return np.float32(np.nan)

    epsilon = DRY_AIR_TO_VAPOR_MOLAR_MASS_RATIO
    one_minus_epsilon = np.float32(1.0) - epsilon
    specific_heat_moist_air = SPECIFIC_HEAT_DRY_AIR + (
        SPECIFIC_HEAT_WATER_VAPOR - SPECIFIC_HEAT_DRY_AIR
    ) * humidity

    wet_bulb = temperature

    for _ in range(WET_BULB_MAX_ITERATIONS):
        previous_wet_bulb = wet_bulb

        latent_heat = LATENT_HEAT_AT_ZERO_CELSIUS + LATENT_HEAT_SLOPE * wet_bulb
        saturation = saturation_vapor_pressure(wet_bulb)
        saturation_slope = saturation_vapor_pressure_slope(wet_bulb)

        denominator = pressure - one_minus_epsilon * saturation
        saturation_humidity = epsilon * saturation / denominator
        saturation_humidity_slope = (
            epsilon * pressure * saturation_slope / (denominator * denominator)
        )

        heat_ratio = specific_heat_moist_air / latent_heat
        residual = saturation_humidity - humidity - heat_ratio * (temperature - wet_bulb)
        residual_slope = (
            saturation_humidity_slope
            + heat_ratio
            + (temperature - wet_bulb)
            * specific_heat_moist_air
            * LATENT_HEAT_SLOPE
            / (latent_heat * latent_heat)
        )

        step = residual / residual_slope
        wet_bulb -= step

        if wet_bulb > temperature:
            wet_bulb = temperature
        elif wet_bulb < temperature - WET_BULB_MAX_DEPRESSION_CELSIUS:
            wet_bulb = temperature - WET_BULB_MAX_DEPRESSION_CELSIUS

        if abs(step) < WET_BULB_TOLERANCE_CELSIUS:
            break

        # A raw Newton step can be far larger than the clamp range, so the
        # clamped iterate can sit at the same bound for many iterations while
        # `step` itself never shrinks below tolerance (the pre-clamp value is
        # what shrinks in a normal convergence, but here it can't -- the true
        # root lies outside [T - 40, T]). Detect that stall directly: if the
        # iterate is bit-identical to what it was at the top of this
        # iteration, it is provably stationary (a step that merely overshot
        # and got clamped once, then moves again next iteration as it
        # recovers, so this does not fire for that case).
        if wet_bulb == previous_wet_bulb:
            break

    return wet_bulb
