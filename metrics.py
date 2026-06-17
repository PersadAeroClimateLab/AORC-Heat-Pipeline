import xarray
import numpy as np
import numba as nb

# Global constants as float32 for consistency and memory efficiency
CELSIUS_TO_F_SCALE = np.float32(1.8)
CELSIUS_TO_F_OFFSET = np.float32(32.0)


@nb.vectorize(target="cpu", fastmath=True, cache=True)
def celsius_to_fahrenheit(temp: float) -> float:
    return (temp * CELSIUS_TO_F_SCALE) + CELSIUS_TO_F_OFFSET


@nb.vectorize(target="cpu", fastmath=True, cache=True)
def fahrenheit_to_celsius(temp: float) -> float:
    return (temp - CELSIUS_TO_F_OFFSET) / CELSIUS_TO_F_SCALE


@nb.vectorize(target="cpu", fastmath=True, cache=True)
def heat_index(temp: float, rel_humid: float) -> float:
    """
    Calculates heat index from temperature and relative humidity.

    This function relies on a regression of the National Weather Service heat index.
    It has an error of +/- 1.3 degrees Fahrenheit.
    https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml

    :param temp: Temperature value in degrees Fahrenheit
    :type temp: float
    :param rel_humid: Relative humidity on scale from 0 (no moisture) to 100 (fully saturated).
    :type rel_humid: float
    :return: Heat index value in degrees Fahrenheit
    :rtype: float
    """
    temp = np.float32(temp)
    rel_humid = np.float32(rel_humid)

    hi = np.float32(0.5) * (temp + np.float32(61.0) + ((temp - np.float32(68.0)) * np.float32(1.2)) + (rel_humid * np.float32(0.094)))
    if hi > np.float32(80.0):
        temp2 = temp * temp
        humid2 = rel_humid * rel_humid
        temp_humid = temp * rel_humid

        hi = np.float32(-42.379)
        hi += np.float32(2.04901523) * temp
        hi += np.float32(10.14333127) * rel_humid
        hi += np.float32(-0.22475541) * temp_humid
        hi += np.float32(-0.00683783) * temp2
        hi += np.float32(-0.05481717) * humid2
        hi += np.float32(0.00122874) * temp2 * rel_humid
        hi += np.float32(0.00085282) * temp * humid2
        hi += np.float32(-0.00000199) * temp_humid * temp_humid

        if rel_humid < np.float32(13.0) and np.float32(80.0) <= temp <= np.float32(112.0):
            delta = np.float32(17.0) - np.abs(temp - np.float32(95.0))
            hi -= ((np.float32(13.0) - rel_humid) / np.float32(4.0)) * np.sqrt(np.abs(delta / np.float32(17.0)))
        elif rel_humid > np.float32(85.0) and np.float32(80.0) <= temp <= np.float32(87.0):
            hi += ((rel_humid - np.float32(85.0)) / np.float32(10.0)) * ((np.float32(87.0) - temp) / np.float32(5.0))

    return hi


@nb.vectorize(target="cpu", fastmath=True, cache=True)
def saturation_vapor_pressure(temp: float) -> float:
    """
    Calculates saturation vapor pressure from temperature using the Tetens equation.

    https://en.wikipedia.org/wiki/Tetens_equation

    :param temp: Temperature value in degrees Celsius
    :type temp: float
    :return: saturation vapor pressure in hPa
    :rtype: float
    """
    temp = np.float32(temp)
    if temp > np.float32(0.0):
        return np.float32(6.1078) * np.exp(np.float32(17.27) * temp / (temp + np.float32(237.3)))
    else:
        return np.float32(6.1078) * np.exp(np.float32(21.875) * temp / (temp + np.float32(265.5)))


@nb.vectorize(target="cpu", fastmath=True, cache=True)
def vapor_pressure(huss: float) -> float:
    """
    Calculates vapor pressure from specific humidity.

    Assumes total pressure is 1013.25 hPa and the ratio of gas constants is 0.622

    :param huss: Specific humidity (kg/kg)
    :type huss: float
    :param svp: saturation vapor pressure in hPa
    :type svp: float
    :return: vapor pressure in hPa
    :rtype: float
    """
    huss = np.float32(huss)
    epsilon = np.float32(0.622)
    p_total = np.float32(1013.25)
    return huss * p_total / (epsilon + (np.float32(1.0) - epsilon) * huss)



@nb.vectorize(target="cpu", fastmath=True, cache=True)
def apparent_temperature(temp: float, vp: float, sfcwind: float) -> float:
    """
    Calculates apparent temperature.

    :param temp: Temperature value in degrees Celsius
    :type temp: float
    :param vp: vapor pressure in hPa
    :type vp: float
    :param sfcwind: Surface winds in meters per second
    :type sfcwind: float
    :return: apparent temperature in degrees Celsius
    :rtype: float
    """
    temp = np.float32(temp)
    vp = np.float32(vp)
    sfcwind = np.float32(sfcwind)
    return temp + np.float32(0.33) * vp - np.float32(0.7) * sfcwind - np.float32(4.0)


@nb.vectorize(target="cpu", fastmath=True, cache=True)
def humidex(temp: float, vp: float) -> float:
    """
    Calculates apparent temperature.

    :param temp: Temperature value in degrees Celsius
    :type temp: float
    :param vp: vapor pressure in hPa
    :type vp: float
    :return: humidex temperature in degrees Celsius
    :rtype: float
    """
    temp = np.float32(temp)
    vp = np.float32(vp)
    return temp + np.float32(5.0) / np.float32(9.0) * (vp - np.float32(10.0))


@nb.vectorize(target="cpu", fastmath=True, cache=True)
def swbgt(temp: float, vp: float) -> float:
    """
    Calculates simple wet-bulb globe temperature.

    :param temp: Temperature value in degrees Celsius
    :type temp: float
    :param vp: vapor pressure in hPa
    :type vp: float
    :return: simple wet-bulb globe temperature in degrees Celsius
    :rtype: float
    """
    temp = np.float32(temp)
    vp = np.float32(vp)
    return np.float32(0.567) * temp + np.float32(0.393) * vp + np.float32(3.94)


@nb.njit(cache=True)
def ddt_saturation_vapor_pressure(temp: float, esat: float) -> float:
    """d(esat)/dT (hPa per degree C), consistent with `_esat_tetens`.

    d/dT [c * exp(a*T/(T+b))] = esat * a*b / (T+b)**2
    """
    temp = np.float32(temp)
    esat = np.float32(esat)
    if temp > np.float32(0.0):
        denom = temp + np.float32(237.3)
        return esat * (np.float32(17.27) * np.float32(237.3)) / (denom * denom)
    else:
        denom = temp + np.float32(265.5)
        return esat * (np.float32(21.875) * np.float32(265.5)) / (denom * denom)


@nb.vectorize(target="cpu", fastmath=True, cache=True)
def wbt(temp: float, specific_humid: float, pressure: float) -> float:
    """
    Calculates the thermodynamic (isobaric/psychrometric) wet-bulb temperature
    by Newton iteration on the psychrometric energy balance:

        cp_moist * (T - Tw) = L(Tw) * (qs(Tw, p) - q)

    i.e. the root of  f(Tw) = qs(Tw) - q - (cp/L(Tw)) * (T - Tw),
    with  f'(Tw) = dqs/dTw + cp/L + (T - Tw) * cp * (dL/dT) / L**2.

    Converges in ~3-5 iterations from the initial guess Tw = T (f is monotonic
    in Tw and f(T) >= 0 for sub-saturated air). For saturated air, Tw == T.

    :param temp: Air temperature in degrees Celsius
    :type temp: float
    :param specific_humid: Specific humidity in kg/kg
    :type specific_humid: float
    :param pressure: Total air pressure in hPa (AORC PRES is in Pa: divide by 100)
    :type pressure: float
    :return: Wet-bulb temperature in degrees Celsius
    :rtype: float
    """
    temp = np.float32(temp)
    specific_humid = np.float32(specific_humid)
    pressure = np.float32(pressure)

    max_iter = 50
    tol = np.float32(1.0e-4)
    epsilon = np.float32(0.622)
    dl_dt = np.float32(-2370.0)
    cp_moist = np.float32(1005.7) + (np.float32(1875.0) - np.float32(1005.7)) * specific_humid
    l_ref = np.float32(2.501e6)
    one_minus_eps = np.float32(1.0) - epsilon
    max_depression = np.float32(40.0)

    temp_wbt = temp

    for _ in range(max_iter):
        latent_heat = l_ref + dl_dt * temp_wbt
        esat = saturation_vapor_pressure(temp_wbt)
        desat = ddt_saturation_vapor_pressure(temp_wbt, esat)
        denominator = pressure - one_minus_eps * esat
        sat_specific_humid = epsilon * esat / denominator
        denom2 = denominator * denominator
        deriv_sat_specific_humid = epsilon * pressure * desat / denom2

        ratio = cp_moist / latent_heat
        f = sat_specific_humid - specific_humid - ratio * (temp - temp_wbt)
        lh2 = latent_heat * latent_heat
        df = deriv_sat_specific_humid + ratio + (temp - temp_wbt) * cp_moist * dl_dt / lh2

        step = f / df
        temp_wbt -= step

        if temp_wbt > temp:
            temp_wbt = temp
        elif temp_wbt < temp - max_depression:
            temp_wbt = temp - max_depression

        if abs(step) < tol:
            break

    return temp_wbt


@nb.njit(cache=True)
def get_aorc_heat_index(air_temp: float, specific_humid: float) -> float:
    air_temp_c = air_temp - np.float32(273.15)
    saturation_vp = saturation_vapor_pressure(air_temp_c)
    rel_humid = (vapor_pressure(specific_humid) / saturation_vp) * np.float32(100.0)
    return heat_index(celsius_to_fahrenheit(air_temp_c), rel_humid)


@nb.njit(cache=True)
def get_aorc_apparent_temp(air_temp: float, u_wind: float, v_wind: float, specific_humid: float) -> float:
    u2 = u_wind * u_wind
    v2 = v_wind * v_wind
    sfcwind = np.sqrt(u2 + v2)
    air_temp_c = air_temp - np.float32(273.15)
    return celsius_to_fahrenheit(apparent_temperature(air_temp_c, vapor_pressure(specific_humid), sfcwind))


@nb.njit(cache=True)
def get_aorc_humidex(air_temp: float, specific_humid: float) -> float:
    air_temp_c = air_temp - np.float32(273.15)
    return celsius_to_fahrenheit(humidex(air_temp_c, vapor_pressure(specific_humid)))


@nb.njit(cache=True)
def get_aorc_swbgt(air_temp: float, specific_humid: float) -> float:
    air_temp_c = air_temp - np.float32(273.15)
    return celsius_to_fahrenheit(swbgt(air_temp_c, vapor_pressure(specific_humid)))


@nb.njit(cache=True)
def get_aorc_wbt(air_temp: float, specific_humid: float, surface_pressure: float) -> float:
    air_temp_c = air_temp - np.float32(273.15)
    pressure_hpa = surface_pressure / np.float32(100.0)
    return celsius_to_fahrenheit(wbt(air_temp_c, specific_humid, pressure_hpa))


@nb.njit(cache=True)#@nb.vectorize(target="cpu", fastmath=True, cache=True)
def get_aorc_romps_heat_index(air_temp: float, specific_humid: float) -> float:    
    if air_temp < np.float32(270.0) or specific_humid < np.float32(0.0):
        return air_temp  # Return input temp for extreme conditions

    pvstar = romps_pvstar(air_temp)
    if pvstar <= np.float32(0.0):
        return air_temp  # Fallback if saturation vapor pressure invalid

    vapor_pres_pa = vapor_pressure(specific_humid) * np.float32(100.0)
    rel_humid = vapor_pres_pa / pvstar
    if rel_humid < 0.0:
        rel_humid = 0.0
    elif rel_humid > 1.0:
        rel_humid = 1.0
    hi_kelvin = romps_heatindex(air_temp, rel_humid)
    return celsius_to_fahrenheit(hi_kelvin - TRIPLE_TEMP)