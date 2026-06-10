from os import listdir
from os.path import isdir
import numpy as np
import xarray
import dask.array as da
import numba as nb
import dask
from dask.distributed import Client, LocalCluster
from time import time


@nb.vectorize#([nb.float32(nb.float32)])
def celsius_to_fahrenheit(temp: float) -> float:
    return (temp * 1.8) + 32


@nb.vectorize#([nb.float32(nb.float32)])
def fahrenheit_to_celsius(temp: float) -> float:
    return (temp - 32) / 1.8


@nb.vectorize#([nb.float32(nb.float32, nb.float32)])
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
    hi = 0.5 * (temp + 61.0 + ((temp - 68.0)*1.2) + (rel_humid*0.094))
    if hi > 80:
        hi = -42.379
        hi += 2.04901523*temp
        hi += 10.14333127*rel_humid 
        hi += -0.22475541*temp*rel_humid 
        hi += -0.00683783*(temp**2)
        hi += -0.05481717*(rel_humid**2)
        hi += 0.00122874*(temp**2)*rel_humid
        hi += 0.00085282*temp*(rel_humid**2)
        hi += -0.00000199*((rel_humid*temp)**2)
        
        if rel_humid < 13 and 80 <= temp <= 112:
            hi -= ((13 - rel_humid)/4)*np.sqrt((np.abs(17 - np.abs(temp - 95))/17))
        elif rel_humid > 85 and 80 <= temp <= 87:
            hi += ((rel_humid - 85)/10) * ((87 - temp)/5)
            
    return hi


@nb.vectorize#([nb.float32(nb.float32)])
def saturation_vapor_pressure(temp: float) -> float:
    """
    Calculates saturation vapor pressure from temperature using the Tetens equation.

    https://en.wikipedia.org/wiki/Tetens_equation

    :param temp: Temperature value in degrees Celsius
    :type temp: float
    :return: saturation vapor pressure in hPa
    :rtype: float
    """
    if temp > 0:
        return 6.1078*np.exp(17.27*temp / (temp + 237.3))
    else:
        return 6.1078*np.exp(21.875*temp / (temp + 265.5))


@nb.vectorize#([nb.float32(nb.float32)])
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
    return huss * 1013.25 / (0.622 + (1 - 0.622) * huss)



@nb.vectorize#([nb.float32(nb.float32, nb.float32, nb.float32)])
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
    return temp + 0.33*vp - 0.7*sfcwind - 4.00


@nb.vectorize#([nb.float32(nb.float32, nb.float32)])
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
    return temp + 5/9*(vp - 10)


@nb.vectorize#([nb.float32(nb.float32, nb.float32)])
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
    return (0.567*(temp) + 0.393*(vp) + 3.94)


@nb.njit(cache=True)
def ddt_saturation_vapor_pressure(temp: float, esat: float) -> float:
    """d(esat)/dT (hPa per degree C), consistent with `_esat_tetens`.
 
    d/dT [c * exp(a*T/(T+b))] = esat * a*b / (T+b)**2
    """
    if temp > 0.0:
        return esat * (17.27 * 237.3) / (temp + 237.3) ** 2
    else:
        return esat * (21.875 * 265.5) / (temp + 265.5) ** 2


@nb.vectorize#([nb.float32(nb.float32, nb.float32, nb.float32, nb.float32, nb.float32)])
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
    max_iter = 50
    tol = 1.0e-4          # degrees C; float32-safe convergence threshold
    epsilon = 0.622       # Rd/Rv
    dl_dt = -2370.0       # d(latent heat)/dT, J/kg per degree C
    # Specific heat of moist air, J/(kg K); q is constant during the iteration
    cp_moist = 1005.7 + (1875.0 - 1005.7) * specific_humid
    
    temp_wbt = temp  # initial guess
    
    for _ in range(max_iter):
        latent_heat = 2.501e6 + dl_dt * temp_wbt          # L(Tw), J/kg
        esat = saturation_vapor_pressure(temp_wbt)                      # hPa
        desat = ddt_saturation_vapor_pressure(temp_wbt, esat)           # hPa / C
        denominator = pressure - (1.0 - epsilon) * esat    # hPa
        sat_specific_humid = epsilon * esat / denominator
        deriv_sat_specific_humid = epsilon * pressure * desat / denominator ** 2
        
        ratio = cp_moist / latent_heat
        f = sat_specific_humid - specific_humid - ratio * (temp - temp_wbt)
        df = (deriv_sat_specific_humid + ratio + (temp - temp_wbt) * cp_moist * dl_dt / latent_heat ** 2)
        
        step = f / df
        temp_wbt -= step
        
        # Physical bounds: Tw cannot exceed T, and a 40 C wet-bulb depression
        # is a generous lower bound for any terrestrial conditions.
        if temp_wbt > temp:
            temp_wbt = temp
        elif temp_wbt < temp - 40.0:
            temp_wbt = temp - 40.0
    
        if abs(step) < tol:
            break
    return temp_wbt


@nb.njit
def get_aorc_heat_index(air_temp: float, specific_humid: float) -> float:
    air_temp -= 273.15
    saturation_vp = saturation_vapor_pressure(air_temp)
    rel_humid = (vapor_pressure(specific_humid) / saturation_vp)*100
    air_temp = celsius_to_fahrenheit(air_temp)
    return heat_index(air_temp, rel_humid)


@nb.njit
def get_aorc_apparent_temp(air_temp: float, u_wind: float, v_wind: float, specific_humid: float) -> float:
    sfcwind = np.sqrt((u_wind**2 + v_wind**2))
    air_temp -= 273.15
    return celsius_to_fahrenheit(apparent_temperature(air_temp, vapor_pressure(specific_humid), sfcwind))


@nb.njit
def get_aorc_humidex(air_temp: float, specific_humid: float) -> float:
    air_temp -= 273.15
    return celsius_to_fahrenheit(humidex(air_temp, vapor_pressure(specific_humid)))


@nb.njit
def get_aorc_swbgt(air_temp: float, specific_humid: float) -> float:
    air_temp -= 273.15
    return celsius_to_fahrenheit(swbgt(air_temp, vapor_pressure(specific_humid)))