from os import listdir
from os.path import isdir
import numpy as np
import xarray
import dask.array as da
import numba as nb
import dask
from dask.distributed import Client, LocalCluster
from time import time


@nb.vectorize([nb.float32(nb.float32)])
def celsius_to_fahrenheit(temp: float) -> float:
    return (temp * 1.8) + 32


@nb.vectorize([nb.float32(nb.float32)])
def fahrenheit_to_celsius(temp: float) -> float:
    return (temp - 32) / 1.8


@nb.vectorize([nb.float32(nb.float32, nb.float32)])
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



@nb.vectorize([nb.float32(nb.float32)])
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

@nb.vectorize([nb.float32(nb.float32)])
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



@nb.vectorize([nb.float32(nb.float32, nb.float32, nb.float32)])
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


@nb.vectorize([nb.float32(nb.float32, nb.float32)])
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
    e = 6.112 * 10**(7.5*temp/(237.7 + temp)) * vp / 100
    return temp + 5/9*(e - 10)


@nb.vectorize([nb.float32(nb.float32, nb.float32)])
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