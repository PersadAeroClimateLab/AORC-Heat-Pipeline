import xarray
import numpy as np
import numba as nb


@nb.vectorize
def celsius_to_fahrenheit(temp: float) -> float:
    return (temp * 1.8) + 32


@nb.vectorize
def fahrenheit_to_celsius(temp: float) -> float:
    return (temp - 32) / 1.8


@nb.vectorize
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


@nb.vectorize
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


@nb.vectorize
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



@nb.vectorize
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


@nb.vectorize
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


@nb.vectorize
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


@nb.vectorize
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

# ---------------------------------------------------------------------------
# Thermodynamic parameters (shared across multiple Romps functions)
# ---------------------------------------------------------------------------
TRIPLE_TEMP = 273.16        # Ttrip, K
E0_VAPOR = 2.3740e6         # E0v, J/kg
RGAS_VAPOR = 461.0          # rgasv, gas constant of water vapor, J/kg/K
CV_VAPOR = 1418.0           # cvv, specific heat of water vapor, J/kg/K
CV_LIQUID = 4119.0          # cvl, specific heat of liquid water, J/kg/K


@nb.njit
def romps_pvstar(temp: float) -> float:
    """Saturation vapor pressure over liquid/ice in Pa; temp in K."""
    ptrip = 611.65          # triple-point pressure, Pa
    cvs = 1861.0            # specific heat of solid water, J/kg/K
    cpv = CV_VAPOR + RGAS_VAPOR
    if temp == 0.0:
        return 0.0
    elif temp < TRIPLE_TEMP:
        return ptrip * (temp / TRIPLE_TEMP) ** ((cpv - cvs) / RGAS_VAPOR) * np.exp(
            (E0_VAPOR + 0.3337e6 - (CV_VAPOR - cvs) * TRIPLE_TEMP) / RGAS_VAPOR
            * (1.0 / TRIPLE_TEMP - 1.0 / temp))
    else:
        return ptrip * (temp / TRIPLE_TEMP) ** ((cpv - CV_LIQUID) / RGAS_VAPOR) * np.exp(
            (E0_VAPOR - (CV_VAPOR - CV_LIQUID) * TRIPLE_TEMP) / RGAS_VAPOR
            * (1.0 / TRIPLE_TEMP - 1.0 / temp))


@nb.njit
def romps_latent_heat(temp: float) -> float:
    """Latent heat of vaporization of water in J/kg; temp in K."""
    return E0_VAPOR + (CV_VAPOR - CV_LIQUID) * (temp - TRIPLE_TEMP) + RGAS_VAPOR * temp


# ---------------------------------------------------------------------------
# Thermoregulatory parameters (shared across multiple Romps functions)
# ---------------------------------------------------------------------------
STEFAN_BOLTZMANN = 5.67e-8                          # sigma, W/m^2/K^4
EMISSIVITY = 0.97                                   # emis, surface emissivity
METABOLIC_RATE = 180.0                              # Q, W/m^2, per skin area
PHI_SALT = 0.9                                      # skin saturation factor
CORE_TEMP = 310.0                                   # Tc, K, core temperature
CORE_VAPOR_PRES = PHI_SALT * romps_pvstar(CORE_TEMP)  # Pc, Pa, core vapor pressure
ZA_EXPOSED = 60.6 / 17.4                            # Za, Pa m^2/W, exposed skin
ZA_CLOTHED = 60.6 / 11.6                            # Za_bar, Pa m^2/W, clothed skin
ZA_NAKED = 60.6 / 12.3                              # Za_un, Pa m^2/W, naked
HI_MAXITER = 100                                    # bisection iteration cap


@nb.njit
def romps_Qv(Ta: float, Pa: float) -> float:
    """Respiratory heat loss in W/m^2.

    Inlined single-use constants: eta=1.43e-6 kg/J, cpa=1006.04 J/kg/K
    (cp of dry air = cva + rgasa), rgasa=287.04 J/kg/K, p_atm=1.013e5 Pa,
    and L = romps_latent_heat(310 K).
    """
    return 1.43e-6 * METABOLIC_RATE * (
        1006.04 * (CORE_TEMP - Ta)
        + romps_latent_heat(310.0) * 287.04 / (1.013e5 * RGAS_VAPOR)
        * (CORE_VAPOR_PRES - Pa))


@nb.njit
def romps_Zs(Rs: float) -> float:
    """Skin mass-transfer resistance in Pa m^2/W."""
    return 52.1 if Rs == 0.0387 else 6.0e8 * Rs ** 5


@nb.njit
def romps_Ra(Ts: float, Ta: float) -> float:
    """Air heat-transfer resistance, exposed skin, K m^2/W."""
    hr = EMISSIVITY * 0.85 * STEFAN_BOLTZMANN * (Ts ** 2 + Ta ** 2) * (Ts + Ta)
    return 1.0 / (17.4 + hr)


@nb.njit
def romps_Ra_bar(Tf: float, Ta: float) -> float:
    """Air heat-transfer resistance, clothed skin, K m^2/W."""
    hr = EMISSIVITY * 0.79 * STEFAN_BOLTZMANN * (Tf ** 2 + Ta ** 2) * (Tf + Ta)
    return 1.0 / (11.6 + hr)


@nb.njit
def romps_Ra_un(Ts: float, Ta: float) -> float:
    """Air heat-transfer resistance, naked, K m^2/W."""
    hr = EMISSIVITY * 0.80 * STEFAN_BOLTZMANN * (Ts ** 2 + Ta ** 2) * (Ts + Ta)
    return 1.0 / (12.3 + hr)


@nb.njit
def _hi_residual_skin(kind: int, x: float, Ta: float, Pa: float, aux: float) -> float:
    """Skin/clothing energy-balance residuals (replaces the find_eqvar lambdas).

    kind: 1 = exposed Ts, 2 = clothed Tf, 3 = region II/III Tf (aux=Ts_bar),
          4 = region IV Ts (aux=QmQv), 5 = region V Ts (aux=QmQv).
    """
    Zs0 = romps_Zs(0.0387)
    if kind == 1:
        return (x - Ta) / romps_Ra(x, Ta) + (CORE_VAPOR_PRES - Pa) / (Zs0 + ZA_EXPOSED) - (CORE_TEMP - x) / 0.0387
    elif kind == 2:
        return (x - Ta) / romps_Ra_bar(x, Ta) + (CORE_VAPOR_PRES - Pa) / (Zs0 + ZA_CLOTHED) - (CORE_TEMP - x) / 0.0387
    elif kind == 3:
        r = 124.0           # Pa/K, skin vapor-pressure sensitivity
        Ts_bar = aux
        return ((x - Ta) / romps_Ra_bar(x, Ta)
                + (CORE_VAPOR_PRES - Pa) * (x - Ta) / ((Zs0 + ZA_CLOTHED) * (x - Ta) + r * romps_Ra_bar(x, Ta) * (Ts_bar - x))
                - (CORE_TEMP - Ts_bar) / 0.0387)
    elif kind == 4:
        QmQv = aux
        return (x - Ta) / romps_Ra_un(x, Ta) + (CORE_VAPOR_PRES - Pa) / (romps_Zs((CORE_TEMP - x) / QmQv) + ZA_NAKED) - QmQv
    else:  # kind == 5
        QmQv = aux
        return (x - Ta) / romps_Ra_un(x, Ta) + (PHI_SALT * romps_pvstar(x) - Pa) / ZA_NAKED - QmQv


@nb.njit
def _hi_bisect_skin(kind: int, x1: float, x2: float, Ta: float, Pa: float, aux: float) -> float:
    a, b = x1, x2
    fa = _hi_residual_skin(kind, a, Ta, Pa, aux)
    fb = _hi_residual_skin(kind, b, Ta, Pa, aux)
    if fa * fb > 0.0:
        raise ValueError("Romps heat index: skin root not bracketed.")
    c = a
    for _ in range(HI_MAXITER):
        c = 0.5 * (a + b)
        fc = _hi_residual_skin(kind, c, Ta, Pa, aux)
        if fb * fc > 0.0:
            b, fb = c, fc
        else:
            a, fa = c, fc
        if abs(a - b) < 1e-8:           # hi_tol, convergence threshold
            return c
    return c


@nb.njit
def romps_find_eqvar(Ta: float, RH: float):
    """Equivalent-variable solver. Returns (code, phi, Rf, Rs, dTcdt).

    code -> region: 1=phi (I), 2=Rf (II/III), 3=Rs (IV), 4=Rs* (V), 5=dTcdt (VI).
    """
    Pa = RH * romps_pvstar(Ta)
    phi = 0.84
    Rf = 0.0
    Rs = 0.0387
    dTcdt = 0.0
    code = 0
    Zs0 = romps_Zs(0.0387)
    m = (CORE_VAPOR_PRES - Pa) / (Zs0 + ZA_EXPOSED)
    m_bar = (CORE_VAPOR_PRES - Pa) / (Zs0 + ZA_CLOTHED)

    Ts = _hi_bisect_skin(1, max(0.0, min(CORE_TEMP, Ta) - 0.0387 * abs(m)),
                         max(CORE_TEMP, Ta) + 0.0387 * abs(m), Ta, Pa, 0.0)
    Tf = _hi_bisect_skin(2, max(0.0, min(CORE_TEMP, Ta) - 0.0387 * abs(m_bar)),
                         max(CORE_TEMP, Ta) + 0.0387 * abs(m_bar), Ta, Pa, 0.0)

    QmQv = METABOLIC_RATE - romps_Qv(Ta, Pa)
    flux1 = QmQv - (1.0 - phi) * (CORE_TEMP - Ts) / 0.0387
    flux2 = flux1 - phi * (CORE_TEMP - Tf) / 0.0387

    if flux1 <= 0.0:                                          # region I
        code = 1
        phi = 1.0 - QmQv * 0.0387 / (CORE_TEMP - Ts)
        Rf = np.inf
    elif flux2 <= 0.0:                                        # region II & III
        code = 2
        Ts_bar = CORE_TEMP - QmQv * 0.0387 / phi + (1.0 / phi - 1.0) * (CORE_TEMP - Ts)
        Tf = _hi_bisect_skin(3, Ta, Ts_bar, Ta, Pa, Ts_bar)
        Rf = romps_Ra_bar(Tf, Ta) * (Ts_bar - Tf) / (Tf - Ta)
    else:                                                     # region IV, V, VI
        Rf = 0.0
        flux3 = QmQv - (CORE_TEMP - Ta) / romps_Ra_un(CORE_TEMP, Ta) - (PHI_SALT * romps_pvstar(CORE_TEMP) - Pa) / ZA_NAKED
        if flux3 < 0.0:                                       # region IV, V
            Ts = _hi_bisect_skin(4, 0.0, CORE_TEMP, Ta, Pa, QmQv)
            Rs = (CORE_TEMP - Ts) / QmQv
            code = 3
            Ps = CORE_VAPOR_PRES - (CORE_VAPOR_PRES - Pa) * romps_Zs(Rs) / (romps_Zs(Rs) + ZA_NAKED)
            if Ps > PHI_SALT * romps_pvstar(Ts):              # region V
                Ts = _hi_bisect_skin(5, 0.0, CORE_TEMP, Ta, Pa, QmQv)
                Rs = (CORE_TEMP - Ts) / QmQv
                code = 4
        else:                                                 # region VI
            Rs = 0.0
            # 1/C with C = M*cpc/A = 150582.19 J/K/m^2 (core heat capacity per area)
            dTcdt = flux3 / 150582.19181485035
            code = 5
    return (code, phi, Rf, Rs, dTcdt)


@nb.njit
def _hi_residual_T(code: int, T: float, eqvar: float) -> float:
    """Heat-index temperature residual (replaces the find_T lambdas)."""
    Pa0 = 1.6e3             # reference air vapor pressure, Pa
    if code == 1:
        return romps_find_eqvar(T, 1.0)[1] - eqvar
    elif code == 2:
        return romps_find_eqvar(T, min(1.0, Pa0 / romps_pvstar(T)))[2] - eqvar
    elif code == 3 or code == 4:
        return romps_find_eqvar(T, Pa0 / romps_pvstar(T))[3] - eqvar
    else:  # code == 5
        return romps_find_eqvar(T, Pa0 / romps_pvstar(T))[4] - eqvar


@nb.njit
def _hi_bisect_T(code: int, x1: float, x2: float, eqvar: float) -> float:
    a, b = x1, x2
    fa = _hi_residual_T(code, a, eqvar)
    fb = _hi_residual_T(code, b, eqvar)
    if fa * fb > 0.0:
        raise ValueError("Romps heat index: temperature root not bracketed.")
    c = a
    for _ in range(HI_MAXITER):
        c = 0.5 * (a + b)
        fc = _hi_residual_T(code, c, eqvar)
        if fb * fc > 0.0:
            b, fb = c, fc
        else:
            a, fa = c, fc
        if abs(a - b) < 1e-8:           # hi_tolT, convergence threshold
            return c
    return c


@nb.njit
def romps_heatindex(Ta: float, RH: float) -> float:
    """
    Calculates the extended heat index of Lu & Romps (2022).

    A first-principles thermoregulation model that stays defined across the full
    temperature/humidity domain, unlike the NWS regression in `heat_index`.

    :param Ta: Air temperature in Kelvin
    :type Ta: float
    :param RH: Relative humidity as a fraction from 0 to 1
    :type RH: float
    :return: Heat index temperature in Kelvin
    :rtype: float
    """
    if Ta == 0.0:
        return 0.0
    e = romps_find_eqvar(Ta, RH)
    code = e[0]
    # Per region, select the equivalent variable (tuple index) and invert it over
    # the bracketing temperature range to the heat-index temperature (K).
    if code == 1:
        return _hi_bisect_T(code, 0.0, 240.0, e[1])
    elif code == 2:
        return _hi_bisect_T(code, 230.0, 300.0, e[2])
    elif code == 3 or code == 4:
        return _hi_bisect_T(code, 295.0, 350.0, e[3])
    else:
        return _hi_bisect_T(code, 340.0, 1000.0, e[4])


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


@nb.njit
def get_aorc_wbt(air_temp: float, specific_humid: float, surface_pressure: float) -> float:
    air_temp -= 273.15
    surface_pressure /= 100
    return celsius_to_fahrenheit(wbt(air_temp, specific_humid, surface_pressure))


@nb.vectorize
def get_aorc_romps_heat_index(air_temp: float, specific_humid: float) -> float:
    # air_temp is AORC air temperature in Kelvin; the Romps model expects Kelvin,
    # so it is NOT converted to Celsius here (unlike the other get_aorc_* wrappers).
    vapor_pres_pa = vapor_pressure(specific_humid) * 100.0      # hPa -> Pa
    rel_humid = vapor_pres_pa / romps_pvstar(air_temp)          # fraction 0-1
    if rel_humid < 0.0:
        rel_humid = 0.0
    elif rel_humid > 1.0:
        rel_humid = 1.0
    hi_kelvin = romps_heatindex(air_temp, rel_humid)
    return celsius_to_fahrenheit(hi_kelvin - 273.15)