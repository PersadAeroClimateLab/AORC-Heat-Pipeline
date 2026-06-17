import numpy as np
import numba as nb
# ---------------------------------------------------------------------------
# Thermodynamic parameters (shared across multiple Romps functions)
# ---------------------------------------------------------------------------
TRIPLE_TEMP = np.float32(273.16)        # Ttrip, K
E0_VAPOR = np.float32(2.3740e6)         # E0v, J/kg
RGAS_VAPOR = np.float32(461.0)          # rgasv, gas constant of water vapor, J/kg/K
CV_VAPOR = np.float32(1418.0)           # cvv, specific heat of water vapor, J/kg/K
CV_LIQUID = np.float32(4119.0)          # cvl, specific heat of liquid water, J/kg/K


@nb.njit(cache=True, fastmath=True)
def romps_pvstar(temp: float) -> float:
    """Saturation vapor pressure over liquid/ice in Pa; temp in K."""
    ptrip = np.float32(611.65)
    cvs = np.float32(1861.0)

    if temp == np.float32(0.0):
        return np.float32(0.0)*temp

    ratio = temp / TRIPLE_TEMP
    inv_ratio = np.float32(1.0) / TRIPLE_TEMP - np.float32(1.0) / temp

    if temp < TRIPLE_TEMP:
        exp_arg = (E0_VAPOR + np.float32(0.3337e6) - (CV_VAPOR - cvs) * TRIPLE_TEMP) / RGAS_VAPOR * inv_ratio
        power_arg = (CV_VAPOR - cvs) / RGAS_VAPOR
    else:
        exp_arg = (E0_VAPOR - (CV_VAPOR - CV_LIQUID) * TRIPLE_TEMP) / RGAS_VAPOR * inv_ratio
        power_arg = (CV_VAPOR - CV_LIQUID) / RGAS_VAPOR

    return ptrip * np.power(ratio, power_arg) * np.exp(exp_arg)


@nb.njit(cache=True, fastmath=True)
def romps_latent_heat(temp: float) -> float:
    """Latent heat of vaporization of water in J/kg; temp in K."""
    return E0_VAPOR + (CV_VAPOR - CV_LIQUID) * (temp - TRIPLE_TEMP) + RGAS_VAPOR * temp


# ---------------------------------------------------------------------------
# Thermoregulatory parameters (shared across multiple Romps functions)
# ---------------------------------------------------------------------------
STEFAN_BOLTZMANN = np.float32(5.67e-8)              # sigma, W/m^2/K^4
EMISSIVITY = np.float32(0.97)                       # emis, surface emissivity
METABOLIC_RATE = np.float32(180.0)                  # Q, W/m^2, per skin area
PHI_SALT = np.float32(0.9)                          # skin saturation factor
CORE_TEMP = np.float32(310.0)                       # Tc, K, core temperature
CORE_VAPOR_PRES = PHI_SALT * romps_pvstar(CORE_TEMP)  # Pc, Pa, core vapor pressure
ZA_EXPOSED = np.float32(60.6 / 17.4)                # Za, Pa m^2/W, exposed skin
ZA_CLOTHED = np.float32(60.6 / 11.6)                # Za_bar, Pa m^2/W, clothed skin
ZA_NAKED = np.float32(60.6 / 12.3)                  # Za_un, Pa m^2/W, naked
HI_MAXITER = 100                                    # bisection iteration cap


@nb.njit(cache=True, fastmath=True)
def romps_Qv(Ta: float, Pa: float) -> float:
    """Respiratory heat loss in W/m^2.

    Inlined single-use constants: eta=1.43e-6 kg/J, cpa=1006.04 J/kg/K
    (cp of dry air = cva + rgasa), rgasa=287.04 J/kg/K, p_atm=1.013e5 Pa,
    and L = romps_latent_heat(310 K).
    """
    eta = np.float32(1.43e-6)
    cpa = np.float32(1006.04)
    lh = romps_latent_heat(np.float32(310.0))
    coef = np.float32(287.04) / (np.float32(1.013e5) * RGAS_VAPOR)

    return eta * METABOLIC_RATE * (cpa * (CORE_TEMP - Ta) + lh * coef * (CORE_VAPOR_PRES - Pa))


@nb.njit(cache=True, fastmath=True)
def romps_Zs(Rs: float) -> float:
    """Skin mass-transfer resistance in Pa m^2/W."""
    Rs = np.float32(Rs)
    if Rs == np.float32(0.0387):
        return np.float32(52.1)
    r2 = Rs * Rs
    return np.float32(6.0e8) * r2 * r2 * Rs


@nb.njit(cache=True, fastmath=True)
def romps_Ra(Ts: float, Ta: float) -> float:
    """Air heat-transfer resistance, exposed skin, K m^2/W."""
    em = EMISSIVITY * np.float32(0.85)
    ts2 = Ts * Ts
    ta2 = Ta * Ta
    hr = em * STEFAN_BOLTZMANN * (ts2 + ta2) * (Ts + Ta)
    return 1.0 / (np.float32(17.4) + hr)


@nb.njit(cache=True, fastmath=True)
def romps_Ra_bar(Tf: float, Ta: float) -> float:
    """Air heat-transfer resistance, clothed skin, K m^2/W."""
    em = EMISSIVITY * np.float32(0.79)
    tf2 = Tf * Tf
    ta2 = Ta * Ta
    hr = em * STEFAN_BOLTZMANN * (tf2 + ta2) * (Tf + Ta)
    return 1.0 / (np.float32(11.6) + hr)


@nb.njit(cache=True, fastmath=True)
def romps_Ra_un(Ts: float, Ta: float) -> float:
    """Air heat-transfer resistance, naked, K m^2/W."""
    em = EMISSIVITY * np.float32(0.80)
    ts2 = Ts * Ts
    ta2 = Ta * Ta
    hr = em * STEFAN_BOLTZMANN * (ts2 + ta2) * (Ts + Ta)
    return 1.0 / (np.float32(12.3) + hr)


@nb.njit(cache=True, fastmath=True)
def _hi_residual_skin(kind: int, x: float, Ta: float, Pa: float, aux: float) -> float:
    """Skin/clothing energy-balance residuals (replaces the find_eqvar lambdas).

    kind: 1 = exposed Ts, 2 = clothed Tf, 3 = region II/III Tf (aux=Ts_bar),
          4 = region IV Ts (aux=QmQv), 5 = region V Ts (aux=QmQv).
    """
    x = np.float32(x)
    Ta = np.float32(Ta)
    Pa = np.float32(Pa)
    aux = np.float32(aux)

    Zs0 = romps_Zs(np.float32(0.0387))
    const_0387 = np.float32(0.0387)
    const_124 = np.float32(124.0)

    if kind == 1:
        return (x - Ta) / romps_Ra(x, Ta) + (CORE_VAPOR_PRES - Pa) / (Zs0 + ZA_EXPOSED) - (CORE_TEMP - x) / const_0387
    elif kind == 2:
        return (x - Ta) / romps_Ra_bar(x, Ta) + (CORE_VAPOR_PRES - Pa) / (Zs0 + ZA_CLOTHED) - (CORE_TEMP - x) / const_0387
    elif kind == 3:
        Ts_bar = aux
        denom = (Zs0 + ZA_CLOTHED) * (x - Ta) + const_124 * romps_Ra_bar(x, Ta) * (Ts_bar - x)
        return (x - Ta) / romps_Ra_bar(x, Ta) + (CORE_VAPOR_PRES - Pa) * (x - Ta) / denom - (CORE_TEMP - Ts_bar) / const_0387
    elif kind == 4:
        QmQv = aux
        return (x - Ta) / romps_Ra_un(x, Ta) + (CORE_VAPOR_PRES - Pa) / (romps_Zs((CORE_TEMP - x) / QmQv) + ZA_NAKED) - QmQv
    else:  # kind == 5
        QmQv = aux
        return (x - Ta) / romps_Ra_un(x, Ta) + (PHI_SALT * romps_pvstar(x) - Pa) / ZA_NAKED - QmQv


@nb.njit(cache=True, fastmath=True)
def _hi_bisect_skin(kind: int, x1: float, x2: float, Ta: float, Pa: float, aux: float) -> float:
    a, b = x1, x2
    fa = _hi_residual_skin(kind, a, Ta, Pa, aux)
    fb = _hi_residual_skin(kind, b, Ta, Pa, aux)
    if fa * fb > 0.0:
        raise ValueError("Romps heat index: skin root not bracketed.")
    c = a
    tol = np.float32(1e-3)
    for _ in range(HI_MAXITER):
        c = np.float32(0.5) * (a + b)
        fc = _hi_residual_skin(kind, c, Ta, Pa, aux)
        if fb * fc > 0.0:
            b, fb = c, fc
        else:
            a, fa = c, fc
        if abs(a - b) < tol:
            return c
    return c


@nb.njit(cache=True, fastmath=True)
def romps_find_eqvar(Ta: float, RH: float):
    """Equivalent-variable solver. Returns (code, phi, Rf, Rs, dTcdt).

    code -> region: 1=phi (I), 2=Rf (II/III), 3=Rs (IV), 4=Rs* (V), 5=dTcdt (VI).
    """
    Ta = np.float32(Ta)
    RH = np.float32(RH)
    Pa = RH * romps_pvstar(Ta)
    phi = np.float32(0.84)
    Rf = np.float32(0.0)
    Rs = np.float32(0.0387)
    dTcdt = np.float32(0.0)
    code = 0
    const_0387 = np.float32(0.0387)

    Zs0 = romps_Zs(const_0387)
    m = (CORE_VAPOR_PRES - Pa) / (Zs0 + ZA_EXPOSED)
    m_bar = (CORE_VAPOR_PRES - Pa) / (Zs0 + ZA_CLOTHED)

    Ts = _hi_bisect_skin(1, max(np.float32(0.0), min(CORE_TEMP, Ta) - const_0387 * abs(m)),
                         max(CORE_TEMP, Ta) + const_0387 * abs(m), Ta, Pa, np.float32(0.0))
    Tf = _hi_bisect_skin(2, max(np.float32(0.0), min(CORE_TEMP, Ta) - const_0387 * abs(m_bar)),
                         max(CORE_TEMP, Ta) + const_0387 * abs(m_bar), Ta, Pa, np.float32(0.0))

    QmQv = METABOLIC_RATE - romps_Qv(Ta, Pa)
    flux1 = QmQv - (np.float32(1.0) - phi) * (CORE_TEMP - Ts) / const_0387
    flux2 = flux1 - phi * (CORE_TEMP - Tf) / const_0387

    if flux1 <= np.float32(0.0):                            # region I
        code = 1
        phi = np.float32(1.0) - QmQv * const_0387 / (CORE_TEMP - Ts)
        Rf = np.inf
    elif flux2 <= np.float32(0.0):                          # region II & III
        code = 2
        Ts_bar = CORE_TEMP - QmQv * const_0387 / phi + (np.float32(1.0) / phi - np.float32(1.0)) * (CORE_TEMP - Ts)
        Tf = _hi_bisect_skin(3, Ta, Ts_bar, Ta, Pa, Ts_bar)
        Rf = romps_Ra_bar(Tf, Ta) * (Ts_bar - Tf) / (Tf - Ta)
    else:                                                    # region IV, V, VI
        Rf = np.float32(0.0)
        flux3 = QmQv - (CORE_TEMP - Ta) / romps_Ra_un(CORE_TEMP, Ta) - (PHI_SALT * romps_pvstar(CORE_TEMP) - Pa) / ZA_NAKED
        if flux3 < np.float32(0.0):                          # region IV, V
            Ts = _hi_bisect_skin(4, np.float32(0.0), CORE_TEMP, Ta, Pa, QmQv)
            Rs = (CORE_TEMP - Ts) / QmQv
            code = 3
            Ps = CORE_VAPOR_PRES - (CORE_VAPOR_PRES - Pa) * romps_Zs(Rs) / (romps_Zs(Rs) + ZA_NAKED)
            if Ps > PHI_SALT * romps_pvstar(Ts):             # region V
                Ts = _hi_bisect_skin(5, np.float32(0.0), CORE_TEMP, Ta, Pa, QmQv)
                Rs = (CORE_TEMP - Ts) / QmQv
                code = 4
        else:                                                # region VI
            Rs = np.float32(0.0)
            # 1/C with C = M*cpc/A = 150582.19 J/K/m^2 (core heat capacity per area)
            dTcdt = flux3 / np.float32(150582.19181485035)
            code = 5
    return (code, phi, Rf, Rs, dTcdt)


@nb.njit(cache=True, fastmath=True)
def _hi_residual_T(code: int, T: float, eqvar: float) -> float:
    """Heat-index temperature residual (replaces the find_T lambdas)."""
    Pa0 = np.float32(1.6e3)
    if code == 1:
        return romps_find_eqvar(T, 1.0)[1] - eqvar
    elif code == 2:
        return romps_find_eqvar(T, min(1.0, Pa0 / romps_pvstar(T)))[2] - eqvar
    elif code == 3 or code == 4:
        return romps_find_eqvar(T, Pa0 / romps_pvstar(T))[3] - eqvar
    else:  # code == 5
        return romps_find_eqvar(T, Pa0 / romps_pvstar(T))[4] - eqvar


@nb.njit(cache=True, fastmath=True)
def _hi_bisect_T(code: int, x1: float, x2: float, eqvar: float) -> float:
    a = np.float32(x1)
    b = np.float32(x2)
    fa = _hi_residual_T(code, a, eqvar)
    fb = _hi_residual_T(code, b, eqvar)

    if fa * fb > np.float32(0.0):
        return (a + b) / np.float32(2.0)

    c = a
    tol = np.float32(1e-3)
    half = np.float32(0.5)
    for _ in range(HI_MAXITER):
        c = half * (a + b)
        fc = _hi_residual_T(code, c, eqvar)
        if fb * fc > np.float32(0.0):
            b, fb = c, fc
        else:
            a, fa = c, fc
        if abs(a - b) < tol:
            return c
    return c


@nb.njit(cache=True, fastmath=True)
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
    Ta = np.float32(Ta)
    RH = np.float32(RH)
    if Ta == np.float32(0.0):
        return np.float32(0.0)
    e = romps_find_eqvar(Ta, RH)
    code = e[0]
    # Per region, select the equivalent variable (tuple index) and invert it over
    # the bracketing temperature range to the heat-index temperature (K).
    if code == 1:
        return _hi_bisect_T(code, np.float32(0.0), np.float32(240.0), e[1])
    elif code == 2:
        return _hi_bisect_T(code, np.float32(230.0), np.float32(300.0), e[2])
    elif code == 3 or code == 4:
        return _hi_bisect_T(code, np.float32(295.0), np.float32(350.0), e[3])
    else:
        return _hi_bisect_T(code, np.float32(340.0), np.float32(1000.0), e[4])


@nb.njit(cache=True, fastmath=True)
def _vapor_pressure_hpa(huss: float) -> float:
    """Convert specific humidity to vapor pressure in hPa."""
    huss = np.float32(huss)
    epsilon = np.float32(0.622)
    p_total = np.float32(1013.25)
    return huss * p_total / (epsilon + (np.float32(1.0) - epsilon) * huss)


@nb.njit(cache=True, fastmath=True)
def _celsius_to_fahrenheit(temp_c: float) -> float:
    """Convert temperature from Celsius to Fahrenheit."""
    return (temp_c * np.float32(1.8)) + np.float32(32.0)


@nb.njit(cache=True, fastmath=True)
def get_aorc_romps_heat_index_scalar(air_temp: float, specific_humid: float) -> float:
    """Scalar Romps heat index (for use by vectorized wrapper)."""
    if air_temp < np.float32(270.0) or specific_humid < np.float32(0.0):
        return air_temp

    pvstar = romps_pvstar(air_temp)
    if pvstar <= np.float32(0.0):
        return air_temp

    vapor_pres_pa = _vapor_pressure_hpa(specific_humid) * np.float32(100.0)
    rel_humid = vapor_pres_pa / pvstar
    if rel_humid < np.float32(0.0):
        rel_humid = np.float32(0.0)
    elif rel_humid > np.float32(1.0):
        rel_humid = np.float32(1.0)

    hi_kelvin = romps_heatindex(air_temp, rel_humid)
    return _celsius_to_fahrenheit(hi_kelvin - TRIPLE_TEMP)


def get_aorc_romps_heat_index(air_temp, specific_humid):
    """
    Calculates the extended Romps (2022) heat index from AORC data.
    Handles both scalar and array inputs via Numba vectorization.

    :param air_temp: Air temperature in Kelvin (scalar or array)
    :param specific_humid: Specific humidity in kg/kg (scalar or array)
    :return: Heat index in degrees Fahrenheit (scalar or array)
    """
    import numpy as _np
    if _np.isscalar(air_temp) and _np.isscalar(specific_humid):
        return get_aorc_romps_heat_index_scalar(float(air_temp), float(specific_humid))
    else:
        return _np.vectorize(get_aorc_romps_heat_index_scalar)(air_temp, specific_humid)
