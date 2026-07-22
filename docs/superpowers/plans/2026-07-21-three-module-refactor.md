# AORC Heat Pipeline Three-Module Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the AORC heat pipeline into `core.py` (scalar science), `pipeline.py` (xarray/dask/zarr orchestration), and `cli.py` (argparse front end), writing five heat metrics into one incrementally extensible zarr store.

**Architecture:** `core.py` holds numba-compiled scalar formulations in literature units with no knowledge of AORC. `pipeline.py` opens one year of AORC at a time, builds a small set of shared intermediate quantities once, feeds them to whichever metrics were requested, reduces 24-hour chunks to daily min/mean/max, and writes each year into a time region of a single output store. `cli.py` parses arguments, starts a Dask `LocalCluster`, and calls `pipeline.run`.

**Tech Stack:** Python 3.14, numba, numpy, xarray, dask.distributed, zarr (v2 format), flox, s3fs, geopandas, pytest.

> **Status: executed.** All nine tasks are implemented on `pipeline-refactor`.
> Four decisions in this plan were revised during implementation because they
> were wrong or incomplete; the spec records the reasoning. If this plan and the
> code disagree, the code is right.
>
> - **Spatial chunking.** This plan leaves it at the source store's native
>   values. That breaks the first region write: the Texas bbox starts at
>   latitude 701 / longitude 2803, both prime, so the cropped array's first
>   chunk is a partial remainder. The crop is now snapped outward to native
>   chunk boundaries instead.
> - **`wet_bulb_temperature`.** `fastmath=True` sets LLVM's `nnan`, which folded
>   away the clamp's `if/elif` and turned masked NaN cells into a finite
>   `T - 40`. The flag set now omits `nnan`. A NaN early-out and stall detection
>   were also added — masked cells were running the full 50-iteration cap.
> - **Completion tracking.** This plan defers detecting incompletely written
>   variables. That left a crash silently reporting success over all-NaN data,
>   so per-year completion is now recorded in a store attribute.
> - **Precision.** Raw AORC variables are float64 on disk and are now cast to
>   float32 on read.

## Global Constraints

- **Design spec:** `docs/superpowers/specs/2026-07-21-three-module-refactor-design.md`. Read it before starting.
- **Test environment is Docker.** There is no local Python environment with the dependencies. Build once with `docker build -t aorc .`, then run tests as `docker run --rm -v "$PWD":/project -w /project aorc python -m pytest tests/ -v`.
- **All `core.py` functions compute in float32.** Follow the established pattern: bare `@nb.vectorize(target="cpu", cache=True, fastmath=True)` with no explicit signature list, and coerce each argument with `np.float32(...)` as the first statement. This keeps the body float32 regardless of caller dtype.
- **Float32 test tolerances.** float32 carries ~7 decimal digits. Use `rel=1e-5` or `abs=1e-4` when comparing against hand-computed float64 expectations. Do not use tighter tolerances; they will fail.
- **`core.py` imports nothing from the project.** Only `numpy` and `numba`.
- **All metric outputs are degrees Fahrenheit** (`units: "deg_F"`), matching the existing pipeline.
- **Zarr stores are written with `zarr_format=2, consolidated=True`.**
- **No new runtime dependencies.** Everything needed is already in `requirements.txt`.
- **Scope excludes** the Lu & Romps (2022) heat index and its lookup tables. Do not port them.
- **xarray version floor:** `xr.date_range(..., use_cftime=True)` requires xarray ≥ 2023.11. `requirements.txt` is unpinned, so the image gets a current release. If `xr.date_range` is missing, `xr.cftime_range` is the older equivalent.

---

### Task 1: Package scaffolding and removal of superseded modules

Establishes the importable package and clears out the code the refactor replaces. Old modules go first so the test suite never sits in a half-migrated state.

**Files:**
- Create: `src/aorc_heat/__init__.py`
- Create: `pyproject.toml`
- Rename: `src/aorc-heat/` → `src/aorc_heat/` (the three files inside are empty)
- Modify: `Dockerfile`
- Rewrite: `tests/conftest.py`
- Create: `tests/test_package.py`
- Delete: `metrics.py`, `run.py`, `post_proc.py`, `benchmark.py`, `shpmask.py`, `romps_heat_index.py`, `lookup_tables.py`, `lut_runtime.py`, `romps_hi_lut.npz`, `wbt_lut.npz`, `tests/test_metrics.py`, `tests/test_romps_heatindex.py`, `tests/test_lookup_tables.py`

**Interfaces:**
- Consumes: nothing.
- Produces: the importable package `aorc_heat`, and the console script `aorc-heat` pointing at `aorc_heat.cli:main`.

- [ ] **Step 1: Remove the superseded modules and their tests**

`shpmask.py` is deleted here rather than moved; Task 5 recreates its contents as `src/aorc_heat/mask.py` with fixes. All of these remain recoverable from git history.

```bash
cd /home/oxygen/Documents/AORC-Heat-Pipeline
git rm -q metrics.py run.py post_proc.py benchmark.py shpmask.py \
          romps_heat_index.py lookup_tables.py lut_runtime.py \
          tests/test_metrics.py tests/test_romps_heatindex.py tests/test_lookup_tables.py
rm -f romps_hi_lut.npz wbt_lut.npz
rm -rf __pycache__ tests/__pycache__ .pytest_cache
```

- [ ] **Step 2: Rename the package directory**

The existing name `src/aorc-heat` contains a hyphen, which is not a legal Python package name. The three files inside are empty.

```bash
git mv src/aorc-heat src/aorc_heat
```

- [ ] **Step 3: Create the package init**

Create `src/aorc_heat/__init__.py`:

```python
"""Heat-stress metrics derived from the NOAA AORC hourly reanalysis archive."""

__all__ = ["core", "pipeline", "cli", "mask"]
```

- [ ] **Step 4: Create pyproject.toml**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "aorc-heat"
version = "0.1.0"
description = "Heat-stress metrics derived from the NOAA AORC hourly reanalysis archive"
requires-python = ">=3.11"
dynamic = ["dependencies"]

[project.scripts]
aorc-heat = "aorc_heat.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.dynamic]
dependencies = { file = ["requirements.txt"] }
```

- [ ] **Step 5: Update the Dockerfile**

The image must install the package so `import aorc_heat` works. Replace the last two lines of `Dockerfile` (`RUN pip install ...` and `CMD ...`) with:

```dockerfile
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install --no-deps -e .

CMD ["aorc-heat", "--help"]
```

`-e` makes the install point at `/project/src`, which stays correct when the host directory is bind-mounted over `/project` at run time. `--no-deps` avoids re-resolving what `requirements.txt` already installed.

- [ ] **Step 6: Rewrite conftest.py**

The `sys.path` manipulation is no longer needed now that the package is installed. This file becomes the home for shared test fixtures, which later tasks add to. Replace the entire contents of `tests/conftest.py`:

```python
"""Shared fixtures for the AORC-Heat-Pipeline test suite.

The package is installed into the image with `pip install -e .`, so tests import
`aorc_heat` directly with no path manipulation.
"""
```

- [ ] **Step 7: Write the failing test**

Create `tests/test_package.py`:

```python
"""Confirms the package is installed and importable in the test environment."""


def test_package_imports():
    import aorc_heat

    assert aorc_heat.__all__ == ["core", "pipeline", "cli", "mask"]
```

- [ ] **Step 8: Rebuild the image and run the test**

```bash
docker build -t aorc . \
  && docker run --rm -v "$PWD":/project -w /project aorc python -m pytest tests/ -v
```

Expected: `tests/test_package.py::test_package_imports PASSED`, 1 passed. No collection errors from deleted modules.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "refactor: scaffold aorc_heat package, remove superseded modules"
```

---

### Task 2: core.py — conversions and moisture primitives

**Files:**
- Create: `src/aorc_heat/core.py`
- Create: `tests/test_core.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces, all `@nb.vectorize` scalar kernels:
  - `celsius_to_fahrenheit(air_temperature_celsius) -> float32` degrees F
  - `fahrenheit_to_celsius(air_temperature_fahrenheit) -> float32` degrees C
  - `kelvin_to_celsius(air_temperature_kelvin) -> float32` degrees C
  - `saturation_vapor_pressure(air_temperature_celsius) -> float32` hPa
  - `saturation_vapor_pressure_slope(air_temperature_celsius) -> float32` hPa per degree C
  - `vapor_pressure(specific_humidity, air_pressure_hpa) -> float32` hPa
  - `relative_humidity(vapor_pressure_hpa, saturation_vapor_pressure_hpa) -> float32` percent
  - Module constant `DRY_AIR_TO_VAPOR_MOLAR_MASS_RATIO = np.float32(0.622)`

**Naming note.** The design spec's table listed these with return-unit suffixes (`saturation_vapor_pressure_hpa`, `vapor_pressure_hpa`). Those names collide with the *parameter* names of the metric functions in Task 3, which take `vapor_pressure_hpa` and `saturation_vapor_pressure_hpa` as arguments. Shadowing a module-level function with a parameter of the same name is legal but hostile to audit, so function names describe what is computed and parameter names carry the units. Return units live in the docstrings.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_core.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
docker run --rm -v "$PWD":/project -w /project aorc python -m pytest tests/test_core.py -v
```

Expected: collection error, `ModuleNotFoundError` or `ImportError: cannot import name 'core' from 'aorc_heat'`.

- [ ] **Step 3: Write the implementation**

Create `src/aorc_heat/core.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
docker run --rm -v "$PWD":/project -w /project aorc python -m pytest tests/test_core.py -v
```

Expected: all tests pass. The first run is slow while numba compiles.

- [ ] **Step 5: Commit**

```bash
git add src/aorc_heat/core.py tests/test_core.py
git commit -m "feat: add core conversions and moisture primitives"
```

---

### Task 3: core.py — closed-form heat metrics

**Files:**
- Modify: `src/aorc_heat/core.py` (append)
- Modify: `tests/test_core.py` (append)

**Interfaces:**
- Consumes: nothing at runtime; the tests use `core.saturation_vapor_pressure` from Task 2.
- Produces:
  - `heat_index(air_temperature_fahrenheit, relative_humidity_percent) -> float32` degrees F
  - `apparent_temperature(air_temperature_celsius, vapor_pressure_hpa, wind_speed_ms) -> float32` degrees C
  - `humidex(air_temperature_celsius, vapor_pressure_hpa) -> float32` degrees C
  - `simplified_wbgt(air_temperature_celsius, vapor_pressure_hpa) -> float32` degrees C

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_core.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
docker run --rm -v "$PWD":/project -w /project aorc python -m pytest tests/test_core.py -v -k "heat_index or apparent or humidex or wbgt"
```

Expected: FAIL with `AttributeError: module 'aorc_heat.core' has no attribute 'heat_index'`.

- [ ] **Step 3: Write the implementation**

Append to `src/aorc_heat/core.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
docker run --rm -v "$PWD":/project -w /project aorc python -m pytest tests/test_core.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/aorc_heat/core.py tests/test_core.py
git commit -m "feat: add closed-form heat metrics to core"
```

---

### Task 4: core.py — wet-bulb temperature

**Files:**
- Modify: `src/aorc_heat/core.py` (append)
- Modify: `tests/test_core.py` (append)

**Interfaces:**
- Consumes: `core.saturation_vapor_pressure` and `core.saturation_vapor_pressure_slope` from Task 2. A `@nb.vectorize` function is callable from inside another one; this is the pattern the previous `metrics.py` used.
- Produces: `wet_bulb_temperature(air_temperature_celsius, specific_humidity, air_pressure_hpa) -> float32` degrees C

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_core.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
docker run --rm -v "$PWD":/project -w /project aorc python -m pytest tests/test_core.py -v -k wet_bulb
```

Expected: FAIL with `AttributeError: module 'aorc_heat.core' has no attribute 'wet_bulb_temperature'`.

- [ ] **Step 3: Write the implementation**

Append to `src/aorc_heat/core.py`:

```python
WET_BULB_MAX_ITERATIONS = 50
WET_BULB_TOLERANCE_CELSIUS = np.float32(1.0e-4)

#: Largest wet-bulb depression the solver will accept, in degrees Celsius. Acts
#: as a guard rail so a Newton step cannot run away on pathological input.
WET_BULB_MAX_DEPRESSION_CELSIUS = np.float32(40.0)

SPECIFIC_HEAT_DRY_AIR = np.float32(1005.7)    # J/kg/K
SPECIFIC_HEAT_WATER_VAPOR = np.float32(1875.0)  # J/kg/K
LATENT_HEAT_AT_ZERO_CELSIUS = np.float32(2.501e6)  # J/kg
LATENT_HEAT_SLOPE = np.float32(-2370.0)       # J/kg per degree Celsius


@nb.vectorize(target="cpu", cache=True, fastmath=True)
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

    epsilon = DRY_AIR_TO_VAPOR_MOLAR_MASS_RATIO
    one_minus_epsilon = np.float32(1.0) - epsilon
    specific_heat_moist_air = SPECIFIC_HEAT_DRY_AIR + (
        SPECIFIC_HEAT_WATER_VAPOR - SPECIFIC_HEAT_DRY_AIR
    ) * humidity

    wet_bulb = temperature

    for _ in range(WET_BULB_MAX_ITERATIONS):
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

    return wet_bulb
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
docker run --rm -v "$PWD":/project -w /project aorc python -m pytest tests/test_core.py -v
```

Expected: all tests pass. `core.py` is now complete.

- [ ] **Step 5: Commit**

```bash
git add src/aorc_heat/core.py tests/test_core.py
git commit -m "feat: add wet-bulb temperature solver to core"
```

---

### Task 5: mask.py — Texas region masking

Ports the deleted `shpmask.py` with one correctness fix: the original built the mask with dimensions `x` and `y` while `run.py` reduced over `latitude` and `longitude`, which cannot work. The port names the dimensions `latitude` and `longitude` to match the AORC grid.

**Files:**
- Create: `src/aorc_heat/mask.py`
- Create: `tests/test_mask.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `Region` frozen dataclass with fields `mask: xr.DataArray`, `latitude_slice: slice`, `longitude_slice: slice`
  - `build_mask(latitudes, longitudes, shapefile_path) -> xr.DataArray` — boolean, dims `("latitude", "longitude")`
  - `crop_to_bounding_box(mask) -> Region`
  - `texas_region(latitudes, longitudes, mask_path=TEXAS_MASK_PATH, shapefile_path=TEXAS_SHAPEFILE_PATH) -> Region`
  - Module constants `TEXAS_MASK_PATH`, `TEXAS_SHAPEFILE_PATH`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mask.py`:

```python
"""Unit tests for region masking.

The point-in-polygon routine is checked against a unit square, where membership
is obvious by inspection, rather than against the Texas shapefile -- which is
gitignored output and not available in a clean checkout.
"""
import numpy as np
import pytest
import xarray as xr

from aorc_heat import mask as mask_module


UNIT_SQUARE = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])


@pytest.mark.parametrize(
    "longitude, latitude, inside",
    [
        (0.5, 0.5, True),    # centre
        (0.1, 0.9, True),    # near a corner, still inside
        (1.5, 0.5, False),   # east of the square
        (0.5, 1.5, False),   # north of the square
        (-0.5, 0.5, False),  # west of the square
    ],
)
def test_point_in_polygon(longitude, latitude, inside):
    assert mask_module.point_in_polygon(longitude, latitude, UNIT_SQUARE) == inside


def test_mask_from_polygon_has_grid_dimensions():
    latitudes = np.linspace(-1.0, 2.0, 13, dtype=np.float64)
    longitudes = np.linspace(-1.0, 2.0, 13, dtype=np.float64)
    result = mask_module.mask_from_polygon(latitudes, longitudes, UNIT_SQUARE)

    assert result.dims == ("latitude", "longitude")
    assert result.shape == (13, 13)
    assert result.dtype == bool
    # Every selected cell must lie inside the unit square.
    selected = result.where(result, drop=True)
    assert float(selected.latitude.min()) >= 0.0
    assert float(selected.latitude.max()) <= 1.0


def test_crop_to_bounding_box_trims_empty_margins():
    latitudes = np.arange(10.0, 20.0)
    longitudes = np.arange(100.0, 110.0)
    values = np.zeros((10, 10), dtype=bool)
    values[3:6, 4:8] = True
    full_mask = xr.DataArray(
        values,
        dims=("latitude", "longitude"),
        coords={"latitude": latitudes, "longitude": longitudes},
    )

    region = mask_module.crop_to_bounding_box(full_mask)

    assert region.latitude_slice == slice(3, 6)
    assert region.longitude_slice == slice(4, 8)
    assert region.mask.shape == (3, 4)
    assert bool(region.mask.all())


def test_crop_to_bounding_box_rejects_empty_mask():
    empty = xr.DataArray(
        np.zeros((4, 4), dtype=bool),
        dims=("latitude", "longitude"),
        coords={"latitude": np.arange(4.0), "longitude": np.arange(4.0)},
    )
    with pytest.raises(ValueError, match="no selected cells"):
        mask_module.crop_to_bounding_box(empty)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
docker run --rm -v "$PWD":/project -w /project aorc python -m pytest tests/test_mask.py -v
```

Expected: collection error, `ImportError: cannot import name 'mask' from 'aorc_heat'`.

- [ ] **Step 3: Write the implementation**

Create `src/aorc_heat/mask.py`:

```python
"""Spatial masking of the AORC grid to the Texas state boundary.

The mask NetCDF is gitignored, so it is generated from the bundled shapefile on
first use and cached to disk. Because the AORC domain covers the whole of CONUS
and Texas occupies a small part of it, the mask is cropped to its own bounding
box; the index slices used for that crop travel with it so the same subsetting
can be applied to a full-domain dataset.
"""

from dataclasses import dataclass
from pathlib import Path

import numba as nb
import numpy as np
import xarray as xr

TEXAS_MASK_PATH = Path("boundary_data/texas_mask.nc")
TEXAS_SHAPEFILE_PATH = Path("boundary_data/texas_shapefile/State_Boundary.shp")

MASK_ATTRIBUTES = {
    "description": "Boolean mask of AORC grid points inside the Texas state boundary",
    "shapefile_url": "https://gis-txdot.opendata.arcgis.com/datasets/texas-state-boundary/explore",
    "credit": "Point-in-polygon routines adapted from a Stack Overflow post by user 'epifanio'",
    "credit_url": "https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python",
}


@dataclass(frozen=True)
class Region:
    """A boolean mask cropped to its own bounding box.

    :param mask: Boolean DataArray over dims ("latitude", "longitude")
    :param latitude_slice: Index slice that crops a full-domain latitude axis
    :param longitude_slice: Index slice that crops a full-domain longitude axis
    """

    mask: xr.DataArray
    latitude_slice: slice
    longitude_slice: slice


# --------------------------------------------------------------------------
# Point-in-polygon. Adapted from the Stack Overflow post credited above.
# --------------------------------------------------------------------------
@nb.njit(cache=True)
def point_in_polygon(longitude, latitude, polygon):
    """Ray-casting test for a single point against a closed polygon.

    :param longitude: Point longitude in degrees east
    :param latitude: Point latitude in degrees north
    :param polygon: (N, 2) array of polygon vertices as (longitude, latitude)
    :return: True when the point lies inside the polygon
    """
    vertex_count = len(polygon)
    inside = False
    previous_longitude, previous_latitude = polygon[0]
    for index in range(vertex_count + 1):
        current_longitude, current_latitude = polygon[index % vertex_count]
        if latitude > min(previous_latitude, current_latitude):
            if latitude <= max(previous_latitude, current_latitude):
                if longitude <= max(previous_longitude, current_longitude):
                    if previous_latitude != current_latitude:
                        crossing = (latitude - previous_latitude) * (
                            current_longitude - previous_longitude
                        ) / (current_latitude - previous_latitude) + previous_longitude
                        if previous_longitude == current_longitude or longitude <= crossing:
                            inside = not inside
                    elif previous_longitude == current_longitude:
                        inside = not inside
        previous_longitude, previous_latitude = current_longitude, current_latitude
    return inside


@nb.njit(parallel=True, cache=True)
def _grid_in_polygon(latitudes, longitudes, polygon):
    """Point-in-polygon over a full lat/lon grid."""
    result = np.empty((latitudes.size, longitudes.size), dtype=nb.boolean)
    for row in nb.prange(latitudes.size):
        for column in range(longitudes.size):
            result[row, column] = point_in_polygon(longitudes[column], latitudes[row], polygon)
    return result


def mask_from_polygon(latitudes, longitudes, polygon):
    """Build a boolean mask over a lat/lon grid from polygon vertices.

    :param latitudes: 1-D array of latitudes in degrees north
    :param longitudes: 1-D array of longitudes in degrees east
    :param polygon: (N, 2) array of polygon vertices as (longitude, latitude)
    :return: Boolean DataArray over dims ("latitude", "longitude")
    """
    values = _grid_in_polygon(
        np.ascontiguousarray(latitudes, dtype=np.float64),
        np.ascontiguousarray(longitudes, dtype=np.float64),
        np.ascontiguousarray(polygon, dtype=np.float64),
    )
    return xr.DataArray(
        values,
        dims=("latitude", "longitude"),
        coords={"latitude": latitudes, "longitude": longitudes},
        name="mask",
        attrs=MASK_ATTRIBUTES,
    )


def build_mask(latitudes, longitudes, shapefile_path):
    """Build a boundary mask by reading a shapefile.

    :param latitudes: 1-D array of latitudes in degrees north
    :param longitudes: 1-D array of longitudes in degrees east
    :param shapefile_path: Path to a shapefile whose first geometry is the boundary
    :return: Boolean DataArray over dims ("latitude", "longitude")
    """
    import geopandas

    shapefile_path = Path(shapefile_path)
    if not shapefile_path.is_file():
        raise FileNotFoundError(f"Boundary shapefile not found at {shapefile_path}")

    boundary = geopandas.read_file(shapefile_path).to_crs(crs=4326)
    polygon = np.array(boundary.geometry[0].exterior.coords.xy).T
    return mask_from_polygon(latitudes, longitudes, polygon)


def crop_to_bounding_box(mask):
    """Crop a mask to the bounding box of its selected cells.

    :param mask: Boolean DataArray over dims ("latitude", "longitude")
    :return: A Region holding the cropped mask and the slices that produced it
    """
    latitude_indices = np.flatnonzero(mask.any("longitude").values)
    longitude_indices = np.flatnonzero(mask.any("latitude").values)
    if latitude_indices.size == 0 or longitude_indices.size == 0:
        raise ValueError("Cannot crop a mask with no selected cells.")

    latitude_slice = slice(int(latitude_indices[0]), int(latitude_indices[-1]) + 1)
    longitude_slice = slice(int(longitude_indices[0]), int(longitude_indices[-1]) + 1)
    return Region(
        mask=mask.isel(latitude=latitude_slice, longitude=longitude_slice),
        latitude_slice=latitude_slice,
        longitude_slice=longitude_slice,
    )


def texas_region(
    latitudes,
    longitudes,
    mask_path=TEXAS_MASK_PATH,
    shapefile_path=TEXAS_SHAPEFILE_PATH,
):
    """Load the Texas mask, generating and caching it from the shapefile if absent.

    :param latitudes: 1-D array of AORC latitudes in degrees north
    :param longitudes: 1-D array of AORC longitudes in degrees east
    :param mask_path: Where the generated mask is cached
    :param shapefile_path: Source shapefile used when the cache is missing
    :return: A Region cropped to the Texas bounding box
    """
    mask_path = Path(mask_path)
    if mask_path.is_file():
        mask = xr.open_dataset(mask_path)["mask"].astype(bool)
        if not np.array_equal(mask.latitude.values, latitudes) or not np.array_equal(
            mask.longitude.values, longitudes
        ):
            raise ValueError(
                f"Cached mask at {mask_path} does not match the AORC grid. "
                "Delete it to regenerate."
            )
    else:
        mask = build_mask(latitudes, longitudes, shapefile_path)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask.to_dataset(name="mask").to_netcdf(mask_path)

    return crop_to_bounding_box(mask)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
docker run --rm -v "$PWD":/project -w /project aorc python -m pytest tests/test_mask.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/aorc_heat/mask.py tests/test_mask.py
git commit -m "feat: add Texas region masking with lat/lon dimension fix"
```

---

### Task 6: pipeline.py — metric registry and daily reduction

**Files:**
- Create: `src/aorc_heat/pipeline.py`
- Create: `tests/test_pipeline.py`
- Modify: `tests/conftest.py` (add the synthetic dataset fixture)

**Interfaces:**
- Consumes: `aorc_heat.core` (Tasks 2–4), `aorc_heat.mask.Region` (Task 5).
- Produces:
  - Constants `AIR_TEMPERATURE`, `SPECIFIC_HUMIDITY`, `SURFACE_PRESSURE`, `EASTWARD_WIND`, `NORTHWARD_WIND`, `DAILY_STATISTICS`, `HOURS_PER_DAY`, `GLOBAL_ATTRIBUTES`, `AORC_S3_BASE_PATH`
  - `MetricSpec` frozen dataclass with fields `inputs: tuple[str, ...]`, `compute: Callable[[dict], xr.DataArray]`
  - `METRICS: dict[str, MetricSpec]` keyed by `heat_index`, `apparent_temperature`, `humidex`, `simplified_wbgt`, `wet_bulb_temperature`
  - `required_variables(metric_names) -> list[str]`
  - `prepare_dataset(dataset, region) -> xr.Dataset`
  - `daily_metrics(aorc_dataset, metric_names) -> xr.Dataset`

- [ ] **Step 1: Add the synthetic dataset fixture**

Append to `tests/conftest.py`:

```python
import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def synthetic_aorc_dataset():
    """Two days of hourly AORC-shaped data on a 2x2 grid.

    Values are physically plausible for a Texas summer so that every metric
    exercises its main branch rather than an edge case. Chunked at 24 hours to
    match what `prepare_dataset` produces.
    """
    hours = 48
    times = xr.date_range("2000-07-01", periods=hours, freq="h", use_cftime=True, calendar="standard")
    latitudes = np.array([30.0, 31.0])
    longitudes = np.array([-100.0, -99.0])
    shape = (hours, latitudes.size, longitudes.size)

    generator = np.random.default_rng(seed=20260721)

    def field(low, high):
        return generator.uniform(low, high, size=shape).astype(np.float32)

    dataset = xr.Dataset(
        {
            "TMP_2maboveground": (("time", "latitude", "longitude"), field(295.0, 315.0)),
            "SPFH_2maboveground": (("time", "latitude", "longitude"), field(0.005, 0.018)),
            "PRES_surface": (("time", "latitude", "longitude"), field(98000.0, 101500.0)),
            "UGRD_10maboveground": (("time", "latitude", "longitude"), field(-8.0, 8.0)),
            "VGRD_10maboveground": (("time", "latitude", "longitude"), field(-8.0, 8.0)),
        },
        coords={"time": times, "latitude": latitudes, "longitude": longitudes},
    )
    return dataset.chunk({"time": 24})
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_pipeline.py`:

```python
"""Unit tests for the AORC metric pipeline.

These run entirely against an in-memory synthetic dataset -- no S3, no network.
"""
import numpy as np
import pytest
import xarray as xr

from aorc_heat import core, pipeline


ALL_METRICS = [
    "heat_index",
    "apparent_temperature",
    "humidex",
    "simplified_wbgt",
    "wet_bulb_temperature",
]


def test_registry_covers_the_five_in_scope_metrics():
    assert sorted(pipeline.METRICS) == sorted(ALL_METRICS)


def test_required_variables_only_reads_wind_for_apparent_temperature():
    without_wind = pipeline.required_variables(["humidex", "wet_bulb_temperature"])
    assert pipeline.EASTWARD_WIND not in without_wind
    assert pipeline.NORTHWARD_WIND not in without_wind

    with_wind = pipeline.required_variables(["apparent_temperature"])
    assert pipeline.EASTWARD_WIND in with_wind
    assert pipeline.NORTHWARD_WIND in with_wind


def test_required_variables_always_reads_temperature_humidity_pressure():
    for name in ALL_METRICS:
        variables = pipeline.required_variables([name])
        assert pipeline.AIR_TEMPERATURE in variables
        assert pipeline.SPECIFIC_HUMIDITY in variables
        assert pipeline.SURFACE_PRESSURE in variables


def test_required_variables_rejects_unknown_metric():
    with pytest.raises(ValueError, match="unknown metric"):
        pipeline.required_variables(["not_a_metric"])


def test_daily_metrics_returns_only_requested_metrics(synthetic_aorc_dataset):
    result = pipeline.daily_metrics(synthetic_aorc_dataset, ["humidex"])

    assert sorted(result.data_vars) == ["humidex_max", "humidex_mean", "humidex_min"]
    for name in ("heat_index_mean", "wet_bulb_temperature_mean"):
        assert name not in result.data_vars


def test_daily_metrics_produces_three_statistics_per_metric(synthetic_aorc_dataset):
    result = pipeline.daily_metrics(synthetic_aorc_dataset, ALL_METRICS)
    assert len(result.data_vars) == len(ALL_METRICS) * 3
    assert result.sizes["time"] == 2
    assert result.sizes["latitude"] == 2
    assert result.sizes["longitude"] == 2


def test_daily_metrics_outputs_are_float32(synthetic_aorc_dataset):
    result = pipeline.daily_metrics(synthetic_aorc_dataset, ALL_METRICS).compute()
    for variable in result.data_vars.values():
        assert variable.dtype == np.float32


def test_daily_metrics_reduction_matches_manual_numpy(synthetic_aorc_dataset):
    result = pipeline.daily_metrics(synthetic_aorc_dataset, ["humidex"]).compute()

    loaded = synthetic_aorc_dataset.compute()
    air_temperature_celsius = core.kelvin_to_celsius(loaded["TMP_2maboveground"].values)
    air_pressure_hpa = (loaded["PRES_surface"].values / 100.0).astype(np.float32)
    vapor = core.vapor_pressure(loaded["SPFH_2maboveground"].values, air_pressure_hpa)
    hourly = core.celsius_to_fahrenheit(core.humidex(air_temperature_celsius, vapor))

    for day in (0, 1):
        window = hourly[day * 24 : (day + 1) * 24]
        np.testing.assert_allclose(
            result["humidex_min"].isel(time=day).values, window.min(axis=0), rtol=1e-5
        )
        np.testing.assert_allclose(
            result["humidex_mean"].isel(time=day).values, window.mean(axis=0), rtol=1e-5
        )
        np.testing.assert_allclose(
            result["humidex_max"].isel(time=day).values, window.max(axis=0), rtol=1e-5
        )


def test_daily_metrics_orders_statistics_min_le_mean_le_max(synthetic_aorc_dataset):
    result = pipeline.daily_metrics(synthetic_aorc_dataset, ALL_METRICS).compute()
    for name in ALL_METRICS:
        assert bool((result[f"{name}_min"] <= result[f"{name}_mean"] + 1e-3).all())
        assert bool((result[f"{name}_mean"] <= result[f"{name}_max"] + 1e-3).all())


def test_daily_metrics_propagates_nan(synthetic_aorc_dataset):
    # Load before assigning: item assignment is not supported on dask-backed arrays.
    masked = synthetic_aorc_dataset.compute()
    masked["TMP_2maboveground"][:, 0, 0] = np.nan
    masked = masked.chunk({"time": 24})
    result = pipeline.daily_metrics(masked, ["humidex"]).compute()
    assert bool(np.isnan(result["humidex_mean"].isel(latitude=0, longitude=0)).all())
    assert not bool(np.isnan(result["humidex_mean"].isel(latitude=1, longitude=1)).any())


def test_daily_metrics_sets_variable_attributes(synthetic_aorc_dataset):
    result = pipeline.daily_metrics(synthetic_aorc_dataset, ["humidex"])
    attributes = result["humidex_mean"].attrs
    assert attributes["units"] == "deg_F"
    assert attributes["source_timestep"] == "hourly"
    assert result.attrs["source_version"] == "AORC Version 1.1"


def test_prepare_dataset_crops_masks_and_chunks(synthetic_aorc_dataset):
    from aorc_heat.mask import Region

    values = np.array([[True, False], [False, False]])
    mask = xr.DataArray(
        values,
        dims=("latitude", "longitude"),
        coords={
            "latitude": synthetic_aorc_dataset.latitude,
            "longitude": synthetic_aorc_dataset.longitude,
        },
    )
    region = Region(mask=mask, latitude_slice=slice(0, 2), longitude_slice=slice(0, 2))

    prepared = pipeline.prepare_dataset(synthetic_aorc_dataset, region)

    assert prepared.chunksizes["time"] == (24, 24)
    loaded = prepared.compute()
    assert not np.isnan(loaded["TMP_2maboveground"].isel(latitude=0, longitude=0)).any()
    assert bool(np.isnan(loaded["TMP_2maboveground"].isel(latitude=1, longitude=1)).all())
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
docker run --rm -v "$PWD":/project -w /project aorc python -m pytest tests/test_pipeline.py -v
```

Expected: collection error, `ImportError: cannot import name 'pipeline' from 'aorc_heat'`.

- [ ] **Step 4: Write the implementation**

Create `src/aorc_heat/pipeline.py`:

```python
"""Derive daily heat metrics from the NOAA AORC hourly zarr archive.

One year is processed at a time. Each raw AORC variable is read exactly once and
the intermediate quantities built from it -- air temperature in Celsius, vapour
pressure, and so on -- are shared by every metric that needs them, so the four
metrics that depend on vapour pressure resolve to a single node in the dask
graph rather than four.

Input is rechunked to 24 hours per chunk so each daily reduction group lives
entirely inside one chunk. Spatial chunking is left at the store's native
values.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import xarray as xr

from aorc_heat import core

AORC_S3_BASE_PATH = "s3://noaa-nws-aorc-v1-1-1km/"

AIR_TEMPERATURE = "TMP_2maboveground"
SPECIFIC_HUMIDITY = "SPFH_2maboveground"
SURFACE_PRESSURE = "PRES_surface"
EASTWARD_WIND = "UGRD_10maboveground"
NORTHWARD_WIND = "VGRD_10maboveground"

HOURS_PER_DAY = 24
DAILY_STATISTICS = ("min", "mean", "max")
PASCALS_PER_HECTOPASCAL = np.float32(100.0)

GLOBAL_ATTRIBUTES = {
    "description": "Heat metrics derived from NOAA NWS AORC data provided via AWS S3 Zarr Bucket",
    "source_version": "AORC Version 1.1",
    "source_url": "https://registry.opendata.aws/noaa-nws-aorc",
}


def _apply(kernel, *data_arrays):
    """Apply a `core` scalar kernel elementwise across dask-backed arrays."""
    return xr.apply_ufunc(
        kernel, *data_arrays, dask="parallelized", output_dtypes=[np.float32]
    )


def _variable_attributes(metric_name, statistic):
    """CF-style attributes for one output variable."""
    readable = metric_name.replace("_", " ")
    return {
        "units": "deg_F",
        "source_timestep": "hourly",
        "description": f"Daily {statistic} of hourly {readable} taken across 24 hours",
    }


# --------------------------------------------------------------------------
# Shared intermediates
# --------------------------------------------------------------------------
def _derived_inputs(aorc_dataset, metric_names):
    """Build the intermediate quantities the requested metrics need.

    Each quantity is built once and handed to every metric that uses it. The
    first four are unconditional because all five metrics depend on temperature,
    humidity, and pressure; relative humidity and wind speed are built only when
    the metric that needs them was requested.

    :param aorc_dataset: Dataset holding raw AORC variables
    :param metric_names: Names of the metrics that will be computed
    :return: Mapping of intermediate name to lazy DataArray
    """
    air_temperature_celsius = _apply(core.kelvin_to_celsius, aorc_dataset[AIR_TEMPERATURE])
    air_pressure_hpa = aorc_dataset[SURFACE_PRESSURE] / PASCALS_PER_HECTOPASCAL
    specific_humidity = aorc_dataset[SPECIFIC_HUMIDITY]

    derived = {
        "air_temperature_celsius": air_temperature_celsius,
        "air_pressure_hpa": air_pressure_hpa,
        "specific_humidity": specific_humidity,
        "vapor_pressure_hpa": _apply(core.vapor_pressure, specific_humidity, air_pressure_hpa),
    }

    if "heat_index" in metric_names:
        saturation = _apply(core.saturation_vapor_pressure, air_temperature_celsius)
        derived["relative_humidity_percent"] = _apply(
            core.relative_humidity, derived["vapor_pressure_hpa"], saturation
        )

    if "apparent_temperature" in metric_names:
        derived["wind_speed_ms"] = np.sqrt(
            aorc_dataset[EASTWARD_WIND] ** 2 + aorc_dataset[NORTHWARD_WIND] ** 2
        )

    return derived


# --------------------------------------------------------------------------
# Per-metric hourly computation, all returning degrees Fahrenheit
# --------------------------------------------------------------------------
def _heat_index(derived):
    return _apply(
        core.heat_index,
        _apply(core.celsius_to_fahrenheit, derived["air_temperature_celsius"]),
        derived["relative_humidity_percent"],
    )


def _apparent_temperature(derived):
    return _apply(
        core.celsius_to_fahrenheit,
        _apply(
            core.apparent_temperature,
            derived["air_temperature_celsius"],
            derived["vapor_pressure_hpa"],
            derived["wind_speed_ms"],
        ),
    )


def _humidex(derived):
    return _apply(
        core.celsius_to_fahrenheit,
        _apply(core.humidex, derived["air_temperature_celsius"], derived["vapor_pressure_hpa"]),
    )


def _simplified_wbgt(derived):
    return _apply(
        core.celsius_to_fahrenheit,
        _apply(
            core.simplified_wbgt,
            derived["air_temperature_celsius"],
            derived["vapor_pressure_hpa"],
        ),
    )


def _wet_bulb_temperature(derived):
    return _apply(
        core.celsius_to_fahrenheit,
        _apply(
            core.wet_bulb_temperature,
            derived["air_temperature_celsius"],
            derived["specific_humidity"],
            derived["air_pressure_hpa"],
        ),
    )


@dataclass(frozen=True)
class MetricSpec:
    """How to compute one metric and what raw AORC variables it needs.

    :param inputs: Raw AORC variable names that must be read
    :param compute: Callable taking the derived-inputs mapping, returning hourly degrees F
    """

    inputs: tuple
    compute: Callable


#: Every metric needs surface pressure, because `core.vapor_pressure` takes the
#: ambient pressure rather than assuming a fixed 1013.25 hPa. Wind is the only
#: variable a metric selection can avoid reading.
METRICS = {
    "heat_index": MetricSpec(
        inputs=(AIR_TEMPERATURE, SPECIFIC_HUMIDITY, SURFACE_PRESSURE),
        compute=_heat_index,
    ),
    "apparent_temperature": MetricSpec(
        inputs=(
            AIR_TEMPERATURE,
            SPECIFIC_HUMIDITY,
            SURFACE_PRESSURE,
            EASTWARD_WIND,
            NORTHWARD_WIND,
        ),
        compute=_apparent_temperature,
    ),
    "humidex": MetricSpec(
        inputs=(AIR_TEMPERATURE, SPECIFIC_HUMIDITY, SURFACE_PRESSURE),
        compute=_humidex,
    ),
    "simplified_wbgt": MetricSpec(
        inputs=(AIR_TEMPERATURE, SPECIFIC_HUMIDITY, SURFACE_PRESSURE),
        compute=_simplified_wbgt,
    ),
    "wet_bulb_temperature": MetricSpec(
        inputs=(AIR_TEMPERATURE, SPECIFIC_HUMIDITY, SURFACE_PRESSURE),
        compute=_wet_bulb_temperature,
    ),
}


def _validate_metric_names(metric_names):
    """Raise if any requested name is not in the registry."""
    unknown = [name for name in metric_names if name not in METRICS]
    if unknown:
        raise ValueError(
            f"unknown metric(s): {', '.join(sorted(unknown))}. "
            f"Available: {', '.join(sorted(METRICS))}"
        )


def required_variables(metric_names):
    """The AORC variables that must be read to compute the requested metrics.

    :param metric_names: Names of metrics from `METRICS`
    :return: Sorted list of raw AORC variable names
    """
    _validate_metric_names(metric_names)
    required = set()
    for name in metric_names:
        required.update(METRICS[name].inputs)
    return sorted(required)


def prepare_dataset(dataset, region):
    """Crop to the region bounding box, mask, and rechunk to one day per chunk.

    Chunking time in 24-hour blocks puts each daily resample group entirely
    inside one chunk, so the reduction never crosses a chunk boundary. Spatial
    chunking is left at the store's native values.

    :param dataset: Full-domain AORC dataset
    :param region: A `mask.Region` describing the area of interest
    :return: Cropped, masked, rechunked dataset
    """
    cropped = dataset.isel(
        latitude=region.latitude_slice, longitude=region.longitude_slice
    )
    return cropped.where(region.mask).chunk({"time": HOURS_PER_DAY})


def daily_metrics(aorc_dataset, metric_names):
    """Reduce hourly AORC data to daily min, mean, and max for each metric.

    :param aorc_dataset: Prepared AORC dataset, chunked at 24 hours
    :param metric_names: Names of metrics from `METRICS`
    :return: Lazy Dataset with one `{metric}_{statistic}` variable per pair.
        Metrics that were not requested are absent, not NaN-filled.
    """
    _validate_metric_names(metric_names)
    derived = _derived_inputs(aorc_dataset, metric_names)

    daily = {}
    for name in metric_names:
        hourly = METRICS[name].compute(derived)
        grouped = hourly.resample(time="1D")
        for statistic in DAILY_STATISTICS:
            reduced = getattr(grouped, statistic)().astype(np.float32)
            reduced.attrs = _variable_attributes(name, statistic)
            daily[f"{name}_{statistic}"] = reduced

    return xr.Dataset(daily, attrs=GLOBAL_ATTRIBUTES)
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
docker run --rm -v "$PWD":/project -w /project aorc python -m pytest tests/test_pipeline.py -v
```

Expected: all tests pass.

If `test_prepare_dataset_crops_masks_and_chunks` reports chunk sizes other than `(24, 24)`, the `.chunk` call is being overridden by the fixture's existing chunking — confirm `prepare_dataset` applies `.chunk` after `.where`, as written.

- [ ] **Step 6: Commit**

```bash
git add src/aorc_heat/pipeline.py tests/test_pipeline.py tests/conftest.py
git commit -m "feat: add metric registry and daily reduction to pipeline"
```

---

### Task 7: pipeline.py — output store and orchestration

Adds the single incrementally extensible zarr store and the year loop.

The output store is chunked at **one day per time chunk**. This is what makes region writes safe without `safe_chunks=False`: input chunked at 24 hours produces daily output chunked at exactly 1, so every yearly region write lands on chunk boundaries and no two writes ever touch the same chunk.

**Files:**
- Modify: `src/aorc_heat/pipeline.py` (append)
- Modify: `tests/test_pipeline.py` (append)

**Interfaces:**
- Consumes: `daily_metrics`, `required_variables`, `prepare_dataset`, `METRICS` from Task 6; `mask.texas_region` from Task 5.
- Produces:
  - `OUTPUT_STORE_NAME = "aorc_heat_metrics.zarr"`
  - `daily_time_axis(start_year, end_year) -> xr.CFTimeIndex`
  - `time_region(time_axis, block_index) -> slice`
  - `pending_metrics(store_path, metric_names, time_axis) -> list[str]`
  - `initialize_output_store(store_path, metric_names, time_axis, template) -> None`
  - `write_block(store_path, daily, time_axis) -> None`
  - `open_aorc_year(year, region, variable_names, filesystem=None) -> xr.Dataset`
  - `run(output_dir, metric_names, start_year, end_year, filesystem=None) -> Path`

**Why `write_block` locates its own region.** An earlier draft took a `year` argument and wrote `region=year_region(axis, year)`, which silently assumes the result covers every day of that year. It does not hold for a partial year, and it makes the function untestable against anything smaller than a full year of synthetic data. Deriving the region from the block's own timestamps is both more general and self-checking.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pipeline.py`:

```python
# ---------------------------------------------------------------------------
# Output store
# ---------------------------------------------------------------------------
def test_daily_time_axis_spans_whole_years_and_handles_leap_days():
    axis = pipeline.daily_time_axis(1999, 2000)
    assert axis.size == 365 + 366
    assert axis[0].year == 1999 and axis[0].month == 1 and axis[0].day == 1
    assert axis[-1].year == 2000 and axis[-1].month == 12 and axis[-1].day == 31


def test_time_region_locates_a_block_within_the_axis():
    axis = pipeline.daily_time_axis(1999, 2000)
    assert pipeline.time_region(axis, axis[:365]) == slice(0, 365)
    assert pipeline.time_region(axis, axis[365:]) == slice(365, 731)
    assert pipeline.time_region(axis, axis[10:12]) == slice(10, 12)


def test_time_region_rejects_a_block_outside_the_axis():
    axis = pipeline.daily_time_axis(2000, 2000)
    outside = pipeline.daily_time_axis(1990, 1990)[:5]
    with pytest.raises(KeyError):
        pipeline.time_region(axis, outside)


def _template_for(dataset, metric_names):
    return pipeline.daily_metrics(dataset, metric_names)


def test_initialize_output_store_creates_requested_variables(tmp_path, synthetic_aorc_dataset):
    store_path = tmp_path / pipeline.OUTPUT_STORE_NAME
    axis = pipeline.daily_time_axis(2000, 2000)
    template = _template_for(synthetic_aorc_dataset, ["humidex"])

    pipeline.initialize_output_store(store_path, ["humidex"], axis, template)

    written = xr.open_zarr(store_path, consolidated=True)
    assert sorted(written.data_vars) == ["humidex_max", "humidex_mean", "humidex_min"]
    assert written.sizes["time"] == 366
    assert written.chunksizes["time"][0] == 1


def test_initialize_output_store_appends_new_metrics(tmp_path, synthetic_aorc_dataset):
    store_path = tmp_path / pipeline.OUTPUT_STORE_NAME
    axis = pipeline.daily_time_axis(2000, 2000)

    pipeline.initialize_output_store(
        store_path, ["humidex"], axis, _template_for(synthetic_aorc_dataset, ["humidex"])
    )
    pipeline.initialize_output_store(
        store_path,
        ["simplified_wbgt"],
        axis,
        _template_for(synthetic_aorc_dataset, ["simplified_wbgt"]),
    )

    written = xr.open_zarr(store_path, consolidated=True)
    assert "humidex_mean" in written.data_vars
    assert "simplified_wbgt_mean" in written.data_vars


def test_pending_metrics_skips_what_is_already_present(tmp_path, synthetic_aorc_dataset):
    store_path = tmp_path / pipeline.OUTPUT_STORE_NAME
    axis = pipeline.daily_time_axis(2000, 2000)

    assert pipeline.pending_metrics(store_path, ["humidex", "heat_index"], axis) == [
        "humidex",
        "heat_index",
    ]

    pipeline.initialize_output_store(
        store_path, ["humidex"], axis, _template_for(synthetic_aorc_dataset, ["humidex"])
    )
    assert pipeline.pending_metrics(store_path, ["humidex", "heat_index"], axis) == ["heat_index"]


def test_pending_metrics_rejects_time_axis_mismatch(tmp_path, synthetic_aorc_dataset):
    store_path = tmp_path / pipeline.OUTPUT_STORE_NAME
    pipeline.initialize_output_store(
        store_path,
        ["humidex"],
        pipeline.daily_time_axis(2000, 2000),
        _template_for(synthetic_aorc_dataset, ["humidex"]),
    )

    with pytest.raises(ValueError, match="does not match"):
        pipeline.pending_metrics(store_path, ["heat_index"], pipeline.daily_time_axis(1999, 2001))


def test_unwritten_days_read_back_as_nan(tmp_path, synthetic_aorc_dataset):
    store_path = tmp_path / pipeline.OUTPUT_STORE_NAME
    axis = pipeline.daily_time_axis(2000, 2000)
    template = _template_for(synthetic_aorc_dataset, ["humidex"])

    pipeline.initialize_output_store(store_path, ["humidex"], axis, template)

    written = xr.open_zarr(store_path, consolidated=True)
    # Nothing has been written yet, so every day must read as NaN rather than 0.
    assert bool(np.isnan(written["humidex_mean"].isel(time=0)).all())


def test_write_block_fills_only_its_own_days(tmp_path, synthetic_aorc_dataset):
    store_path = tmp_path / pipeline.OUTPUT_STORE_NAME
    axis = pipeline.daily_time_axis(2000, 2000)
    daily = _template_for(synthetic_aorc_dataset, ["humidex"])

    pipeline.initialize_output_store(store_path, ["humidex"], axis, daily)
    pipeline.write_block(store_path, daily, axis)

    written = xr.open_zarr(store_path, consolidated=True).compute()
    # The synthetic dataset covers 1-2 July 2000: days 182 and 183 of a leap year.
    assert not bool(np.isnan(written["humidex_mean"].isel(time=slice(182, 184))).any())
    assert bool(np.isnan(written["humidex_mean"].isel(time=0)).all())
    assert bool(np.isnan(written["humidex_mean"].isel(time=365)).all())


def test_adding_a_metric_leaves_earlier_values_untouched(tmp_path, synthetic_aorc_dataset):
    store_path = tmp_path / pipeline.OUTPUT_STORE_NAME
    axis = pipeline.daily_time_axis(2000, 2000)

    first = _template_for(synthetic_aorc_dataset, ["humidex"])
    pipeline.initialize_output_store(store_path, ["humidex"], axis, first)
    pipeline.write_block(store_path, first, axis)
    before = xr.open_zarr(store_path, consolidated=True)["humidex_mean"].compute()

    second = _template_for(synthetic_aorc_dataset, ["simplified_wbgt"])
    pipeline.initialize_output_store(store_path, ["simplified_wbgt"], axis, second)
    pipeline.write_block(store_path, second, axis)

    after = xr.open_zarr(store_path, consolidated=True)
    np.testing.assert_array_equal(before.values, after["humidex_mean"].compute().values)
    assert "simplified_wbgt_mean" in after.data_vars
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
docker run --rm -v "$PWD":/project -w /project aorc python -m pytest tests/test_pipeline.py -v -k "store or time_axis or time_region or pending or write_block or nan or untouched"
```

Expected: FAIL with `AttributeError: module 'aorc_heat.pipeline' has no attribute 'daily_time_axis'`.

- [ ] **Step 3: Write the implementation**

Add these imports to the top of `src/aorc_heat/pipeline.py`, alongside the existing ones:

```python
from datetime import datetime
from pathlib import Path

import dask.array
```

Then append to `src/aorc_heat/pipeline.py`:

```python
OUTPUT_STORE_NAME = "aorc_heat_metrics.zarr"

#: Output time chunk, in days. One day per chunk is deliberate: the input is
#: chunked at 24 hours, so each daily result is already one chunk along time and
#: every yearly region write lands exactly on chunk boundaries. Any larger value
#: would leave partial chunks at year boundaries, which two different region
#: writes could then race on.
OUTPUT_TIME_CHUNK_DAYS = 1

OUTPUT_DIMENSIONS = ("time", "latitude", "longitude")


def _log(message):
    """Timestamped progress line, matching the previous pipeline's output style."""
    print(f"{datetime.now().timestamp()} {message}", flush=True)


def daily_time_axis(start_year, end_year):
    """The daily time coordinate the output store spans.

    A single explicit calendar is used for the whole range so that each year's
    result can be converted to match before it is written.

    :param start_year: First calendar year, inclusive
    :param end_year: Last calendar year, inclusive
    :return: CFTimeIndex with one entry per day
    """
    if end_year < start_year:
        raise ValueError(f"end_year {end_year} precedes start_year {start_year}")
    return xr.date_range(
        f"{start_year}-01-01",
        f"{end_year}-12-31",
        freq="D",
        use_cftime=True,
        calendar="standard",
    )


def time_region(time_axis, block_index):
    """Locate a contiguous block of days within the store's time axis.

    Deriving the region from the block's own timestamps rather than from a year
    number means a partial year writes correctly, and a block that does not line
    up with the store is caught here instead of corrupting a region.

    :param time_axis: Full daily time axis from `daily_time_axis`
    :param block_index: Time index of the block about to be written
    :return: Slice of positional indices into `time_axis`
    """
    start = int(time_axis.get_loc(block_index[0]))
    stop = start + block_index.size
    if stop > time_axis.size:
        raise ValueError(
            f"Block of {block_index.size} days starting at index {start} runs past "
            f"the end of the store's {time_axis.size}-day axis."
        )
    if not np.array_equal(np.asarray(time_axis[start:stop]), np.asarray(block_index)):
        raise ValueError(
            "Block timestamps are not contiguous within the store's time axis."
        )
    return slice(start, stop)


def pending_metrics(store_path, metric_names, time_axis):
    """Metrics that still need computing, after validating the existing store.

    A metric counts as present when its `_mean` variable exists. This does not
    verify that every year was written; see the known limitation in the design
    spec.

    :param store_path: Path to the output zarr store
    :param metric_names: Requested metric names
    :param time_axis: Expected daily time axis
    :return: The subset of `metric_names` not already in the store
    """
    _validate_metric_names(metric_names)
    store_path = Path(store_path)
    if not store_path.exists():
        return list(metric_names)

    existing = xr.open_zarr(store_path, consolidated=True)
    if existing.sizes.get("time") != time_axis.size:
        raise ValueError(
            f"Existing store at {store_path} spans {existing.sizes.get('time')} days, "
            f"which does not match the {time_axis.size} days requested. "
            "Delete the store or request the range it already covers."
        )
    return [name for name in metric_names if f"{name}_mean" not in existing.data_vars]


def initialize_output_store(store_path, metric_names, time_axis, template):
    """Allocate NaN-filled variables for `metric_names` without computing anything.

    Creates the store when absent, otherwise appends the new variables to it.
    The template supplies the spatial coordinates and chunk sizes.

    :param store_path: Path to the output zarr store
    :param metric_names: Metrics to allocate
    :param time_axis: Full daily time axis
    :param template: One year of `daily_metrics` output, used for its grid
    """
    store_path = Path(store_path)
    spatial_chunks = tuple(
        template.chunksizes[dimension][0] for dimension in ("latitude", "longitude")
    )
    shape = (time_axis.size, template.sizes["latitude"], template.sizes["longitude"])
    chunks = (OUTPUT_TIME_CHUNK_DAYS,) + spatial_chunks

    if store_path.exists():
        existing = xr.open_zarr(store_path, consolidated=True)
        for dimension in ("latitude", "longitude"):
            if not np.array_equal(existing[dimension].values, template[dimension].values):
                raise ValueError(
                    f"Existing store at {store_path} has a different {dimension} axis "
                    "than the requested region. Delete the store to rebuild it."
                )

    placeholder = dask.array.full(shape, np.nan, dtype=np.float32, chunks=chunks)
    variables = {
        f"{name}_{statistic}": (
            OUTPUT_DIMENSIONS,
            placeholder,
            _variable_attributes(name, statistic),
        )
        for name in metric_names
        for statistic in DAILY_STATISTICS
    }
    allocation = xr.Dataset(
        variables,
        coords={
            "time": time_axis,
            "latitude": template["latitude"],
            "longitude": template["longitude"],
        },
        attrs=GLOBAL_ATTRIBUTES,
    )
    # Without an explicit fill value, zarr defaults to 0.0 for float arrays and
    # days that were never written would read back as a plausible-looking zero
    # instead of NaN.
    encoding = {name: {"_FillValue": np.float32("nan")} for name in variables}

    allocation.to_zarr(
        store_path,
        mode="a" if store_path.exists() else "w",
        compute=False,
        zarr_format=2,
        consolidated=True,
        encoding=encoding,
    )


def write_block(store_path, daily, time_axis):
    """Write a contiguous block of daily metrics into its region of the store.

    The block is converted to the store's calendar and rechunked to one day per
    chunk so the write aligns exactly with the store's chunking. Coordinate
    variables are dropped because they already exist in the store.

    :param store_path: Path to the output zarr store
    :param daily: A contiguous block of `daily_metrics` output
    :param time_axis: Full daily time axis
    """
    aligned = daily.convert_calendar("standard", use_cftime=True)
    region = time_region(time_axis, aligned.indexes["time"])
    aligned = aligned.chunk({"time": OUTPUT_TIME_CHUNK_DAYS})
    aligned = aligned.drop_vars(list(aligned.coords))
    aligned.to_zarr(Path(store_path), region={"time": region})


def open_aorc_year(year, region, variable_names, filesystem=None):
    """Open one AORC year, restricted to the requested variables and region.

    :param year: Calendar year to open
    :param region: A `mask.Region` describing the area of interest
    :param variable_names: Raw AORC variable names to read
    :param filesystem: Optional preconfigured s3fs filesystem
    :return: Cropped, masked, rechunked dataset
    """
    if filesystem is None:
        import s3fs

        filesystem = s3fs.S3FileSystem(anon=True)

    dataset = xr.open_zarr(
        filesystem.get_mapper(f"{AORC_S3_BASE_PATH}{year}.zarr"), consolidated=True
    )
    return prepare_dataset(dataset[list(variable_names)], region)


def run(output_dir, metric_names, start_year, end_year, filesystem=None):
    """Compute the requested metrics across the year range into one zarr store.

    Metrics already present in the store are skipped rather than recomputed, so
    a store can be extended one metric at a time across several invocations.

    :param output_dir: Directory the zarr store lives in
    :param metric_names: Metrics from `METRICS` to compute
    :param start_year: First calendar year, inclusive
    :param end_year: Last calendar year, inclusive
    :param filesystem: Optional preconfigured s3fs filesystem
    :return: Path to the output store
    """
    from aorc_heat import mask as mask_module

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    store_path = output_dir / OUTPUT_STORE_NAME
    time_axis = daily_time_axis(start_year, end_year)

    pending = pending_metrics(store_path, metric_names, time_axis)
    if not pending:
        _log("[init] Every requested metric is already in the store; nothing to compute.")
        return store_path
    _log(f"[init] Computing {', '.join(pending)} for {start_year}-{end_year}.")

    if filesystem is None:
        import s3fs

        filesystem = s3fs.S3FileSystem(anon=True)

    sample = xr.open_zarr(
        filesystem.get_mapper(f"{AORC_S3_BASE_PATH}{start_year}.zarr"), consolidated=True
    )
    _log("[init] Loading the Texas mask.")
    region = mask_module.texas_region(sample.latitude.values, sample.longitude.values)

    variables = required_variables(pending)
    first_year = open_aorc_year(start_year, region, variables, filesystem)
    initialize_output_store(store_path, pending, time_axis, daily_metrics(first_year, pending))
    _log(f"[init] Allocated {len(pending) * len(DAILY_STATISTICS)} variables in {store_path}.")

    for year in range(start_year, end_year + 1):
        _log(f"[compute] Deriving metrics for {year}.")
        dataset = open_aorc_year(year, region, variables, filesystem)
        write_block(store_path, daily_metrics(dataset, pending), time_axis)
        _log(f"[compute] Wrote {year}.")

    _log("[final] Finished.")
    return store_path
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
docker run --rm -v "$PWD":/project -w /project aorc python -m pytest tests/test_pipeline.py -v
```

Expected: all tests pass.

If `write_block` raises about unaligned chunks, confirm the store was created with `OUTPUT_TIME_CHUNK_DAYS = 1` and that `aligned.chunk({"time": 1})` is applied before `to_zarr`. If `test_unwritten_days_read_back_as_nan` sees `0.0` instead of NaN, the `_FillValue` encoding did not reach the zarr array.

- [ ] **Step 5: Commit**

```bash
git add src/aorc_heat/pipeline.py tests/test_pipeline.py
git commit -m "feat: add incremental zarr output store and year loop"
```

---

### Task 8: cli.py — command-line front end

**Files:**
- Create: `src/aorc_heat/cli.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: `pipeline.METRICS` and `pipeline.run` from Tasks 6–7.
- Produces:
  - `build_parser() -> argparse.ArgumentParser`
  - `selected_metrics(arguments) -> list[str]`
  - `main(argv=None) -> int`
  - Constants `DEFAULT_START_YEAR = 1979`, `DEFAULT_END_YEAR = 2024`, `DASHBOARD_ADDRESS = ":8002"`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli.py`:

```python
"""Unit tests for argument parsing.

No Dask cluster is started here; `main` is not exercised because it opens a
network connection to S3.
"""
import pytest

from aorc_heat import cli, pipeline


BASE_ARGUMENTS = ["--output-dir", "/tmp/out", "--cores", "8", "--memory-limit", "10GB"]


def test_all_selects_every_registered_metric():
    arguments = cli.build_parser().parse_args(BASE_ARGUMENTS + ["--all"])
    assert cli.selected_metrics(arguments) == sorted(pipeline.METRICS)


def test_metrics_selects_only_what_was_asked_for():
    arguments = cli.build_parser().parse_args(
        BASE_ARGUMENTS + ["--metrics", "humidex", "heat_index"]
    )
    assert cli.selected_metrics(arguments) == ["humidex", "heat_index"]


def test_metric_selection_is_required():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(BASE_ARGUMENTS)


def test_all_and_metrics_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(BASE_ARGUMENTS + ["--all", "--metrics", "humidex"])


def test_unknown_metric_is_rejected():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(BASE_ARGUMENTS + ["--metrics", "not_a_metric"])


def test_year_range_defaults_to_the_full_aorc_record():
    arguments = cli.build_parser().parse_args(BASE_ARGUMENTS + ["--all"])
    assert arguments.start_year == cli.DEFAULT_START_YEAR
    assert arguments.end_year == cli.DEFAULT_END_YEAR


def test_year_range_can_be_narrowed():
    arguments = cli.build_parser().parse_args(
        BASE_ARGUMENTS + ["--all", "--start-year", "2010", "--end-year", "2011"]
    )
    assert (arguments.start_year, arguments.end_year) == (2010, 2011)


def test_inverted_year_range_is_rejected():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            BASE_ARGUMENTS + ["--all", "--start-year", "2011", "--end-year", "2010"]
        )


@pytest.mark.parametrize(
    "incomplete_arguments",
    [
        ["--cores", "8", "--memory-limit", "10GB"],               # no --output-dir
        ["--output-dir", "/tmp/out", "--memory-limit", "10GB"],   # no --cores
        ["--output-dir", "/tmp/out", "--cores", "8"],             # no --memory-limit
    ],
)
def test_required_arguments(incomplete_arguments):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(incomplete_arguments + ["--all"])
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
docker run --rm -v "$PWD":/project -w /project aorc python -m pytest tests/test_cli.py -v
```

Expected: collection error, `ImportError: cannot import name 'cli' from 'aorc_heat'`.

- [ ] **Step 3: Write the implementation**

Create `src/aorc_heat/cli.py`:

```python
"""Command-line front end for the AORC heat pipeline.

Owns argument parsing and the Dask cluster lifecycle, and nothing else. All
science lives in `core` and all dataset handling in `pipeline`.
"""

import argparse
from pathlib import Path

from aorc_heat import pipeline

DEFAULT_START_YEAR = 1979
DEFAULT_END_YEAR = 2024
DASHBOARD_ADDRESS = ":8002"


class _YearRangeParser(argparse.ArgumentParser):
    """Parser that rejects an inverted year range at parse time.

    The check lives here rather than in `main` so that it applies to every
    caller, including the tests, and so a bad range fails before any cluster is
    started.
    """

    def parse_args(self, args=None, namespace=None):
        arguments = super().parse_args(args, namespace)
        if arguments.end_year < arguments.start_year:
            self.error(
                f"--end-year {arguments.end_year} precedes "
                f"--start-year {arguments.start_year}"
            )
        return arguments


def build_parser():
    """Build the argument parser.

    :return: A configured ArgumentParser
    """
    parser = _YearRangeParser(
        prog="aorc-heat",
        description=(
            "Compute daily minimum, mean, and maximum heat metrics from the NOAA "
            "AORC hourly archive and write them to a zarr store. Metrics already "
            "present in the store are skipped, so a store can be extended one "
            "metric at a time."
        ),
    )

    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--all",
        action="store_true",
        help="Compute every available metric.",
    )
    selection.add_argument(
        "--metrics",
        nargs="+",
        choices=sorted(pipeline.METRICS),
        metavar="METRIC",
        help=f"Metrics to compute. One or more of: {', '.join(sorted(pipeline.METRICS))}.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory the output zarr store is written to.",
    )
    parser.add_argument(
        "--cores",
        required=True,
        type=int,
        help="Number of Dask worker processes, one thread each.",
    )
    parser.add_argument(
        "--memory-limit",
        required=True,
        help="Memory limit per Dask worker, for example '10GB'.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=DEFAULT_START_YEAR,
        help=f"First year to compute, inclusive. Default {DEFAULT_START_YEAR}.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=DEFAULT_END_YEAR,
        help=f"Last year to compute, inclusive. Default {DEFAULT_END_YEAR}.",
    )

    return parser


def selected_metrics(arguments):
    """Resolve `--all` or `--metrics` into a concrete metric list.

    :param arguments: Parsed arguments
    :return: Metric names to compute
    """
    if arguments.all:
        return sorted(pipeline.METRICS)
    return list(arguments.metrics)


def main(argv=None):
    """Parse arguments, start a Dask cluster, and run the pipeline.

    :param argv: Argument list, defaulting to sys.argv[1:]
    :return: Process exit code
    """
    import xarray as xr
    from dask.distributed import LocalCluster

    arguments = build_parser().parse_args(argv)
    metric_names = selected_metrics(arguments)
    xr.set_options(use_flox=True)

    cluster = LocalCluster(
        n_workers=arguments.cores,
        threads_per_worker=1,
        memory_limit=arguments.memory_limit,
        dashboard_address=DASHBOARD_ADDRESS,
    )
    try:
        print(cluster.get_client())
        store_path = pipeline.run(
            output_dir=arguments.output_dir,
            metric_names=metric_names,
            start_year=arguments.start_year,
            end_year=arguments.end_year,
        )
        print(f"Metrics written to {store_path}")
    finally:
        cluster.close()
    return 0
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
docker run --rm -v "$PWD":/project -w /project aorc python -m pytest tests/ -v
```

Expected: every test in the suite passes.

- [ ] **Step 5: Verify the console script is wired up**

```bash
docker run --rm -v "$PWD":/project -w /project aorc aorc-heat --help
```

Expected: the usage message listing `--all`, `--metrics`, `--output-dir`, `--cores`, `--memory-limit`, `--start-year`, `--end-year`.

- [ ] **Step 6: Commit**

```bash
git add src/aorc_heat/cli.py tests/test_cli.py
git commit -m "feat: add command-line front end"
```

---

### Task 9: Documentation

**Files:**
- Modify: `README.md`
- Modify: `tests/README.md`

**Interfaces:**
- Consumes: the finished CLI from Task 8.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Rewrite README.md**

Replace the entire contents of `README.md`:

````markdown
# AORC-Heat-Pipeline

Daily minimum, mean, and maximum heat metrics derived from the
[NOAA AORC](https://registry.opendata.aws/noaa-nws-aorc) 1 km hourly archive,
masked to the state of Texas.

## Metrics

| Name | Description | Units |
| --- | --- | --- |
| `heat_index` | NWS Rothfusz regression | °F |
| `apparent_temperature` | Steadman apparent temperature | °F |
| `humidex` | Canadian humidex | °F |
| `simplified_wbgt` | ACSM simplified wet-bulb globe temperature | °F |
| `wet_bulb_temperature` | Thermodynamic wet-bulb temperature | °F |

## Structure

- `src/aorc_heat/core.py` — scalar scientific formulations in literature units,
  compiled with numba and computed in float32. Knows nothing about AORC.
- `src/aorc_heat/pipeline.py` — reads AORC from S3, builds shared intermediate
  quantities once, applies the core functions, reduces each 24-hour chunk to
  daily statistics, and writes the result into a single zarr store.
- `src/aorc_heat/cli.py` — argument parsing and Dask cluster lifecycle.
- `src/aorc_heat/mask.py` — generates and caches the Texas boundary mask.

## Running

Build the image once:

```bash
docker build -t aorc .
```

Compute every metric across the full record:

```bash
docker run --rm -v "$PWD":/project -w /project aorc \
    aorc-heat --all --output-dir output_zarrs --cores 80 --memory-limit 10GB
```

Compute a subset, or a shorter period:

```bash
docker run --rm -v "$PWD":/project -w /project aorc \
    aorc-heat --metrics heat_index humidex \
              --output-dir output_zarrs --cores 80 --memory-limit 10GB \
              --start-year 2020 --end-year 2021
```

Metrics already present in the store are skipped rather than recomputed, so a
later run that adds `--metrics wet_bulb_temperature` appends those variables
without touching what is already there. The year range must match the range the
store was built for.

## Output

A single store at `<output-dir>/aorc_heat_metrics.zarr` with dimensions
`(time, latitude, longitude)`, one entry per day, and a
`{metric}_{min,mean,max}` variable for each computed metric. Cells outside the
Texas boundary are NaN.

## Tests

```bash
docker run --rm -v "$PWD":/project -w /project aorc python -m pytest tests/ -v
```
````

- [ ] **Step 2: Rewrite tests/README.md**

Replace the entire contents of `tests/README.md`:

```markdown
# Tests

Unit tests for the heat-stress formulations in
[`src/aorc_heat/core.py`](../src/aorc_heat/core.py), the region masking in
[`src/aorc_heat/mask.py`](../src/aorc_heat/mask.py), the dataset orchestration in
[`src/aorc_heat/pipeline.py`](../src/aorc_heat/pipeline.py), and argument parsing
in [`src/aorc_heat/cli.py`](../src/aorc_heat/cli.py).

Expected values in `test_core.py` are taken from published references — the NWS
heat-index chart and Rothfusz regression, standard Tetens saturation-vapour-
pressure tables, the Steadman apparent-temperature, Canadian humidex and ACSM
simplified-WBGT definitions, and Stull's 2011 wet-bulb reference — so that a
formula transcription error is caught rather than reproduced. Tolerances are
loose enough to absorb float32 rounding, which the kernels use by design.

`test_pipeline.py` runs against an in-memory synthetic dataset and a `tmp_path`
zarr store. Nothing in the suite touches S3 or the network.

## Running

```bash
docker build -t aorc .
docker run --rm -v "$PWD":/project -w /project aorc python -m pytest tests/ -v
```
```

- [ ] **Step 3: Run the full suite one final time**

```bash
docker run --rm -v "$PWD":/project -w /project aorc python -m pytest tests/ -v
```

Expected: every test passes.

- [ ] **Step 4: Commit**

```bash
git add README.md tests/README.md
git commit -m "docs: describe the refactored package and CLI"
```

---

## Post-implementation verification

The test suite never touches S3, so the first real run is the only thing that
exercises the S3 read path, the Texas mask generation, and the region-write
alignment together. Run a single year against a scratch directory before
launching the full record:

```bash
docker run --rm -v "$PWD":/project -w /project aorc \
    aorc-heat --metrics humidex --output-dir /project/scratch \
              --cores 8 --memory-limit 4GB \
              --start-year 2020 --end-year 2020
```

Then confirm the store looks right:

```bash
docker run --rm -v "$PWD":/project -w /project aorc python -c "
import xarray as xr
ds = xr.open_zarr('scratch/aorc_heat_metrics.zarr', consolidated=True)
print(ds)
print('NaN fraction:', float(ds.humidex_mean.isnull().mean().compute()))
"
```

Two things to watch, both consequences of decisions recorded in the spec:

1. **Spatial chunking is left native**, where the previous pipeline forced
   `latitude: -1, longitude: -1`. If rechunking to 24-hour time blocks turns out
   to read far more than expected, this is the knob to revisit in
   `prepare_dataset`.
2. **Values will differ from previous runs** because `core.vapor_pressure` now
   takes `PRES_surface` instead of assuming 1013.25 hPa. This affects heat
   index, apparent temperature, humidex, and simplified WBGT, most visibly at
   elevation in west Texas. This is intended.
