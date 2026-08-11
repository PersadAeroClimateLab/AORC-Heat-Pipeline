---
name: adding-a-core-function
description: Use when adding or changing a scalar science kernel in aorc_heat/core.py — a thermodynamic formula, unit conversion, or numba vectorize function for the AORC heat pipeline.
---

# Adding a Core Function

## Overview

`core.py` holds scalar formulations in **literature units**. It imports only numpy
and numba, knows nothing about AORC, and every function is independently testable
against published values. Unit translation lives in `pipeline.py`.

## The contract

Every function in `core.py`:

1. Takes and returns **scalars in literature units** (°C, hPa, kg/kg, %, m/s)
2. Computes in **float32** — coerce each argument with `np.float32(...)` first
3. Is decorated `@nb.vectorize(target="cpu", cache=True, fastmath=...)` with **no signature list**
4. Imports nothing from the project
5. Is **self-contained** — never accepts a value derivable from its other arguments

Rule 5 has teeth. `saturation_vapor_pressure_slope` recomputes the saturation
pressure internally rather than accepting `(temperature, esat)`, because a caller
could pass an inconsistent pair and get a plausible wrong answer. Paying an extra
`exp()` to make that impossible is the right trade.

## Choosing the fastmath flag

**This is the one decision that has caused a production-corrupting bug in this
codebase. Get it right before writing the body.**

```
Does your function branch on a value that could be NaN?
(any if/elif on an argument, or on something computed from one)

  YES → fastmath=NAN_SAFE_FASTMATH
  NO  → fastmath=True
```

`fastmath=True` includes LLVM's `nnan`, which asserts no operand is ever NaN.
Under that assumption a `x != x` guard is dead code and gets folded away, and an
`if/elif` chain may be treated as exhaustive.

The pipeline masks to a region with `.where`, so **roughly half of all cells
reaching these kernels are NaN by design**. `wet_bulb_temperature` shipped with
`fastmath=True` and its clamp collapsed into an unconditional `else`, turning
every out-of-region cell into a finite `T − 40` instead of NaN. Correct-looking
cold temperatures across the whole masked area.

## Naming

Function names say **what is computed**; parameter names carry **the unit**;
return units live in the docstring.

```python
# Correct
def relative_humidity(vapor_pressure_hpa, saturation_vapor_pressure_hpa): ...

# Wrong — the function name collides with four other functions' parameter names
def relative_humidity_percent(vapor_pressure_hpa, saturation_vapor_pressure_hpa): ...
```

Reuse the established parameter names exactly: `air_temperature_celsius`,
`air_temperature_fahrenheit`, `air_temperature_kelvin`, `vapor_pressure_hpa`,
`saturation_vapor_pressure_hpa`, `specific_humidity`, `air_pressure_hpa`,
`wind_speed_ms`, `relative_humidity_percent`.

## Example

```python
@nb.vectorize(target="cpu", cache=True, fastmath=NAN_SAFE_FASTMATH)
def dew_point_celsius(air_temperature_celsius: float, relative_humidity_percent: float) -> float:
    """Dew point by the Magnus approximation.

    Magnus (1844) with the Alduchov-Eskridge (1996) coefficients, valid
    -40..50 C to within 0.1 C.

    :param air_temperature_celsius: Temperature in degrees Celsius
    :param relative_humidity_percent: Relative humidity from 0 (dry) to 100 (saturated)
    :return: Dew point in degrees Celsius
    """
    temperature = np.float32(air_temperature_celsius)
    humidity = np.float32(relative_humidity_percent)

    # Branches below compare against a possibly-NaN value; see NAN_SAFE_FASTMATH.
    if temperature != temperature or humidity != humidity:
        return np.float32(np.nan)
    if humidity <= np.float32(0.0):
        return np.float32(np.nan)

    a = np.float32(17.625)
    b = np.float32(243.04)
    gamma = np.log(humidity / np.float32(100.0)) + a * temperature / (b + temperature)
    return b * gamma / (a - gamma)
```

## Checklist

- [ ] Literature units in, literature units out; units named in the docstring
- [ ] `np.float32(...)` coercion as the first statements
- [ ] fastmath flag chosen by the branch rule above
- [ ] Self-contained — no argument derivable from another
- [ ] Docstring cites the source (paper, chart, or standard) and states valid range
- [ ] Parameter names match the established vocabulary
- [ ] Tests written — **REQUIRED SUB-SKILL:** use `writing-core-unit-tests`

## Common mistakes

| Mistake | Why it bites |
|---|---|
| Copying `fastmath=True` from the neighbouring function | Silently breaks NaN handling if you branch. Half the grid is NaN. |
| Accepting a precomputed intermediate "for speed" | A caller can pass an inconsistent pair. Recompute it. |
| Putting the return unit in the function name | Collides with the parameter names other functions use for that quantity. |
| Assuming a helper's signature from memory | `vapor_pressure` takes **two** arguments — ambient pressure, not an assumed 1013.25 hPa. |
| Tightening test tolerances because float32 "should" be exact | float32 carries ~7 digits. Use `rel=1e-5` / `abs=1e-4`. |

## Verifying

Docker only — there is no local environment with the dependencies:

```bash
docker run --rm -v "$PWD":/project -w /project aorc python -m pytest tests/test_core.py -v
```

Numba caches compiled kernels to `.nbi`/`.nbc` keyed partly by line number, and
those files are often root-owned from earlier container runs. If behaviour
contradicts the source you are reading, clear the cache **as root** and re-run:

```bash
docker run --rm -v "$PWD":/project -w /project aorc \
    bash -c "find /project -name '*.nbi' -delete; find /project -name '*.nbc' -delete"
```
