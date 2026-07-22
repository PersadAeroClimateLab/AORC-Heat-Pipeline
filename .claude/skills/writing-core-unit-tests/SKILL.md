---
name: writing-core-unit-tests
description: Use when writing or reviewing unit tests for scalar kernels in aorc_heat/core.py, or when a core function has no test, tolerances that look too tight, or no NaN coverage.
---

# Writing Core Unit Tests

## Overview

`tests/test_core.py` exists to catch a **transcription error in a formula**. That
only works if the expected value comes from somewhere other than the code being
tested.

**A test whose expected value is computed by the implementation cannot fail.**

## The tautology trap

This is the single most common failure here, and it has shipped in this
repository more than once.

```python
# BROKEN — cannot fail. A typo in 61.0, 1.2, or 0.094 is copied into both sides.
def test_heat_index_below_threshold():
    temperature, humidity = 75.0, 40.0
    expected = 0.5 * (temperature + 61.0 + ((temperature - 68.0) * 1.2) + (humidity * 0.094))
    assert core.heat_index(temperature, humidity) == pytest.approx(expected, rel=1e-5)

# CORRECT — the expected value is an independent constant.
def test_heat_index_below_threshold():
    # Below 80 F the NWS returns the simple Steadman form. Hand-evaluated at
    # 75 F / 40% RH: 0.5 * (75 + 61 + 7*1.2 + 40*0.094) = 74.08.
    assert core.heat_index(75.0, 40.0) == pytest.approx(74.08, abs=0.01)
```

Legitimate sources for an expected value: a published table or chart, a
hand-evaluated formula from the paper, a definitional fixed point (0 °C = 32 °F),
a physical invariant (`Tw == T` at saturation), or an **independent**
implementation such as a float64 `scipy.optimize.brentq` solve.

Not legitimate: calling the function under test, or re-typing its formula.

## Required coverage

Every core function needs all four:

| Check | Why |
|---|---|
| **Literature value** | Catches a transcription error in a coefficient |
| **Physical invariant** | Monotonicity, bounds, or an identity that must hold |
| **NaN propagation** | Half the production grid is NaN; this is how a Critical bug shipped |
| **Finite in → finite out** | Guards the pipeline's warning suppression |

The NaN check is not optional. `wet_bulb_temperature` had none, and a
`fastmath` bug that turned every masked cell into a finite `T − 40` went
undetected through implementation and review.

```python
@pytest.mark.parametrize(
    "temperature, humidity, pressure",
    [(np.nan, 0.01, 1013.25), (25.0, np.nan, 1013.25),
     (25.0, 0.01, np.nan), (np.nan, np.nan, np.nan)],
)
def test_wet_bulb_temperature_propagates_nan(temperature, humidity, pressure):
    assert np.isnan(core.wet_bulb_temperature(temperature, humidity, pressure))
```

## Tolerances

float32 carries ~7 decimal digits. Use `rel=1e-5` or `abs=1e-4` against
hand-computed float64 expectations. **Do not tighten them** — a passing tight
tolerance means you are comparing float32 against itself somewhere.

Wider tolerances are correct where the reference is itself approximate: `abs=2.0`
for NWS chart values (chart rounding plus the regression's stated ±1.3 °F),
`abs=0.7` for Stull's wet-bulb reference.

## Prove the test can fail

Before you commit, break the implementation and watch the test go red. A test
never observed failing is a test you have not verified.

```bash
# Temporarily change a coefficient, or swap two arguments, then:
docker run --rm -v "$PWD":/project -w /project aorc python -m pytest tests/test_core.py -v
# Restore, confirm green.
```

A symmetric fixture is the usual way this goes wrong: a mask test used the same
`linspace` for both latitude and longitude, so a deliberately transposed
implementation produced byte-identical output and passed. Make the two axes
different lengths and different ranges so a swap changes the answer.

## Example

```python
@pytest.mark.parametrize(
    "air_temperature_celsius, expected_hpa, tolerance",
    [
        (0.0, 6.11, 0.01),    # triple-point reference
        (20.0, 23.39, 0.05),  # standard psychrometric table
        (30.0, 42.43, 0.05),
        (-10.0, 2.60, 0.05),  # over-ice branch
    ],
)
def test_saturation_vapor_pressure(air_temperature_celsius, expected_hpa, tolerance):
    result = core.saturation_vapor_pressure(air_temperature_celsius)
    assert result == pytest.approx(expected_hpa, abs=tolerance)


def test_saturation_vapor_pressure_increases_with_temperature():
    temperatures = np.linspace(-30.0, 45.0, 76)
    pressures = np.array([core.saturation_vapor_pressure(t) for t in temperatures])
    assert np.all(np.diff(pressures) > 0)
```

The first is anchored to published table values. The second asserts a physical
invariant that no table lookup could fake.

## Common mistakes

| Mistake | Fix |
|---|---|
| Expected value computed by the function under test | Use a published value or an independent implementation |
| Re-typing the implementation's formula in the test | Hand-evaluate once, hardcode the number, comment the source |
| No NaN case | Add one per argument position, plus all-NaN |
| Tolerance tightened until it "looks rigorous" | `rel=1e-5` / `abs=1e-4`; tighter means something is circular |
| Symmetric fixture (same array for both axes) | Different lengths and ranges, so a transposition changes the result |
| Never watched the test fail | Break the implementation, confirm red, restore |

## Running

```bash
docker run --rm -v "$PWD":/project -w /project aorc python -m pytest tests/test_core.py -v
```

There is no local Python environment with the dependencies; running pytest on the
host fails with `ModuleNotFoundError`.
