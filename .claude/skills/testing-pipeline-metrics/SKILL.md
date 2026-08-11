---
name: testing-pipeline-metrics
description: Use when writing or reviewing tests for AORC pipeline metrics, daily min/mean/max reductions, unit conversion between AORC and core.py, region masking, or the zarr output store.
---

# Testing Pipeline Metrics

## Overview

`core.py` tests prove the science is right. **Pipeline tests prove the wiring is
right** — that the correct AORC variable reached the correct kernel argument in
the correct unit, and that 24 hourly values reduced to the correct day.

Every test runs against an in-memory synthetic dataset. **Nothing in the suite
touches S3 or the network.**

## The cross-check pattern

The one test every metric needs. Recompute the expected value using `core`
functions called **directly**, never through pipeline internals.

```python
def test_daily_metrics_heat_index_matches_manual_numpy(synthetic_aorc_dataset):
    result = pipeline.daily_metrics(synthetic_aorc_dataset, ["heat_index"]).compute()

    loaded = synthetic_aorc_dataset.compute()
    air_temperature_celsius = core.kelvin_to_celsius(loaded["TMP_2maboveground"].values)
    air_temperature_fahrenheit = core.celsius_to_fahrenheit(air_temperature_celsius)
    air_pressure_hpa = (loaded["PRES_surface"].values / 100.0).astype(np.float32)
    vapor = core.vapor_pressure(loaded["SPFH_2maboveground"].values, air_pressure_hpa)
    saturation = core.saturation_vapor_pressure(air_temperature_celsius)
    relative_humidity = core.relative_humidity(vapor, saturation)
    hourly = core.heat_index(air_temperature_fahrenheit, relative_humidity)

    for day in (0, 1):
        window = hourly[day * 24 : (day + 1) * 24]
        np.testing.assert_allclose(
            result["heat_index_mean"].isel(time=day).values, window.mean(axis=0), rtol=1e-5
        )
```

**Why direct calls matter.** Routing the expected value through
`pipeline._heat_index` or `pipeline._derived_inputs` would make the test
tautological — it would reproduce a wiring bug rather than catch it. Spelling out
the conversions by hand is the entire point: the test asserts that the pipeline
performs *these specific conversions in this specific order*.

This is not hypothetical. For a while only `humidex` had such a test, leaving
`heat_index` and `apparent_temperature` — the two with the trickiest conversion
ordering — covered by nothing but a min ≤ mean ≤ max check.

## Prove the test can fail

A cross-check that has never gone red proves nothing. Introduce the specific bug
it targets and watch it fail:

- **heat_index**: wrap the expected value in an extra `celsius_to_fahrenheit`
  (the double-conversion bug)
- **apparent_temperature**: use raw `u` instead of `sqrt(u² + v²)` as wind speed
- **any metric**: swap two arguments of the same dtype

Then restore and confirm green. Report both outcomes.

## What else to cover

| Check | What it catches |
|---|---|
| Only requested metrics returned | Store gaining variables nobody asked for |
| Three statistics per metric, correct day count | Off-by-one in the resample |
| Output dtype is float32 | Silent upcast doubling store size |
| Masked cells NaN, unmasked finite | Mask not applied, or NaN not propagating |
| min ≤ mean ≤ max | Cheap sanity net across every metric |
| Variable attributes (`deg_F`, `hourly`) | Metadata silently dropped by resample |

For the store: build it in `tmp_path`, write, reopen, assert. Cover that unwritten
days read back as **NaN and not 0.0**, that adding a metric leaves existing
variables bit-identical, and that consecutive years both survive the chunk they
share at their boundary.

## Day boundaries

Encode the day index in the value so a misplaced day is visible rather than
plausible:

```python
values = np.repeat((np.arange(days, dtype=np.float32) + offset)[:, None, None], 2, axis=1)
# ... write, reopen ...
np.testing.assert_array_equal(written.isel(latitude=0, longitude=0).values, np.arange(731))
```

Random data would hide an off-by-one behind numbers that all look equally
reasonable.

## The fixture

`synthetic_aorc_dataset` in `tests/conftest.py`: 48 hours, 2×2 grid, all five AORC
variables, cftime standard calendar, chunked at 24 hours. Two days is deliberate —
enough to catch a reduction that collapses the time axis.

Build a local dataset instead when you need a larger grid. Anything above **8
cells** crosses the SIMD width where numba's behaviour on NaN lanes changes, so
warning-related tests need at least that.

## Common mistakes

| Mistake | Fix |
|---|---|
| Expected value routed through `pipeline._*` internals | Call `core` functions directly with explicit arguments |
| Testing only `humidex` because it is simplest | The tricky conversions are `heat_index` and `apparent_temperature` |
| Random values in a day-boundary test | Encode the day index so a misplacement is visible |
| Asserting "no NaN" on a masked region | Masked cells *must* be NaN; assert both halves separately |
| Reaching for real AORC data | The suite must never touch S3; use the fixture |
| Fixture too small for the behaviour under test | Below 8 cells the scalar path runs and SIMD effects vanish |

## Running

```bash
docker run --rm -v "$PWD":/project -w /project aorc python -m pytest tests/test_pipeline.py -v
```

There is no local Python environment with the dependencies.
