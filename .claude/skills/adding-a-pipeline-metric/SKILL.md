---
name: adding-a-pipeline-metric
description: Use when adding a new heat metric to the AORC pipeline, wiring a core.py formula into the METRICS registry, or when a metric needs a shared intermediate quantity that does not exist yet.
---

# Adding a Pipeline Metric

## Overview

`pipeline.py` translates AORC's raw units into what `core.py` expects, applies the
kernels, and reduces 24 hourly values to a daily min/mean/max. A metric is three
things: an entry in `METRICS`, a `_compute` function, and whatever shared
intermediates it needs.

**No science belongs in `pipeline.py`.** If you are writing a formula here, it
belongs in `core.py` — **REQUIRED SUB-SKILL:** use `adding-a-core-function`.

## The unit boundary

This is where silent scientific errors get introduced. AORC and `core.py` disagree
about everything:

| Quantity | AORC gives | `core.py` wants |
|---|---|---|
| Air temperature | `TMP_2maboveground`, Kelvin | degrees Celsius |
| Specific humidity | `SPFH_2maboveground`, kg/kg | kg/kg (no conversion) |
| Surface pressure | `PRES_surface`, Pascals | hectopascals |
| Wind | `UGRD_`/`VGRD_10maboveground`, components | **speed**, `sqrt(u² + v²)` |

Two traps worth naming:

- **`core.heat_index` takes Fahrenheit and returns Fahrenheit.** Convert before
  the call, and do not convert the result again. A double conversion produces
  plausible numbers and no error.
- **`core.vapor_pressure` takes ambient pressure as a second argument.** It does
  not assume 1013.25 hPa. Every metric therefore needs `PRES_surface`.

All metric outputs are **degrees Fahrenheit**, `units: "deg_F"`.

## Shared intermediates

`_derived_inputs` builds each quantity **once** and hands it to every metric that
needs it, so the four vapour-pressure metrics resolve to a single node in the dask
graph rather than four. Never recompute an intermediate inside a metric function.

Unconditional (every metric needs them): `air_temperature_celsius`,
`air_pressure_hpa`, `specific_humidity`, `vapor_pressure_hpa`.

Conditional — built only when a metric that needs them was requested:
`relative_humidity_percent` (heat index), `wind_speed_ms` (apparent temperature).

If your metric needs a new intermediate that others might share, add it
conditionally in the same style:

```python
if "dew_point" in metric_names or "frost_risk" in metric_names:
    derived["dew_point_celsius"] = _apply(
        core.dew_point_celsius,
        derived["air_temperature_celsius"],
        derived["relative_humidity_percent"],
    )
```

Guard it on the metrics that use it, not on a raw variable name — that is what
keeps an unrequested metric from pulling data off S3.

## Adding the metric

**1. Write the compute function.** Take the derived mapping, return hourly °F.
Use `_apply` for every kernel call — it carries the NaN warning suppression.

```python
def _dew_point(derived):
    return _apply(
        core.celsius_to_fahrenheit,
        _apply(
            core.dew_point_celsius,
            derived["air_temperature_celsius"],
            derived["relative_humidity_percent"],
        ),
    )
```

**2. Register it.** `inputs` is the raw AORC variables that must be read; it
drives both what gets pulled from S3 and the CLI's `--metrics` choices.

```python
"dew_point": MetricSpec(
    inputs=(AIR_TEMPERATURE, SPECIFIC_HUMIDITY, SURFACE_PRESSURE),
    compute=_dew_point,
),
```

**3. Add the intermediate** if it needs one that does not exist.

**4. Test it.** **REQUIRED SUB-SKILL:** use `testing-pipeline-metrics`. A metric
without a numeric cross-check is not done — the conversion boundary above is
exactly what that test catches.

Nothing else needs touching. `cli.py` derives `--metrics` from `METRICS`, and the
store gains `dew_point_{min,mean,max}` on the next run.

## How it lands in the store

Each metric becomes three variables. The store holds **only metrics actually
computed** — a later run with an added metric appends its variables and leaves
existing ones bit-identical, so adding a metric never means recomputing the
others.

Completion is tracked per year in the `completed_years` store attribute. A metric
counts as done only when every requested year is recorded, so an interrupted run
recomputes rather than silently reporting success over NaN.

## Checklist

- [ ] Formula lives in `core.py`, not here
- [ ] Every kernel call goes through `_apply`
- [ ] Kelvin → Celsius, Pascals → hPa, u/v → speed, each applied exactly once
- [ ] Output converted to °F exactly once
- [ ] Shared intermediates reused, never recomputed
- [ ] New intermediates guarded on the metrics that need them
- [ ] `inputs` lists every raw variable the metric touches
- [ ] Numeric cross-check test written and proven to fail against a conversion bug

## Common mistakes

| Mistake | Consequence |
|---|---|
| Converting `core.heat_index`'s result to °F | Double conversion; plausible numbers, no error |
| Assuming `vapor_pressure` uses standard pressure | Wrong at elevation; it takes ambient pressure |
| Recomputing vapour pressure inside the metric | Four dask nodes instead of one, over billions of points |
| Calling a kernel directly instead of via `_apply` | Loses NaN warning suppression; floods the log |
| Omitting a variable from `inputs` | `KeyError` mid-run, after the store is already allocated |
| Putting the formula in `pipeline.py` | Untestable against literature values; breaks the module boundary |

## Verifying

```bash
docker run --rm -v "$PWD":/project -w /project aorc python -m pytest tests/ -q
docker run --rm -v "$PWD":/project -w /project aorc aorc-heat --help   # new metric listed
```
