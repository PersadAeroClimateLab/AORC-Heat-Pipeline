# Three-module refactor of the AORC heat pipeline

Date: 2026-07-21

## Goal

Restructure the pipeline into three auditable modules — scalar science, dataset
orchestration, and a command-line front end — replacing the current flat
collection of scripts. Optimize for ease of auditing over brevity.

## Scope

Five heat metrics are in scope:

| Metric | Registry name | Source |
| --- | --- | --- |
| NWS heat index | `heat_index` | Rothfusz regression |
| Apparent temperature | `apparent_temperature` | Steadman |
| Humidex | `humidex` | Canadian humidex |
| Simplified WBGT | `simplified_wbgt` | ACSM |
| Wet-bulb temperature | `wet_bulb_temperature` | Newton solve on the psychrometric balance |

The Lu & Romps (2022) heat index is **out of scope** for this refactor and will be
revisited separately. `romps_heat_index.py`, `lookup_tables.py`, `lut_runtime.py`,
their tests, and the generated `.npz` lookup tables are deleted; they remain
recoverable from git history.

## Package layout

```
src/aorc_heat/
    __init__.py
    core.py             scalar science; numba; float32
    pipeline.py         xarray / dask / zarr orchestration
    cli.py              argparse entry point
    mask.py             Texas mask generation (today's shpmask.py)
tests/
    test_core.py        literature-derived values; no dask, no network
    test_pipeline.py    synthetic in-memory dataset
pyproject.toml          minimal packaging + console script
```

`src/aorc-heat/` is renamed to `src/aorc_heat/` because a hyphen is not a legal
Python package name.

`mask.py` exists as a fourth module rather than being folded into `pipeline.py`
for two reasons: `*.nc` is gitignored, so `boundary_data/texas_mask.nc` is not
tracked and the generator must ship with the code; and keeping it separate keeps
`geopandas` off `pipeline.py`'s import path.

A minimal `pyproject.toml` enables `pip install -e .`, which removes the
`sys.path` manipulation currently in `tests/conftest.py` and provides an
`aorc-heat` console script.

### Files removed

`metrics.py`, `run.py`, `post_proc.py`, `benchmark.py`, `shpmask.py` (moved to
`src/aorc_heat/mask.py`), `romps_heat_index.py`, `lookup_tables.py`,
`lut_runtime.py`, `tests/test_metrics.py`, `tests/test_romps_heatindex.py`,
`tests/test_lookup_tables.py`, `romps_hi_lut.npz`, `wbt_lut.npz`.

`post_proc.py` disappears because the pipeline now writes a single store
directly rather than one store per metric per statistic per year.

## core.py

Pure scientific formulations in literature units, with no knowledge of AORC
variable names or units. Every function takes scalars, is decorated
`@nb.vectorize(target="cpu", cache=True, fastmath=True)`, and computes in
float32 throughout.

Parameter names encode both the quantity and its unit, and are identical
wherever the same quantity appears:

- `air_temperature_celsius`, `air_temperature_fahrenheit`, `air_temperature_kelvin`
- `relative_humidity_percent`
- `vapor_pressure_hpa`, `saturation_vapor_pressure_hpa`
- `specific_humidity` (kg/kg)
- `air_pressure_hpa`
- `wind_speed_ms`

### Function list

Function names say what is computed; parameter names carry the units; return
units are documented per function. Putting the return unit in the function name
instead would make `vapor_pressure_hpa` both a module-level function and a
parameter of four other functions, and that shadowing is exactly the kind of
thing this refactor is meant to remove.

| Function | Returns |
| --- | --- |
| `celsius_to_fahrenheit(air_temperature_celsius)` | °F |
| `fahrenheit_to_celsius(air_temperature_fahrenheit)` | °C |
| `kelvin_to_celsius(air_temperature_kelvin)` | °C |
| `saturation_vapor_pressure(air_temperature_celsius)` | hPa (Tetens, separate branches above and below 0 °C) |
| `saturation_vapor_pressure_slope(air_temperature_celsius)` | hPa/°C |
| `vapor_pressure(specific_humidity, air_pressure_hpa)` | hPa |
| `relative_humidity(vapor_pressure_hpa, saturation_vapor_pressure_hpa)` | % |
| `heat_index(air_temperature_fahrenheit, relative_humidity_percent)` | °F |
| `apparent_temperature(air_temperature_celsius, vapor_pressure_hpa, wind_speed_ms)` | °C |
| `humidex(air_temperature_celsius, vapor_pressure_hpa)` | °C |
| `simplified_wbgt(air_temperature_celsius, vapor_pressure_hpa)` | °C |
| `wet_bulb_temperature(air_temperature_celsius, specific_humidity, air_pressure_hpa)` | °C |

### Deliberate changes from `metrics.py`

1. **`vapor_pressure` takes ambient pressure.** Today it hardcodes
   1013.25 hPa. AORC provides `PRES_surface`, which is already read for wet-bulb,
   so the real value is used. This is a scientific change, not a pure refactor:
   output values for heat index, apparent temperature, humidex, and sWBGT will
   differ from previous runs, most noticeably at elevation in west Texas.

2. **`wet_bulb_temperature` returns NaN immediately for NaN input.** Added after
   measurement, not in the original design. A NaN iterate can never converge —
   every Newton step is NaN, so neither the tolerance test nor the stall test
   fires — so masked cells ran the full 50-iteration cap at roughly 14× the cost
   of a real cell. With about half the bounding box masked, 96% of wet-bulb
   runtime was being spent on cells that are then discarded. The guard is only
   sound because the fastmath flag set omits `nnan`; under `fastmath=True` LLVM
   assumes no NaN operands and folds the comparison away — the same failure mode
   that originally corrupted NaN propagation here.

3. **`saturation_vapor_pressure_slope` takes only temperature.** Today's
   `ddt_saturation_vapor_pressure(temp, esat)` accepts a pair that a caller
   could make physically inconsistent. Recomputing `esat` internally makes the
   function stand-alone and independently testable, at the cost of one extra
   `exp()` per Newton iteration inside `wet_bulb_temperature`.

4. **The `get_aorc_*` composers are removed.** Unit conversion and composition
   from AORC's raw variables move to `pipeline.py`, leaving `core.py` free of
   dataset-specific knowledge.

## pipeline.py

### Metric registry

A module-level mapping declares, for each metric, the raw AORC variables it
requires and the function that computes its hourly value:

```python
TMP  = "TMP_2maboveground"      # air temperature, K
SPFH = "SPFH_2maboveground"     # specific humidity, kg/kg
PRES = "PRES_surface"           # surface pressure, Pa
UGRD = "UGRD_10maboveground"    # eastward wind, m/s
VGRD = "VGRD_10maboveground"    # northward wind, m/s

METRICS = {
    "heat_index":           MetricSpec(inputs=(TMP, SPFH, PRES),             compute=_heat_index),
    "apparent_temperature": MetricSpec(inputs=(TMP, SPFH, PRES, UGRD, VGRD), compute=_apparent_temperature),
    "humidex":              MetricSpec(inputs=(TMP, SPFH, PRES),             compute=_humidex),
    "simplified_wbgt":      MetricSpec(inputs=(TMP, SPFH, PRES),             compute=_simplified_wbgt),
    "wet_bulb_temperature": MetricSpec(inputs=(TMP, SPFH, PRES),             compute=_wet_bulb_temperature),
}
```

Every metric now requires `PRES_surface`, because `vapor_pressure_hpa` takes
ambient pressure rather than assuming 1013.25 hPa. Wind is the only variable
that a metric selection can avoid reading.

The union of `inputs` over the requested metrics determines which variables are
opened, so requesting only `humidex` never touches the wind arrays. `cli.py`
derives its `--metrics` choices from this same mapping.

### Data flow, per year

```
open_zarr(s3://noaa-nws-aorc-v1-1-1km/{year}.zarr, consolidated=True)
  → isel to the Texas bounding box, snapped outward to native chunk boundaries
  → .astype(float32)
  → .where(mask_bbox)
  → .chunk({"time": 24})          spatial chunking left at the store's native values

shared intermediates — computed once, reused by every metric that needs them:
  air_temperature_celsius   = core.kelvin_to_celsius(TMP_2maboveground)
  air_pressure_hpa          = PRES_surface / 100
  vapor_pressure_hpa        = core.vapor_pressure(SPFH_2maboveground, air_pressure_hpa)
  saturation_vapor_pressure = core.saturation_vapor_pressure(air_temperature_celsius)       [heat index only]
  relative_humidity_percent = core.relative_humidity(vapor_pressure_hpa, saturation_vapor_pressure)
                                                                                            [heat index only]
  wind_speed_ms             = sqrt(UGRD_10maboveground² + VGRD_10maboveground²)             [apparent temp only]

hourly metric values, all converted to °F, then
  .resample(time="1D").min() / .mean() / .max()      via flox
  → three output variables per metric
```

Each raw variable is read from the zarr store exactly once per year and reused
across every metric that needs it; dask's blockwise fusion keeps the shared
intermediates from being materialized more than once.

Time chunking is left at AORC's native 144 hours. (An earlier version rechunked
to 24 hours here; see the revision note below.) A daily resample group still
falls entirely within one chunk — 144 hours is exactly six midnight-aligned days
— so the reduction is chunk-local without any rechunk.

**Revised after implementation.** Rechunking the input to 24 hours was found to
inflate the per-year task graph roughly sixfold (measured: ~1.40M tasks versus
~242k) for bit-identical daily output, because it split each native 144-hour
chunk into six and the reduction immediately regrouped them. Since a native
144-hour chunk already holds six whole days and AORC year files begin at
midnight on 1 January, every daily group is already chunk-local. The rechunk was
removed; `prepare_dataset` now leaves time at the native chunking.

### Chunk alignment (revised after implementation)

The original decision to leave spatial chunking native was correct in intent but
incomplete, and broke the pipeline in practice. `initialize_output_store` derives
the output store's chunk size from the cropped array's *first* chunk, and the
Texas bounding box starts at latitude index 701 and longitude 2803 — both prime,
so they never land on a native boundary. The first chunk was therefore a partial
remainder, every later chunk overlapped two store chunks, and the very first
region write failed zarr's alignment check after a full year of compute.

The fix is to snap the crop outward to native chunk boundaries
(`mask.crop_to_bounding_box(mask, alignment=...)`). This is free rather than
merely cheap: zarr chunks are atomic, so a crop starting mid-chunk already
downloads the whole chunk. Snapping keeps cells that were being transferred and
discarded. It widens the region 42.9% but costs about 1.6% more time, because
every added cell lies outside the boundary and is masked to NaN — and NaN cells
are nearly free (see below).

With the crop aligned, no spatial rechunk is needed anywhere: source, dask
graph, and output store all share one chunk grid.

### Precision

Raw AORC variables are float64 on disk and are cast to float32 immediately after
cropping. The kernels compute in float32 regardless; casting early means the
intermediates are narrow too. Measured, the memory-bound closed-form metrics run
about 1.5× faster on float32 operands, while `wet_bulb_temperature` — which is
compute-bound on `exp()` — is indifferent.

One hazard worth recording: because the kernels use bare `@nb.vectorize` with no
signature list, loops compile lazily and registration order is sticky. If a
float64 call registers first, later float32 calls find no matching loop, get
upcast to float64, and run *slower* than passing float64 would have. Cast
wholesale or not at all.

### Output store

A single store at `{output_dir}/aorc_heat_metrics.zarr`, dimensions
`(time, latitude, longitude)` where `time` is one entry per day across the
requested year range and the spatial grid is the Texas mask bounding box.

Variables are named `{metric}_{statistic}` — for example `heat_index_min`,
`heat_index_mean`, `heat_index_max`. All are float32 in °F. Cells outside the
Texas mask are NaN.

**The store contains only the metrics that have actually been computed.** A
metric that was never requested has no variables in the store at all — not
NaN-filled ones. NaN appears only where the Texas mask excludes a cell, or
transiently in a freshly allocated variable that has not yet been filled by its
region writes.

This supersedes the original framing of "parameters that return NaNs for metrics
that shouldn't be computed". Writing NaN for unselected metrics would mean a
later run for one metric erases every metric computed earlier, since region
writes cover whole variables. Omitting them instead makes the store
incrementally extensible, which is the behavior actually wanted.

### Incremental metric computation

`run()` supports building a store up over several invocations:

1. If the store does not exist, `pending = requested`.
2. If it does exist, validate that its time axis matches the daily axis implied
   by `--start-year`/`--end-year` and that its spatial coordinates match the mask
   bounding box. Mismatch is a hard error — never a partial write.
3. `pending = [m for m in requested if f"{m}_mean" not in store.data_vars]`.
   Metrics already present are skipped, not recomputed.
4. If `pending` is empty, log and exit successfully.
5. Write a template holding only the `pending` variables — dask-backed, all NaN,
   full time range — with `to_zarr(mode="w" if new else "a", compute=False)`.
   This allocates the variables without computing anything.
6. For each year, compute the daily metrics and write them with
   `region={"time": slice(start_index, end_index)}`.

Region writes require the target variables to already exist in the store, which
step 5 guarantees. Coordinate variables are dropped from each yearly result
before the region write, since they are already present in the store.

Re-running a year that was previously written simply overwrites that region.
There is no per-year resume state; the granularity of "skip work already done"
is the metric, not the year.

Revised after implementation: the original skip check tested only whether a
variable existed. Because `initialize_output_store` allocates variables before
any year is computed, a crash at any point after allocation — including before
the first write — left a store that looked finished but held 100% NaN, and a
rerun printed "nothing to compute" and exited 0. Silent, and near-certain on a
46-year S3 job.

Completion is now recorded explicitly: `record_completed_years` writes each
successfully written year into a `completed_years` zarr attribute after
`write_block` returns, and `pending_metrics` reports a metric done only when the
recorded years cover the whole requested range. The record is written after the
write succeeds, so a crash mid-write leaves the year unrecorded.

Remaining limitation, accepted: recovery is at metric granularity, not year. A
partially complete metric is recomputed from its first year rather than resumed.
Redundant work, but far better than silently reporting success on empty data.

### Calendar handling

The full daily time axis is constructed once with a single explicit calendar
(`xr.date_range(..., freq="D", use_cftime=True, calendar="standard")`). Each
year's daily result is converted to that calendar before its region write. This
is the `convert_calendar` step that `post_proc.py` performed after the fact;
doing it up front is what makes a single continuous store possible.

### Public surface

```python
daily_metrics(aorc_dataset, metric_names) -> xr.Dataset
    Lazy. Returns exactly the requested metrics × {min, mean, max}.

initialize_output_store(store_path, metric_names, time_axis, template)
    Creates or extends the store with NaN-filled variables. No computation.
    `template` is one block of daily_metrics output, supplying the spatial
    coordinates and chunk sizes.

write_block(store_path, daily, time_axis)
    Writes a contiguous block of days into its region, locating that region
    from the block's own timestamps rather than from a year number.

run(output_dir, metric_names, start_year, end_year)
    Orchestrates the skip check, template write, and per-year region writes.
```

The output store is chunked at one day per time chunk. Input chunked at 24 hours
yields daily output chunked at exactly 1, so every region write lands on chunk
boundaries and no two writes can touch the same chunk — which is what makes the
region writes safe without `safe_chunks=False`. Variables are allocated with an
explicit `_FillValue` of NaN, because zarr otherwise defaults float arrays to a
fill value of 0 and unwritten days would read back as plausible-looking zeroes.

### Metadata

Global attributes carry over from `run.py`: description, `source_version`
("AORC Version 1.1"), and `source_url`. Per-variable attributes record
`units: "deg_F"`, `source_timestep: "hourly"`, and a description naming the
statistic and the 24-hour window it covers.

## cli.py

```
aorc-heat (--all | --metrics heat_index apparent_temperature …)
          --output-dir PATH
          --cores N
          --memory-limit 10GB
          [--start-year 1979]
          [--end-year 2024]
```

- `--all` and `--metrics` are mutually exclusive; exactly one is required.
- `--metrics` uses argparse `choices` drawn from `METRICS`, so a misspelled
  metric fails before the Dask cluster starts.
- `--cores` maps to `LocalCluster(n_workers=cores, threads_per_worker=1)`. The
  current 80 workers × 2 threads becomes a single knob: the `@nb.vectorize`
  kernels are single-threaded, so worker count is the only dimension that
  affects throughput now.
- `--memory-limit` passes straight to `LocalCluster`.
- The Dask/Bokeh dashboard stays on port 8002, as in `run.py` today.
- `--start-year` and `--end-year` default to 1979 and 2024, the full AORC range.
  They exist so a single-year validation run is possible without editing code.

`cli.py` is responsible for cluster lifecycle (`LocalCluster`, `client.shutdown()`
in a `finally`) and for `xr.set_options(use_flox=True)`. It contains no science
and no dataset logic.

## Error handling

- Missing `boundary_data/texas_mask.nc`: regenerate from the shapefile via
  `mask.py`, as today, and write it back. Missing shapefile raises with the
  expected path in the message.
- Existing output store with a time axis or spatial grid that does not match the
  request: raise before any write, naming the mismatch.
- Unknown metric name: rejected by argparse before cluster startup.
- S3 or zarr read failures propagate; a failed year aborts the run rather than
  silently leaving a NaN region.

## Testing

### `tests/test_core.py`

Ports the existing literature-derived expectations from `tests/test_metrics.py`
onto the renamed functions, keeping the published references as the source of
truth so that a transcription error is caught rather than reproduced:

- Temperature conversions against definitional fixed points.
- Heat index against the NWS chart and Rothfusz regression, ±2 °F.
- Saturation vapor pressure against standard psychrometric tables.
- Apparent temperature, humidex, and simplified WBGT against their definitional
  formulae evaluated by hand.
- Wet-bulb temperature against Stull (2011) and the identity `Tw == T` at
  saturation.

New coverage for the two changed functions: `vapor_pressure_hpa` across a range
of ambient pressures, and `saturation_vapor_pressure_slope` against a numerical
derivative of `saturation_vapor_pressure_hpa`.

Tolerances account for float32 precision. No dask, zarr, or network access.

### `tests/test_pipeline.py`

Builds a synthetic in-memory dataset — 48 hours, a 2×2 grid, AORC variable
names and units — and asserts:

- `daily_metrics` returns exactly the requested metrics × three statistics, and
  no variables for unrequested metrics.
- The output time axis has two days.
- Daily min, mean, and max match a hand-rolled numpy reduction over each
  24-hour block.
- NaN input cells produce NaN outputs rather than propagating garbage.

Store-level behavior is tested against a `tmp_path` zarr store: writing one
metric, then re-running with two metrics, and asserting that the first metric's
values are unchanged and the second has been appended.

## Out of scope

- Lu & Romps (2022) heat index and the lookup-table machinery.
- Configurable regions; Texas stays hardcoded.
- Per-year resume state.
- Reworking the Docker image beyond whatever the package rename requires.
