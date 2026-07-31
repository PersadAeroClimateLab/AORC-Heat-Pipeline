"""Derive daily heat metrics from the NOAA AORC hourly zarr archive.

One year is processed at a time. Each raw AORC variable is read exactly once and
the intermediate quantities built from it -- air temperature in Celsius, vapour
pressure, and so on -- are shared by every metric that needs them, so the four
metrics that depend on vapour pressure resolve to a single node in the dask
graph rather than four.

Time chunking is left at AORC's native 144 hours. That is exactly six days, and
year files start at midnight on 1 January, so every native chunk boundary falls
on a midnight and each chunk holds six whole days. The daily reduction groups
every 24 consecutive hours with `coarsen(time=24)`, which is therefore pure
blockwise regrouping -- it never crosses a chunk boundary and, unlike
`resample`, builds no pandas grouper to get there. That blockwise grouping is
exactly why `daily_metrics` requires its input to start at 00Z and hold a whole
number of 24-hour days: `coarsen` has no calendar awareness of its own to fall
back on. Splitting to 24-hour chunks instead (as an earlier version did)
multiplied the task graph roughly sixfold for bit-identical output, because the
reduction immediately regroups what the split had just divided.

Spatial chunking is left at the source store's native values, and the region
crop is snapped outward to start on a native chunk boundary (see
`mask.crop_to_bounding_box`). Zarr chunks are atomic, so a crop starting
mid-chunk downloads the whole chunk anyway; snapping keeps the cells already
paid for instead of discarding them, and it makes every chunk boundary in the
cropped array line up with the output store's -- the condition zarr region
writes require. Imposing our own spatial chunking instead would mean a rechunk
on every read for no benefit.

The bounding box is not, however, the unit of work. A box is a rectangle and a
boundary is not, so many of the chunks inside the box hold no in-region cells at
all -- for Texas, 38 of 88, measured. `run` therefore iterates the occupied
blocks from `mask.occupied_chunk_blocks` instead of the whole box, one dask
graph per block. Because the opened year is lazy, a chunk-aligned `isel` never
fetches the chunks it excludes, so those 43% of bytes are never read from S3,
never decompressed, never pushed through a kernel, and never written; they keep
the NaN `initialize_output_store` gave them, which is the correct value for
them. Texas is convex within each row of chunks, so its 50 occupied chunks merge
into just 11 blocks -- 11 writes per year, each large enough to keep a wide dask
cluster busy.

Raw AORC variables are float64 on disk. They are cast to float32 immediately
after cropping: the science kernels compute in float32 regardless, and the
memory-bound closed-form metrics run ~1.5x faster on float32 operands.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import dask.array
import numpy as np
import pandas as pd
import xarray as xr
import zarr

from aorc_heat import core

AORC_S3_BASE_PATH = "s3://noaa-nws-aorc-v1-1-1km/"

AIR_TEMPERATURE = "TMP_2maboveground"
SPECIFIC_HUMIDITY = "SPFH_2maboveground"
SURFACE_PRESSURE = "PRES_surface"
EASTWARD_WIND = "UGRD_10maboveground"
NORTHWARD_WIND = "VGRD_10maboveground"

DAILY_STATISTICS = ("min", "mean", "max")
PASCALS_PER_HECTOPASCAL = np.float32(100.0)

#: Minimum count of non-missing hourly samples (out of 24) a day must have
#: before its min/mean/max are reported, rather than suppressed to NaN.
#:
#: Strict by default -- every source gap measured in the AORC archive so far
#: is whole-day (see the module docstring's note on source data gaps), so 24
#: and a looser value like 20 discard exactly the same data today. It also
#: matters because daily max/min are order statistics, not just a mean with a
#: wider error bar: a day missing its hottest hours is not noisier, it is
#: biased low, and there is no safe way to interpolate around a missing
#: afternoon. Lower this to ~20 to adopt the common climatological convention
#: of keeping a day once at least 20 of its 24 hours are present, if partial
#: days should be kept instead of dropped.
MINIMUM_VALID_HOURS = 24

GLOBAL_ATTRIBUTES = {
    "description": "Heat metrics derived from NOAA NWS AORC data provided via AWS S3 Zarr Bucket",
    "source_version": "AORC Version 1.1",
    "source_url": "https://registry.opendata.aws/noaa-nws-aorc",
}


def _quiet_on_masked_nan(kernel):
    """Wrap a `core` kernel so masked cells do not raise spurious FP warnings.

    `prepare_dataset` masks with `.where`, so most cells reaching these kernels
    are NaN by design. The kernels handle that correctly -- NaN in, NaN out --
    but the branchy ones still emit
    "RuntimeWarning: invalid value encountered in ...", once per block, which
    on a 46-year run buries anything worth reading.

    The warning is an artefact of vectorisation, not a bad computation. Above
    the SIMD width (measured: exactly 8 float32 lanes, one AVX register) LLVM
    if-converts the branches into selects and evaluates *both* sides for *all*
    lanes. The NaN lanes reach an ordered comparison, which IEEE-754 says must
    raise the invalid-operation flag, and numpy reports that flag after the
    loop even though those lanes' results are discarded by the select.

    Suppressing `invalid` here is narrow -- it covers only kernels this module
    deliberately feeds NaN -- and is backed by
    `test_finite_inputs_never_produce_non_finite_outputs`, which fails if any
    kernel ever turns finite input into NaN. Without that test this would be
    hiding evidence rather than filtering noise.
    """

    def apply_quietly(*blocks):
        with np.errstate(invalid="ignore"):
            return kernel(*blocks)

    apply_quietly.__name__ = getattr(kernel, "__name__", "kernel")
    return apply_quietly


def _apply(kernel, *data_arrays):
    """Apply a `core` scalar kernel elementwise across dask-backed arrays."""
    return xr.apply_ufunc(
        _quiet_on_masked_nan(kernel),
        *data_arrays,
        dask="parallelized",
        output_dtypes=[np.float32],
    )


# --------------------------------------------------------------------------
# Daily reduction: NaN-quiet replacements for coarsen's nanmin/nanmean/nanmax
# --------------------------------------------------------------------------
#
# A masked cell is NaN for all 24 hours of every day (see the module
# docstring's NaN contract), so `daily_metrics`'s `coarsen(time=24)` window
# reduces an entirely-NaN slice for roughly half the domain, every day. The
# obvious implementation -- `coarsen(...).min()/.mean()/.max()`, which
# resolve to `numpy.nanmin`/`nanmean`/`nanmax` -- computes the right answer
# (NaN) for those windows, but *also* calls `warnings.warn` unconditionally
# to report it, once per block, which buries anything worth reading on a
# multi-decade run exactly like the FP-flag warning `_quiet_on_masked_nan`
# already suppresses for the hourly kernels. That warning is not gated by
# `np.errstate` the way the kernels' invalid-operation flag is, and -- because
# these reducers are evaluated lazily by dask, potentially in a worker thread
# long after this module's code has returned -- wrapping the call site in
# `warnings.catch_warnings()` would not reliably reach the warning at all.
#
# The three functions below sidestep the problem instead of suppressing it:
# they never execute a code path that raises the warning, so nothing needs
# catching regardless of when or on which thread dask evaluates them.
#   - min/max replace NaN with +/-inf before an ordinary (non-nan-aware) min
#     or max, then map any remaining +/-inf in the result -- meaning every
#     value in the window was NaN -- back to NaN. Plain `min`/`max` never
#     special-cases an all-inf slice, so this never touches the warning path.
#   - mean divides by `count` only after replacing a zero count with one,
#     so total/count is never literally 0/0; the result is then explicitly
#     overwritten with NaN wherever count actually was zero. No division by
#     zero ever happens, so there is nothing for `np.errstate` to guard.
# Correctness against `numpy.nanmin`/`nanmean`/`nanmax` is covered by the
# `test_daily_metrics_*_matches_manual_numpy` family; the no-warning property
# is covered by `test_masked_cells_do_not_emit_runtime_warnings`.
def _reduce_min_quietly(values, axis, **kwargs):
    filled = np.where(np.isnan(values), np.asarray(np.inf, dtype=values.dtype), values)
    reduced = filled.min(axis=axis)
    return np.where(np.isinf(reduced), np.asarray(np.nan, dtype=values.dtype), reduced)


def _reduce_max_quietly(values, axis, **kwargs):
    filled = np.where(np.isnan(values), np.asarray(-np.inf, dtype=values.dtype), values)
    reduced = filled.max(axis=axis)
    return np.where(np.isinf(reduced), np.asarray(np.nan, dtype=values.dtype), reduced)


def _reduce_mean_quietly(values, axis, **kwargs):
    valid = ~np.isnan(values)
    count = valid.sum(axis=axis, dtype=np.float32)
    total = np.where(valid, values, np.asarray(0, dtype=values.dtype)).sum(
        axis=axis, dtype=np.float32
    )
    safe_count = np.where(count == 0, np.float32(1), count)
    mean = total / safe_count
    return np.where(count == 0, np.asarray(np.nan, dtype=np.float32), mean)


def _count_valid_hours_reducer(values, axis, **kwargs):
    """Count of a window's non-NaN samples -- how many of the 24 hours counted.

    Takes the same `(values, axis, **kwargs)` shape as the three reducers
    above so it can be plugged into the same `Coarsen.reduce` call, but it is
    not a "quiet" reducer in their sense: a plain `count_nonzero` never touches
    `numpy.nanmin`/`nanmean`/`nanmax`'s empty-slice warning path in the first
    place, all-NaN window or not.
    """
    return np.count_nonzero(~np.isnan(values), axis=axis)


#: One quiet reducer per name in `DAILY_STATISTICS`, passed to
#: `Coarsen.reduce` in place of the `.min()`/`.mean()`/`.max()` convenience
#: methods (which resolve to the warning-raising `nanmin`/`nanmean`/`nanmax`).
_QUIET_DAILY_REDUCERS = {
    "min": _reduce_min_quietly,
    "mean": _reduce_mean_quietly,
    "max": _reduce_max_quietly,
}


def _variable_attributes(metric_name, statistic):
    """CF-style attributes for one output variable.

    Units come from the metric's registry entry: most metrics are degrees
    Fahrenheit, but `relative_humidity` is a percent.
    """
    readable = metric_name.replace("_", " ")
    return {
        "units": METRICS[metric_name].units,
        "source_timestep": "hourly",
        "description": f"Daily {statistic} of hourly {readable} taken across 24 UTC hours",
    }


def _valid_hours_attributes(metric_name):
    """CF-style attributes for one metric's `_valid_hours` variable.

    Units are always "hours" -- never the metric's own units from
    `_variable_attributes`, since this counts non-missing hourly samples
    rather than reporting the metric's physical quantity.
    """
    readable = metric_name.replace("_", " ")
    return {
        "units": "hours",
        "source_timestep": "hourly",
        "description": (
            f"Count of the 24 hourly {readable} values that were non-missing and "
            "contributed to that day's minimum, mean, and maximum."
        ),
    }


# --------------------------------------------------------------------------
# Shared intermediates
# --------------------------------------------------------------------------
def _derived_inputs(aorc_dataset, metric_names):
    """Build the intermediate quantities the requested metrics need.

    Each quantity is built once and handed to every metric that uses it. A base
    intermediate is built only when the raw variable it derives from is present,
    because `open_aorc_year` subsets the dataset to exactly the requested
    metrics' `inputs` -- so a variable is present precisely when some requested
    metric needs it. Building unconditionally would `KeyError` on a metric like
    `temperature` that reads air temperature alone. Relative humidity and wind
    speed are further guarded on the specific metric that consumes them.

    :param aorc_dataset: Dataset holding the raw AORC variables the metrics need
    :param metric_names: Names of the metrics that will be computed
    :return: Mapping of intermediate name to lazy DataArray
    """
    derived = {}

    if AIR_TEMPERATURE in aorc_dataset:
        derived["air_temperature_celsius"] = _apply(
            core.kelvin_to_celsius, aorc_dataset[AIR_TEMPERATURE]
        )
    if SPECIFIC_HUMIDITY in aorc_dataset:
        derived["specific_humidity"] = aorc_dataset[SPECIFIC_HUMIDITY]
    if SURFACE_PRESSURE in aorc_dataset:
        derived["air_pressure_hpa"] = aorc_dataset[SURFACE_PRESSURE] / PASCALS_PER_HECTOPASCAL
    if SPECIFIC_HUMIDITY in aorc_dataset and SURFACE_PRESSURE in aorc_dataset:
        derived["vapor_pressure_hpa"] = _apply(
            core.vapor_pressure, derived["specific_humidity"], derived["air_pressure_hpa"]
        )

    if "heat_index" in metric_names or "relative_humidity" in metric_names:
        saturation = _apply(core.saturation_vapor_pressure, derived["air_temperature_celsius"])
        # Clamped to [0, 100] here, once, so both consumers -- heat_index (whose
        # regression is only defined over that range; a supersaturated cell can
        # otherwise swing the result tens of degrees) and the exported
        # relative_humidity metric -- see the same value. Float noise in
        # saturated air routinely pushes the raw ratio a little past 100.
        derived["relative_humidity_percent"] = _apply(
            core.relative_humidity, derived["vapor_pressure_hpa"], saturation
        ).clip(0, 100)

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


def _temperature(derived):
    """Dry-bulb air temperature in degrees Fahrenheit.

    Not a heat-stress index -- just the raw AORC temperature carried through
    the same daily reduction, converted Kelvin -> Celsius -> Fahrenheit.
    """
    return _apply(core.celsius_to_fahrenheit, derived["air_temperature_celsius"])


def _relative_humidity(derived):
    """Relative humidity as a percentage.

    Already built as a shared intermediate (heat index needs it too) and
    already a percent, so this is a passthrough with no further conversion --
    unlike every other metric, its output is not degrees Fahrenheit.
    """
    return derived["relative_humidity_percent"]


def _specific_humidity(derived):
    """Specific humidity in kg/kg.

    The raw AORC value carried straight through: core.py's science is all in
    kg/kg, so there is no conversion, and the output is neither degrees
    Fahrenheit nor a percent.
    """
    return derived["specific_humidity"]


@dataclass(frozen=True)
class MetricSpec:
    """How to compute one metric and what raw AORC variables it needs.

    :param inputs: Raw AORC variable names that must be read
    :param compute: Callable taking the derived-inputs mapping, returning the
        hourly metric value in the units named below
    :param units: Output units for the CF `units` attribute; defaults to the
        degrees Fahrenheit that every heat-stress metric produces
    """

    inputs: tuple
    compute: Callable
    units: str = "deg_F"


#: Most metrics need surface pressure, because `core.vapor_pressure` takes the
#: ambient pressure rather than assuming a fixed 1013.25 hPa. `temperature` is
#: the exception -- it needs only air temperature. Wind is read only by
#: `apparent_temperature`.
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
    "temperature": MetricSpec(
        inputs=(AIR_TEMPERATURE,),
        compute=_temperature,
    ),
    "relative_humidity": MetricSpec(
        inputs=(AIR_TEMPERATURE, SPECIFIC_HUMIDITY, SURFACE_PRESSURE),
        compute=_relative_humidity,
        units="percent",
    ),
    "specific_humidity": MetricSpec(
        inputs=(SPECIFIC_HUMIDITY,),
        compute=_specific_humidity,
        units="kg/kg",
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


def native_spatial_chunks(dataset):
    """The source store's (latitude, longitude) chunk sizes.

    Read from the zarr encoding rather than from dask, because xarray may
    coalesce several stored chunks into one dask chunk when opening. It is the
    on-disk chunk grid that a region crop has to align to, since zarr chunks
    are atomic: a read starting mid-chunk still fetches the whole chunk.

    :param dataset: An opened AORC dataset
    :return: (latitude_chunk, longitude_chunk) in cells
    """
    for name in (AIR_TEMPERATURE, SPECIFIC_HUMIDITY, SURFACE_PRESSURE):
        if name not in dataset.variables:
            continue
        chunks = dataset[name].encoding.get("chunks")
        if chunks and len(chunks) == 3:
            return int(chunks[1]), int(chunks[2])
    raise ValueError(
        "Could not determine the source store's native spatial chunking from "
        "its zarr encoding; cannot align the region crop to chunk boundaries."
    )


def prepare_dataset(dataset, region):
    """Crop to the region bounding box, cast to float32, and mask.

    No rechunking happens here. AORC's native 144-hour time chunk already holds
    six whole, midnight-aligned days, so `daily_metrics`'s `coarsen(time=24)`
    grouping never crosses a chunk boundary -- splitting to 24 hours would only
    inflate the task graph (~6x, measured) for identical output. Spatial
    chunking is likewise native: `region`'s slices are snapped to chunk
    boundaries, so the cropped array lines up with both the source store and
    the output store as-is.

    The cast to float32 happens before masking so that every downstream
    intermediate is float32 too, not just the kernel outputs.

    :param dataset: Full-domain AORC dataset
    :param region: A `mask.Region` whose slices are snapped to chunk boundaries
    :return: Cropped, masked float32 dataset at the source's native chunking
    """
    cropped = dataset.isel(
        latitude=region.latitude_slice, longitude=region.longitude_slice
    ).astype(np.float32)
    return cropped.where(region.mask)


def _first_timestamp(aorc_dataset):
    """The dataset's first time value, normalised to something with `.hour`.

    Mirrors `_as_comparable_dates`: a "standard"-calendar time coordinate may
    arrive as either cftime objects (already `.hour`-bearing) or, once it has
    round-tripped through zarr, plain `numpy.datetime64` (which is not), so
    the latter is converted through `pandas.Timestamp`.

    :param aorc_dataset: Dataset with a `time` dimension of size >= 1
    :return: The first timestamp, with `.hour`/`.minute`/`.second`/`.microsecond`
    """
    first = aorc_dataset["time"].values[0]
    if isinstance(first, np.datetime64):
        first = pd.Timestamp(first)
    return first


def _validate_starts_on_midnight_whole_days(aorc_dataset):
    """Raise unless the input begins at 00Z and holds a whole number of days.

    `resample(time="1D")` grouped by calendar date, so it tolerated an
    off-midnight start or a partial trailing day -- it just produced a short
    or oddly-bounded first/last group. `coarsen(time=24)` has no calendar
    awareness: it blindly bundles every 24 consecutive samples, so this
    assumption (previously just a comment on `prepare_dataset`) becomes
    load-bearing. Both `daily_time_axis` and `write_block`'s store lookup
    depend on day *n* of the input actually being calendar day *n*, so a
    violation must fail loudly here rather than silently mislabel every day
    from the violation onward.

    :param aorc_dataset: Prepared AORC dataset about to be reduced
    :raises ValueError: If the first timestamp is not 00Z, or the time
        dimension's length is not a multiple of 24 hours
    """
    hour_count = aorc_dataset.sizes["time"]
    if hour_count % 24 != 0:
        raise ValueError(
            "daily_metrics requires a whole number of 24-hour days; got "
            f"{hour_count} hours."
        )
    first = _first_timestamp(aorc_dataset)
    if (first.hour, first.minute, first.second, first.microsecond) != (0, 0, 0, 0):
        raise ValueError(
            "daily_metrics requires the input to start at 00Z; first "
            f"timestamp is {first}."
        )


def daily_metrics(aorc_dataset, metric_names):
    """Reduce hourly AORC data to daily min, mean, and max for each metric.

    The reduction is `coarsen(time=24)`, a blockwise regrouping rather than a
    calendar-aware resample, so the input must start at 00Z and hold a whole
    number of 24-hour days -- checked explicitly here rather than assumed. The
    output time coordinate is left-labelled at midnight, matching what
    `resample(time="1D")` produced and what `daily_time_axis` and
    `write_block` expect: `coord_func="min"` takes each day's earliest
    timestamp rather than coarsen's default of averaging the window's
    coordinates into a spurious midday value. Each statistic is computed with
    `Coarsen.reduce` and one of `_QUIET_DAILY_REDUCERS` rather than the
    `.min()`/`.mean()`/`.max()` convenience methods, so that a day where every
    hour is masked NaN -- true for roughly half the domain, every day --
    resolves to NaN without dask's evaluation of `numpy.nanmin`/`nanmean`/
    `nanmax` emitting a "Mean of empty slice" / "All-NaN slice encountered"
    warning per block, once per day, across a 46-year run.

    Each metric also gets a `{metric}_valid_hours` companion (uint8, 0-24):
    the count of that metric's own hourly values that were non-NaN and
    contributed to the day's statistics. It is computed from the metric's own
    hourly series -- after the kernel, not from the raw inputs -- so it
    reflects the union of gaps across exactly that metric's own inputs;
    `temperature` (reads only air temperature) and `heat_index` (reads air
    temperature, humidity, and pressure) can disagree about which days are
    incomplete. A day whose valid-hour count falls below `MINIMUM_VALID_HOURS`
    has its min/mean/max suppressed to NaN -- but not its `valid_hours` count,
    which is deliberately reported even when the statistics it would have fed
    are withheld, so the store always records *why* a value is missing (no
    coverage at all vs. partial coverage below the threshold vs. genuinely
    never computed).

    :param aorc_dataset: Prepared AORC dataset, starting at 00Z and holding a
        whole number of 24-hour days
    :param metric_names: Names of metrics from `METRICS`
    :return: Lazy Dataset with `{metric}_{statistic}` and `{metric}_valid_hours`
        variables per requested metric. Metrics that were not requested are
        absent, not NaN-filled.
    :raises ValueError: If `aorc_dataset` does not start at 00Z or does not
        hold a whole number of 24-hour days
    """
    _validate_metric_names(metric_names)
    _validate_starts_on_midnight_whole_days(aorc_dataset)
    derived = _derived_inputs(aorc_dataset, metric_names)

    daily = {}
    for name in metric_names:
        hourly = METRICS[name].compute(derived)
        grouped = hourly.coarsen(time=24, coord_func="min")

        valid_hours = grouped.reduce(_count_valid_hours_reducer).astype(np.uint8)
        valid_hours.attrs = _valid_hours_attributes(name)
        daily[f"{name}_valid_hours"] = valid_hours

        has_enough_coverage = valid_hours >= MINIMUM_VALID_HOURS
        for statistic in DAILY_STATISTICS:
            reduced = grouped.reduce(_QUIET_DAILY_REDUCERS[statistic]).astype(np.float32)
            # Cells already NaN (out of region, or every hour of the metric's
            # own inputs missing) stay NaN here regardless -- this only ever
            # turns a *computed* value back into NaN for a day that fell short
            # of MINIMUM_VALID_HOURS.
            reduced = reduced.where(has_enough_coverage)
            reduced.attrs = _variable_attributes(name, statistic)
            daily[f"{name}_{statistic}"] = reduced

    return xr.Dataset(daily, attrs=GLOBAL_ATTRIBUTES)


OUTPUT_STORE_NAME = "aorc_heat_metrics.zarr"

#: Output time chunk, in days. Alongside the native (128, 256) spatial chunking
#: this gives 365 * 128 * 256 * 4 B = ~48 MB chunks.
#:
#: The spatial half of that shape is inherited, not chosen: the region crop is
#: snapped to the source store's chunk grid so that no rechunk is needed on
#: either read or write (see the module docstring), and the output store is
#: allocated to match. The time chunk is then what makes the resulting chunk a
#: reasonable size rather than a tiny one.
#:
#: **This is a spatially-oriented layout, and reading a long time series at a
#: single point is expensive under it.** One chunk holds a year for a whole
#: 128x256 tile, so a 46-year record at one grid cell means decompressing 47
#: chunks -- about 2.2 GB -- to extract 67 kB, and 48 MB is itself above the
#: 1-16 MB zarr generally wants. An earlier version of this comment claimed the
#: opposite, that the shape was "sized for reading long time series at a point";
#: it was not, and it never has been. Reading whole-domain maps, which is what
#: the layout genuinely suits, costs the same ~2.4 GB for every cell of a day
#: rather than for one.
#:
#: Improving point extraction means either splitting the output spatially
#: (365 x 32 x 64 gives 3 MB chunks and cuts a point series to ~140 MB, at no
#: cost to map reads, and still writes incrementally) or building a separate
#: store chunked along time in a post-pass. Both were considered and
#: deliberately not done; the layout is left as-is.
#:
#: This cannot divide the year boundaries: years alternate 365 and 366 days, so
#: no fixed chunk size lines up with every year's start. Each yearly write
#: therefore lands partially inside the chunks at its two ends. That is safe
#: here, but only because of how `write_block` and `run` are arranged -- see the
#: contract on `write_block` before changing either.
OUTPUT_TIME_CHUNK_DAYS = 365

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


#: Zarr group attribute holding, per metric, the sorted list of calendar years
#: that have actually been written (as opposed to merely allocated as NaN).
#: See `record_completed_years` and `pending_metrics`.
COMPLETED_YEARS_ATTR = "completed_years"


def _as_comparable_dates(time_index):
    """Reduce a time index to plain (year, month, day) tuples for comparison.

    `daily_time_axis` builds a `CFTimeIndex` of cftime objects, but xarray
    silently decodes a "standard"-calendar time coordinate read back from zarr
    into a plain `DatetimeIndex` of `numpy.datetime64` instead (it is
    representable as one, so xarray prefers it), which has no `.year`
    attribute. Normalising both flavours to plain (year, month, day) tuples
    makes an existing store's time coordinate and a freshly built time axis
    comparable regardless of which representation either one arrived in.

    :param time_index: A time index (e.g. `Dataset.indexes["time"]`, or a
        `CFTimeIndex` from `daily_time_axis`)
    :return: List of (year, month, day) tuples
    """
    dates = []
    for timestamp in np.asarray(time_index):
        if isinstance(timestamp, np.datetime64):
            timestamp = pd.Timestamp(timestamp)
        dates.append((timestamp.year, timestamp.month, timestamp.day))
    return dates


def _validate_time_axis(existing, time_axis, store_path):
    """Raise unless the store's time axis is exactly the requested one.

    Comparing only `.size` lets two different ranges of equal length -- for
    example 1999-2000 and 2000-2001 -- pass as equivalent, which then lets
    `initialize_output_store` silently relabel the store's existing time
    coordinate out from under data already written for a different metric.
    Comparing the actual calendar dates closes that gap.

    :param existing: The already-open existing store's Dataset
    :param time_axis: The time axis the caller is requesting
    :param store_path: Path to the store, for the error message
    """
    existing_time = existing.indexes["time"]
    if existing_time.size == time_axis.size and np.array_equal(
        _as_comparable_dates(existing_time), _as_comparable_dates(time_axis)
    ):
        return

    def _span(index):
        first, last = index[0], index[-1]
        return (
            f"{first.year:04d}-{first.month:02d}-{first.day:02d} to "
            f"{last.year:04d}-{last.month:02d}-{last.day:02d} ({index.size} days)"
        )

    raise ValueError(
        f"Existing store at {store_path} spans {_span(existing_time)}, "
        f"which does not match the {_span(time_axis)} requested. "
        "Delete the store or request the range it already covers."
    )


def _read_completed_years(store_path):
    """The `COMPLETED_YEARS_ATTR` attribute from a store, or an empty mapping.

    :param store_path: Path to the output zarr store
    :return: Mapping of metric name to the set of years recorded as written
    """
    group = zarr.open_group(str(store_path), mode="r")
    raw = group.attrs.get(COMPLETED_YEARS_ATTR, {})
    return {name: set(years) for name, years in raw.items()}


def record_completed_years(store_path, metric_names, years):
    """Record that `years` were actually written for `metric_names`.

    `initialize_output_store` allocates a metric's variables, NaN-filled,
    before any year is computed. Without an explicit completion record, a
    crash between allocation and the first successful write leaves a store
    that looks finished -- the variables exist -- but holds no real data, and
    a subsequent run would silently report nothing to do. Recording progress
    explicitly, per year, closes that gap: a metric is only ever reported as
    done when every requested year has actually been written.

    :param store_path: Path to the output zarr store
    :param metric_names: Metrics that were just written for `years`
    :param years: Calendar years that were successfully written
    """
    store_path = Path(store_path)
    group = zarr.open_group(str(store_path), mode="a")
    completed = {name: set(existing) for name, existing in group.attrs.get(COMPLETED_YEARS_ATTR, {}).items()}
    for name in metric_names:
        completed.setdefault(name, set()).update(years)
    group.attrs[COMPLETED_YEARS_ATTR] = {
        name: sorted(recorded_years) for name, recorded_years in completed.items()
    }
    # Attribute writes go through the plain (unconsolidated) metadata; without
    # re-consolidating, `xr.open_zarr(..., consolidated=True)` would keep
    # reading the stale, pre-update attributes.
    zarr.consolidate_metadata(str(store_path))


def pending_metrics(store_path, metric_names, time_axis):
    """Metrics that still need computing, after validating the existing store.

    A metric counts as done only when its `_mean` variable exists AND its
    recorded completed-years (see `record_completed_years`) cover every year
    in `time_axis`. Variables existing alone is not enough: they are allocated
    NaN-filled up front, before any year is actually computed, so a metric
    whose completion record is missing or incomplete is reported as pending
    so the run redoes it -- silently reporting success on all-NaN data would
    be worse than repeating the work.

    :param store_path: Path to the output zarr store
    :param metric_names: Requested metric names
    :param time_axis: Expected daily time axis
    :return: The subset of `metric_names` not already fully in the store
    """
    _validate_metric_names(metric_names)
    store_path = Path(store_path)
    if not store_path.exists():
        return list(metric_names)

    existing = xr.open_zarr(store_path, consolidated=True)
    _validate_time_axis(existing, time_axis, store_path)

    required_years = {timestamp.year for timestamp in np.asarray(time_axis)}
    completed_years = _read_completed_years(store_path)

    pending = []
    for name in metric_names:
        if f"{name}_mean" not in existing.data_vars:
            pending.append(name)
        elif not required_years.issubset(completed_years.get(name, set())):
            pending.append(name)
    return pending


def initialize_output_store(store_path, metric_names, time_axis, template):
    """Allocate NaN-filled variables for `metric_names` without computing anything.

    Creates the store when absent, otherwise appends the new variables to it.
    The template supplies the spatial coordinates and chunk sizes.

    A metric already reported by `pending_metrics` can still have its
    variables already sitting in the store -- that is exactly the
    crash-then-rerun case `pending_metrics` exists to catch (variables
    allocated by an earlier, incomplete run, whose completed-years record is
    missing or partial). Re-declaring an already-existing zarr variable with
    fresh encoding is both unnecessary and rejected outright by xarray, so
    allocation here is limited to genuinely new variables; anything already
    present is left as-is and simply gets overwritten in place by the write
    loop that follows.

    :param store_path: Path to the output zarr store
    :param metric_names: Metrics to allocate
    :param time_axis: Full daily time axis
    :param template: One year of `daily_metrics` output, used for its grid
    """
    store_path = Path(store_path)
    # Taken from the template's own first chunk. That is only safe because the
    # region's slices are snapped to native chunk boundaries, so the first
    # chunk is a full one rather than a partial leading remainder.
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
        metric_names = [name for name in metric_names if f"{name}_mean" not in existing.data_vars]
        if not metric_names:
            return

    # Two placeholders, not one: the three daily statistics are float32 with a
    # NaN fill (so an unwritten day reads back as missing, not as a
    # plausible-looking value), but `valid_hours` is uint8 -- NaN has no uint8
    # representation, so it gets its own placeholder filled with 0 instead.
    # Keeping `DAILY_STATISTICS` itself float32-only (rather than smuggling
    # "valid_hours" into it) is what keeps this split honest; every site that
    # iterates `DAILY_STATISTICS` would otherwise need to special-case it.
    float_placeholder = dask.array.full(shape, np.nan, dtype=np.float32, chunks=chunks)
    valid_hours_placeholder = dask.array.full(shape, 0, dtype=np.uint8, chunks=chunks)

    variables = {
        f"{name}_{statistic}": (
            OUTPUT_DIMENSIONS,
            float_placeholder,
            _variable_attributes(name, statistic),
        )
        for name in metric_names
        for statistic in DAILY_STATISTICS
    }
    variables.update(
        {
            f"{name}_valid_hours": (
                OUTPUT_DIMENSIONS,
                valid_hours_placeholder,
                _valid_hours_attributes(name),
            )
            for name in metric_names
        }
    )
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
    #
    # `valid_hours` gets no `_FillValue` at all, deliberately: it is uint8, so
    # NaN is not representable, and 0 is already the right "never computed"
    # value -- indistinguishable, on purpose, from a day this run genuinely
    # found zero valid hours for. Declaring an explicit `_FillValue` here
    # would not just be redundant, it would actively break the dtype: xarray's
    # CF decoding treats any integer variable carrying a `_FillValue` as
    # needing mask-and-scale handling and widens it to int64 on read, even
    # though nothing here is ever masked (0 is real data, not a sentinel).
    # Leaving the attribute off keeps the round trip uint8 -- zarr's own
    # default fill already zero-fills unwritten chunks with no encoding needed.
    encoding = {
        f"{name}_{statistic}": {"_FillValue": np.float32("nan")}
        for name in metric_names
        for statistic in DAILY_STATISTICS
    }

    allocation.to_zarr(
        store_path,
        mode="a" if store_path.exists() else "w",
        compute=False,
        zarr_format=2,
        consolidated=True,
        encoding=encoding,
    )


def chunk_sizes_aligned_to_store(start, stop, chunk):
    """Dask chunk sizes for [start, stop) that never straddle a store boundary.

    A block written into the middle of the store generally starts mid-chunk,
    because year lengths alternate 365/366 and no fixed chunk size divides every
    year boundary. Splitting the block at the store's own boundaries means each
    resulting dask chunk falls inside exactly one store chunk, so no two tasks
    in a single write ever target the same chunk.

    :param start: First index of the block within the store's axis
    :param stop: One past the block's last index
    :param chunk: The store's chunk size along that axis
    :return: Tuple of dask chunk sizes summing to stop - start
    """
    sizes = []
    position = start
    while position < stop:
        next_boundary = min((position // chunk + 1) * chunk, stop)
        sizes.append(next_boundary - position)
        position = next_boundary
    return tuple(sizes)


def write_block(store_path, daily, time_axis, latitude_slice=None, longitude_slice=None):
    """Write a contiguous block of daily metrics into its region of the store.

    The block is converted to the store's calendar, then split along time at the
    store's chunk boundaries so no dask chunk spans two store chunks. Spatial
    chunks are left untouched: they came from the source store via a
    chunk-boundary-snapped crop, and the output store was allocated with those
    same sizes, so they already align. Coordinate variables are dropped because
    they already exist in the store.

    `latitude_slice` and `longitude_slice` name where the block sits in the
    store's spatial grid. They come from `mask.occupied_chunk_blocks`, so both
    edges land on store chunk boundaries and each block covers whole chunks.
    Omit them (the default) to write the full spatial extent, which is what a
    caller holding an entire bounding box wants.

    **Contract: callers must write blocks one at a time, never concurrently.**
    `safe_chunks=False` is required because a block's first and last chunks land
    partway into a store chunk *along time*, which xarray otherwise refuses.
    That refusal guards against a real hazard -- two tasks read-modify-writing
    one chunk and losing an update. Three things make it safe here, and all
    three must hold:

    1. Within a write, the time split above gives every dask chunk its own store
       chunk, so no two tasks in the same graph collide.
    2. Across writes, adjacent *years* do share the chunk straddling their
       boundary, and correctness rests on zarr read-modify-writing it. That is
       only sound while writes are sequential. Parallelising `run`'s year loop
       would silently corrupt one day at each year boundary.
    3. Across writes, two spatial blocks of the *same* year never share a chunk,
       because every block edge is a chunk boundary. This is why the spatial
       slices must come from `occupied_chunk_blocks` and not from an arbitrary
       crop -- an unaligned spatial block would reintroduce exactly the
       read-modify-write race that `safe_chunks=False` stops xarray catching.

    :param store_path: Path to the output zarr store
    :param daily: A contiguous block of `daily_metrics` output
    :param time_axis: Full daily time axis
    :param latitude_slice: Where the block sits on the store's latitude axis,
        or None for the full extent
    :param longitude_slice: Where the block sits on the store's longitude axis,
        or None for the full extent
    """
    aligned = daily.convert_calendar("standard", use_cftime=True)
    region = time_region(time_axis, aligned.indexes["time"])
    aligned = aligned.chunk(
        {"time": chunk_sizes_aligned_to_store(
            region.start, region.stop, OUTPUT_TIME_CHUNK_DAYS
        )}
    )
    aligned = aligned.drop_vars(list(aligned.coords))

    store_region = {"time": region}
    if latitude_slice is not None:
        store_region["latitude"] = latitude_slice
    if longitude_slice is not None:
        store_region["longitude"] = longitude_slice
    aligned.to_zarr(Path(store_path), region=store_region, safe_chunks=False)


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

    Each year's block loop makes one extra pass over that block's
    `valid_hours` variables, purely to log source-gap coverage (see the
    `[compute]` line below): `write_block` already computes them once to
    write the store, and logging re-computes them a second time from the same
    lazy graph to summarize before moving on, rather than plumbing the
    already-computed values back out of `to_zarr`. Cheap relative to the
    kernel evaluation and I/O it sits next to, but real.
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
    alignment = native_spatial_chunks(sample)
    region = mask_module.texas_region(
        sample.latitude.values, sample.longitude.values, alignment=alignment
    )
    _log(
        f"[init] Native spatial chunks {alignment}; region cropped to "
        f"latitude[{region.latitude_slice.start}:{region.latitude_slice.stop}] "
        f"longitude[{region.longitude_slice.start}:{region.longitude_slice.stop}]."
    )

    variables = required_variables(pending)
    first_year = open_aorc_year(start_year, region, variables, filesystem)
    # The template spans the whole bounding box on purpose: the store's shape
    # and spatial coordinates are those of the full box, and only the *writes*
    # are restricted to occupied blocks. Chunks no block ever touches keep the
    # NaN this allocation gives them, which is the right answer for them.
    initialize_output_store(store_path, pending, time_axis, daily_metrics(first_year, pending))
    # +1 per metric for its `_valid_hours` companion, alongside the three
    # float32 statistics in DAILY_STATISTICS.
    _log(
        f"[init] Allocated {len(pending) * (len(DAILY_STATISTICS) + 1)} variables "
        f"in {store_path}."
    )

    blocks = mask_module.occupied_chunk_blocks(region.mask, alignment)
    total_chunks = -(-region.mask.sizes["latitude"] // alignment[0]) * -(
        -region.mask.sizes["longitude"] // alignment[1]
    )
    covered = sum(
        -(-(block[0].stop - block[0].start) // alignment[0])
        * -(-(block[1].stop - block[1].start) // alignment[1])
        for block in blocks
    )
    _log(
        f"[init] {len(blocks)} occupied blocks covering {covered} of {total_chunks} "
        f"chunks; skipping {total_chunks - covered} that hold no in-region cells."
    )

    for year in range(start_year, end_year + 1):
        _log(f"[compute] Deriving metrics for {year}.")
        dataset = open_aorc_year(year, region, variables, filesystem)
        # In-region cell-days summed across every pending metric's own
        # `valid_hours` -- a cell-day incomplete for two requested metrics is
        # counted twice, because the two metrics can disagree about which
        # days are incomplete (see `daily_metrics`'s docstring). 24 here is
        # literal hour-of-day completeness, not `MINIMUM_VALID_HOURS`: this
        # log reports actual source coverage regardless of how lenient the
        # suppression threshold is configured to be.
        incomplete_cell_days = 0
        fully_missing_cell_days = 0
        # One graph per block rather than one per year. The year is lazy, so
        # selecting a block reads only that block's chunks -- the empty ones are
        # never fetched, decompressed, or reduced at all.
        for index, (latitude_slice, longitude_slice) in enumerate(blocks, start=1):
            block = dataset.isel(latitude=latitude_slice, longitude=longitude_slice)
            block_mask = region.mask.isel(
                latitude=latitude_slice, longitude=longitude_slice
            ).values
            daily = daily_metrics(block, pending)
            write_block(
                store_path,
                daily,
                time_axis,
                latitude_slice=latitude_slice,
                longitude_slice=longitude_slice,
            )
            valid_hours = daily[[f"{name}_valid_hours" for name in pending]].compute()
            for name in pending:
                in_region_hours = valid_hours[f"{name}_valid_hours"].values[:, block_mask]
                incomplete_cell_days += int(np.count_nonzero(in_region_hours < 24))
                fully_missing_cell_days += int(np.count_nonzero(in_region_hours == 0))
            _log(f"[compute] {year}: wrote block {index}/{len(blocks)}.")
        _log(
            f"[compute] {year}: {incomplete_cell_days} in-region cell-days incomplete "
            f"(valid_hours<24), {fully_missing_cell_days} fully missing (valid_hours==0), "
            f"summed across {len(pending)} metric(s)."
        )
        # Recorded only after every block of the year has been written, so a
        # crash partway through leaves the year unrecorded and `pending_metrics`
        # correctly reports the metric as still pending on the next run.
        record_completed_years(store_path, pending, [year])
        _log(f"[compute] Wrote {year}.")

    _log("[final] Finished.")
    return store_path
