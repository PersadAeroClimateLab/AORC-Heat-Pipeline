"""Derive daily heat metrics from the NOAA AORC hourly zarr archive.

One year is processed at a time. Each raw AORC variable is read exactly once and
the intermediate quantities built from it -- air temperature in Celsius, vapour
pressure, and so on -- are shared by every metric that needs them, so the four
metrics that depend on vapour pressure resolve to a single node in the dask
graph rather than four.

Time chunking is left at AORC's native 144 hours. That is exactly six days, and
year files start at midnight on 1 January, so every native chunk boundary falls
on a midnight and each chunk holds six whole days. A daily `resample` group
therefore never crosses a chunk boundary -- the property the reduction needs --
without any rechunk. Splitting to 24 hours instead (as an earlier version did)
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


def _variable_attributes(metric_name, statistic):
    """CF-style attributes for one output variable.

    Units come from the metric's registry entry: most metrics are degrees
    Fahrenheit, but `relative_humidity` is a percent.
    """
    readable = metric_name.replace("_", " ")
    return {
        "units": METRICS[metric_name].units,
        "source_timestep": "hourly",
        "description": f"Daily {statistic} of hourly {readable} taken across 24 hours",
    }


# --------------------------------------------------------------------------
# Shared intermediates
# --------------------------------------------------------------------------
def _derived_inputs(aorc_dataset, metric_names):
    """Build the intermediate quantities the requested metrics need.

    Each quantity is built once and handed to every metric that uses it. Air
    temperature, pressure, humidity, and vapour pressure are unconditional
    because most metrics depend on them; relative humidity and wind speed are
    built only when a metric that needs them was requested.

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

    if "heat_index" in metric_names or "relative_humidity" in metric_names:
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
    six whole, midnight-aligned days, so a daily `resample` group never crosses
    a chunk boundary -- splitting to 24 hours would only inflate the task graph
    (~6x, measured) for identical output. Spatial chunking is likewise native:
    `region`'s slices are snapped to chunk boundaries, so the cropped array
    lines up with both the source store and the output store as-is.

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


OUTPUT_STORE_NAME = "aorc_heat_metrics.zarr"

#: Output time chunk, in days, giving ~48 MB chunks alongside the native
#: (128, 256) spatial chunking. Sized for reading long time series at a point,
#: the dominant access pattern for trend work.
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


def write_block(store_path, daily, time_axis):
    """Write a contiguous block of daily metrics into its region of the store.

    The block is converted to the store's calendar, then split along time at the
    store's chunk boundaries so no dask chunk spans two store chunks. Spatial
    chunks are left untouched: they came from the source store via a
    chunk-boundary-snapped crop, and the output store was allocated with those
    same sizes, so they already align. Coordinate variables are dropped because
    they already exist in the store.

    **Contract: callers must write blocks one at a time, never concurrently.**
    `safe_chunks=False` is required because a block's first and last chunks land
    partway into a store chunk, which xarray otherwise refuses. That refusal
    guards against a real hazard -- two tasks read-modify-writing one chunk and
    losing an update. Two things make it safe here, and both must hold:

    1. Within a write, the split above gives every dask chunk its own store
       chunk, so no two tasks in the same graph collide.
    2. Across writes, adjacent blocks do share the chunk straddling their
       boundary, and correctness rests on zarr read-modify-writing it. That is
       only sound while writes are sequential. Parallelising `run`'s year loop
       would silently corrupt one day at each year boundary.

    :param store_path: Path to the output zarr store
    :param daily: A contiguous block of `daily_metrics` output
    :param time_axis: Full daily time axis
    """
    aligned = daily.convert_calendar("standard", use_cftime=True)
    region = time_region(time_axis, aligned.indexes["time"])
    aligned = aligned.chunk(
        {"time": chunk_sizes_aligned_to_store(
            region.start, region.stop, OUTPUT_TIME_CHUNK_DAYS
        )}
    )
    aligned = aligned.drop_vars(list(aligned.coords))
    aligned.to_zarr(Path(store_path), region={"time": region}, safe_chunks=False)


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
    initialize_output_store(store_path, pending, time_axis, daily_metrics(first_year, pending))
    _log(f"[init] Allocated {len(pending) * len(DAILY_STATISTICS)} variables in {store_path}.")

    for year in range(start_year, end_year + 1):
        _log(f"[compute] Deriving metrics for {year}.")
        dataset = open_aorc_year(year, region, variables, filesystem)
        write_block(store_path, daily_metrics(dataset, pending), time_axis)
        # Recorded only after the write above actually succeeds, so a crash
        # mid-write leaves the year unrecorded and `pending_metrics` correctly
        # reports the metric as still pending on the next run.
        record_completed_years(store_path, pending, [year])
        _log(f"[compute] Wrote {year}.")

    _log("[final] Finished.")
    return store_path
