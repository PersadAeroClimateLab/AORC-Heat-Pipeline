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
from datetime import datetime
from pathlib import Path
from typing import Callable

import dask.array
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
