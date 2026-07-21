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
