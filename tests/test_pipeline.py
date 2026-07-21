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
            result["heat_index_min"].isel(time=day).values, window.min(axis=0), rtol=1e-5
        )
        np.testing.assert_allclose(
            result["heat_index_mean"].isel(time=day).values, window.mean(axis=0), rtol=1e-5
        )
        np.testing.assert_allclose(
            result["heat_index_max"].isel(time=day).values, window.max(axis=0), rtol=1e-5
        )


def test_daily_metrics_apparent_temperature_matches_manual_numpy(synthetic_aorc_dataset):
    result = pipeline.daily_metrics(synthetic_aorc_dataset, ["apparent_temperature"]).compute()

    loaded = synthetic_aorc_dataset.compute()
    air_temperature_celsius = core.kelvin_to_celsius(loaded["TMP_2maboveground"].values)
    air_pressure_hpa = (loaded["PRES_surface"].values / 100.0).astype(np.float32)
    vapor = core.vapor_pressure(loaded["SPFH_2maboveground"].values, air_pressure_hpa)
    wind_speed = np.sqrt(
        loaded["UGRD_10maboveground"].values ** 2 + loaded["VGRD_10maboveground"].values ** 2
    )
    hourly_celsius = core.apparent_temperature(air_temperature_celsius, vapor, wind_speed)
    hourly = core.celsius_to_fahrenheit(hourly_celsius)

    for day in (0, 1):
        window = hourly[day * 24 : (day + 1) * 24]
        np.testing.assert_allclose(
            result["apparent_temperature_min"].isel(time=day).values, window.min(axis=0), rtol=1e-5
        )
        np.testing.assert_allclose(
            result["apparent_temperature_mean"].isel(time=day).values, window.mean(axis=0), rtol=1e-5
        )
        np.testing.assert_allclose(
            result["apparent_temperature_max"].isel(time=day).values, window.max(axis=0), rtol=1e-5
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
