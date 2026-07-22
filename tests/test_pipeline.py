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


def test_prepare_dataset_imposes_its_own_spatial_chunking(synthetic_aorc_dataset):
    """`prepare_dataset` must not inherit the source store's spatial chunking.

    The 2x2 synthetic grid is far smaller than `SPATIAL_CHUNK_SIZE`, so dask
    collapses it to a single spatial chunk either way; this only checks that
    the dimension is chunked at all (rather than left as one native chunk
    that happens to also be size 2). The real, size-dependent behaviour is
    covered by `test_write_block_succeeds_when_the_crop_does_not_start_on_a_chunk_boundary`.
    """
    from aorc_heat.mask import Region

    mask = xr.DataArray(
        np.ones((2, 2), dtype=bool),
        dims=("latitude", "longitude"),
        coords={
            "latitude": synthetic_aorc_dataset.latitude,
            "longitude": synthetic_aorc_dataset.longitude,
        },
    )
    region = Region(mask=mask, latitude_slice=slice(0, 2), longitude_slice=slice(0, 2))

    prepared = pipeline.prepare_dataset(synthetic_aorc_dataset, region)

    assert prepared.chunksizes["latitude"] == (2,)
    assert prepared.chunksizes["longitude"] == (2,)


def test_write_block_succeeds_when_the_crop_does_not_start_on_a_chunk_boundary(tmp_path):
    """A region write must succeed when the crop starts mid native-chunk.

    Mirrors the real Texas bounding box crop, whose start indices (701, 2803
    into the AORC domain) are prime and so never land on a native chunk
    boundary no matter what chunk size the source store used. Before the fix,
    `prepare_dataset` inherited that native spatial chunking, so
    `initialize_output_store` derived a declared zarr chunk size from a
    partial first dask chunk, and `write_block` handed xarray dask chunks that
    didn't match it -- raising a "would overlap multiple Dask chunks" error on
    the very first write. This builds a synthetic "source store" natively
    chunked at sizes that don't divide the crop's start index, to reproduce
    that misalignment, and asserts the write completes instead.
    """
    from aorc_heat.mask import Region

    domain_size = 320
    crop_length = 290  # bigger than SPATIAL_CHUNK_SIZE, so the crop spans a
    # full 256 chunk plus a smaller final one -- proving a partial *last*
    # chunk is fine while a partial *first* chunk (what this test guards
    # against) is not.
    latitude_start, longitude_start = 5, 11  # neither divides the native chunk sizes below

    full_latitudes = np.arange(domain_size, dtype=np.float64)
    full_longitudes = np.arange(domain_size, dtype=np.float64)
    hours = 48
    times = xr.date_range("2000-07-01", periods=hours, freq="h", use_cftime=True, calendar="standard")
    shape = (hours, domain_size, domain_size)
    generator = np.random.default_rng(seed=99)

    def field(low, high):
        return generator.uniform(low, high, size=shape).astype(np.float32)

    full_domain = xr.Dataset(
        {
            "TMP_2maboveground": (("time", "latitude", "longitude"), field(295.0, 315.0)),
            "SPFH_2maboveground": (("time", "latitude", "longitude"), field(0.005, 0.018)),
            "PRES_surface": (("time", "latitude", "longitude"), field(98000.0, 101500.0)),
        },
        coords={"time": times, "latitude": full_latitudes, "longitude": full_longitudes},
    ).chunk({"time": 24, "latitude": 41, "longitude": 43})  # native chunk sizes, both prime

    latitude_slice = slice(latitude_start, latitude_start + crop_length)
    longitude_slice = slice(longitude_start, longitude_start + crop_length)
    mask = xr.DataArray(
        np.ones((crop_length, crop_length), dtype=bool),
        dims=("latitude", "longitude"),
        coords={
            "latitude": full_domain.latitude.isel(latitude=latitude_slice),
            "longitude": full_domain.longitude.isel(longitude=longitude_slice),
        },
    )
    region = Region(mask=mask, latitude_slice=latitude_slice, longitude_slice=longitude_slice)

    prepared = pipeline.prepare_dataset(full_domain, region)
    daily = pipeline.daily_metrics(prepared, ["humidex"])

    store_path = tmp_path / pipeline.OUTPUT_STORE_NAME
    axis = pipeline.daily_time_axis(2000, 2000)
    pipeline.initialize_output_store(store_path, ["humidex"], axis, daily)

    # This must not raise the zarr "would overlap multiple Dask chunks" error.
    pipeline.write_block(store_path, daily, axis)

    written = xr.open_zarr(store_path, consolidated=True).compute()
    # The synthetic data covers 1-2 July 2000: days 182 and 183 of a leap year.
    assert not bool(np.isnan(written["humidex_mean"].isel(time=slice(182, 184))).any())
    assert bool(np.isnan(written["humidex_mean"].isel(time=0)).all())


# ---------------------------------------------------------------------------
# Output store
# ---------------------------------------------------------------------------
def test_daily_time_axis_spans_whole_years_and_handles_leap_days():
    axis = pipeline.daily_time_axis(1999, 2000)
    assert axis.size == 365 + 366
    assert axis[0].year == 1999 and axis[0].month == 1 and axis[0].day == 1
    assert axis[-1].year == 2000 and axis[-1].month == 12 and axis[-1].day == 31


def test_time_region_locates_a_block_within_the_axis():
    axis = pipeline.daily_time_axis(1999, 2000)
    assert pipeline.time_region(axis, axis[:365]) == slice(0, 365)
    assert pipeline.time_region(axis, axis[365:]) == slice(365, 731)
    assert pipeline.time_region(axis, axis[10:12]) == slice(10, 12)


def test_time_region_rejects_a_block_outside_the_axis():
    axis = pipeline.daily_time_axis(2000, 2000)
    outside = pipeline.daily_time_axis(1990, 1990)[:5]
    with pytest.raises(KeyError):
        pipeline.time_region(axis, outside)


def _template_for(dataset, metric_names):
    return pipeline.daily_metrics(dataset, metric_names)


def test_initialize_output_store_creates_requested_variables(tmp_path, synthetic_aorc_dataset):
    store_path = tmp_path / pipeline.OUTPUT_STORE_NAME
    axis = pipeline.daily_time_axis(2000, 2000)
    template = _template_for(synthetic_aorc_dataset, ["humidex"])

    pipeline.initialize_output_store(store_path, ["humidex"], axis, template)

    written = xr.open_zarr(store_path, consolidated=True)
    assert sorted(written.data_vars) == ["humidex_max", "humidex_mean", "humidex_min"]
    assert written.sizes["time"] == 366
    assert written.chunksizes["time"][0] == 1


def test_initialize_output_store_appends_new_metrics(tmp_path, synthetic_aorc_dataset):
    store_path = tmp_path / pipeline.OUTPUT_STORE_NAME
    axis = pipeline.daily_time_axis(2000, 2000)

    pipeline.initialize_output_store(
        store_path, ["humidex"], axis, _template_for(synthetic_aorc_dataset, ["humidex"])
    )
    pipeline.initialize_output_store(
        store_path,
        ["simplified_wbgt"],
        axis,
        _template_for(synthetic_aorc_dataset, ["simplified_wbgt"]),
    )

    written = xr.open_zarr(store_path, consolidated=True)
    assert "humidex_mean" in written.data_vars
    assert "simplified_wbgt_mean" in written.data_vars


def test_pending_metrics_skips_what_is_already_present(tmp_path, synthetic_aorc_dataset):
    store_path = tmp_path / pipeline.OUTPUT_STORE_NAME
    axis = pipeline.daily_time_axis(2000, 2000)

    assert pipeline.pending_metrics(store_path, ["humidex", "heat_index"], axis) == [
        "humidex",
        "heat_index",
    ]

    template = _template_for(synthetic_aorc_dataset, ["humidex"])
    pipeline.initialize_output_store(store_path, ["humidex"], axis, template)
    pipeline.write_block(store_path, template, axis)
    pipeline.record_completed_years(store_path, ["humidex"], [2000])
    assert pipeline.pending_metrics(store_path, ["humidex", "heat_index"], axis) == ["heat_index"]


def test_pending_metrics_stays_pending_until_every_requested_year_is_recorded(
    tmp_path, synthetic_aorc_dataset
):
    """Allocating a metric's variables must not, by itself, mark it done.

    `initialize_output_store` allocates NaN-filled variables before any year is
    computed. If a crash happens right after that -- which is exactly what
    Finding 1 does on the very first region write -- re-running must recompute
    rather than report "nothing to compute" against all-NaN data.
    """
    store_path = tmp_path / pipeline.OUTPUT_STORE_NAME
    axis = pipeline.daily_time_axis(1999, 2000)
    template = _template_for(synthetic_aorc_dataset, ["humidex"])

    pipeline.initialize_output_store(store_path, ["humidex"], axis, template)
    # Variables exist now, but nothing has actually been written or recorded.
    assert pipeline.pending_metrics(store_path, ["humidex"], axis) == ["humidex"]

    pipeline.write_block(store_path, template, axis)
    pipeline.record_completed_years(store_path, ["humidex"], [2000])
    # Only one of the two requested years (1999, 2000) is recorded complete.
    assert pipeline.pending_metrics(store_path, ["humidex"], axis) == ["humidex"]

    pipeline.record_completed_years(store_path, ["humidex"], [1999])
    # Both years recorded now: the metric is finally done.
    assert pipeline.pending_metrics(store_path, ["humidex"], axis) == []


def test_pending_metrics_rejects_time_axis_mismatch(tmp_path, synthetic_aorc_dataset):
    store_path = tmp_path / pipeline.OUTPUT_STORE_NAME
    pipeline.initialize_output_store(
        store_path,
        ["humidex"],
        pipeline.daily_time_axis(2000, 2000),
        _template_for(synthetic_aorc_dataset, ["humidex"]),
    )

    with pytest.raises(ValueError, match="does not match"):
        pipeline.pending_metrics(store_path, ["heat_index"], pipeline.daily_time_axis(1999, 2001))


def test_pending_metrics_rejects_a_shifted_range_of_the_same_length(tmp_path, synthetic_aorc_dataset):
    """Two different ranges can have identical day counts and must not be confused.

    1999-2000 spans 365 + 366 = 731 days; 2000-2001 spans 366 + 365 = 731 days
    too. A length-only check lets the second silently pass validation against a
    store built for the first, which then lets `initialize_output_store`
    relabel the existing time coordinate -- so already-written data for one
    date would be reported back under a different date.
    """
    store_path = tmp_path / pipeline.OUTPUT_STORE_NAME
    original_axis = pipeline.daily_time_axis(1999, 2000)
    shifted_axis = pipeline.daily_time_axis(2000, 2001)
    assert original_axis.size == shifted_axis.size

    pipeline.initialize_output_store(
        store_path,
        ["humidex"],
        original_axis,
        _template_for(synthetic_aorc_dataset, ["humidex"]),
    )

    with pytest.raises(ValueError, match="does not match"):
        pipeline.pending_metrics(store_path, ["heat_index"], shifted_axis)


def test_unwritten_days_read_back_as_nan(tmp_path, synthetic_aorc_dataset):
    store_path = tmp_path / pipeline.OUTPUT_STORE_NAME
    axis = pipeline.daily_time_axis(2000, 2000)
    template = _template_for(synthetic_aorc_dataset, ["humidex"])

    pipeline.initialize_output_store(store_path, ["humidex"], axis, template)

    written = xr.open_zarr(store_path, consolidated=True)
    # Nothing has been written yet, so every day must read as NaN rather than 0.
    assert bool(np.isnan(written["humidex_mean"].isel(time=0)).all())


def test_write_block_fills_only_its_own_days(tmp_path, synthetic_aorc_dataset):
    store_path = tmp_path / pipeline.OUTPUT_STORE_NAME
    axis = pipeline.daily_time_axis(2000, 2000)
    daily = _template_for(synthetic_aorc_dataset, ["humidex"])

    pipeline.initialize_output_store(store_path, ["humidex"], axis, daily)
    pipeline.write_block(store_path, daily, axis)

    written = xr.open_zarr(store_path, consolidated=True).compute()
    # The synthetic dataset covers 1-2 July 2000: days 182 and 183 of a leap year.
    assert not bool(np.isnan(written["humidex_mean"].isel(time=slice(182, 184))).any())
    assert bool(np.isnan(written["humidex_mean"].isel(time=0)).all())
    assert bool(np.isnan(written["humidex_mean"].isel(time=365)).all())


def test_adding_a_metric_leaves_earlier_values_untouched(tmp_path, synthetic_aorc_dataset):
    store_path = tmp_path / pipeline.OUTPUT_STORE_NAME
    axis = pipeline.daily_time_axis(2000, 2000)

    first = _template_for(synthetic_aorc_dataset, ["humidex"])
    pipeline.initialize_output_store(store_path, ["humidex"], axis, first)
    pipeline.write_block(store_path, first, axis)
    before = xr.open_zarr(store_path, consolidated=True)["humidex_mean"].compute()

    second = _template_for(synthetic_aorc_dataset, ["simplified_wbgt"])
    pipeline.initialize_output_store(store_path, ["simplified_wbgt"], axis, second)
    pipeline.write_block(store_path, second, axis)

    after = xr.open_zarr(store_path, consolidated=True)
    np.testing.assert_array_equal(before.values, after["humidex_mean"].compute().values)
    assert "simplified_wbgt_mean" in after.data_vars
