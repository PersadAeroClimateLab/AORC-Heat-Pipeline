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
    "temperature",
    "relative_humidity",
    "specific_humidity",
]


def test_registry_covers_every_in_scope_metric():
    assert sorted(pipeline.METRICS) == sorted(ALL_METRICS)


def test_required_variables_only_reads_wind_for_apparent_temperature():
    without_wind = pipeline.required_variables(["humidex", "wet_bulb_temperature"])
    assert pipeline.EASTWARD_WIND not in without_wind
    assert pipeline.NORTHWARD_WIND not in without_wind

    with_wind = pipeline.required_variables(["apparent_temperature"])
    assert pipeline.EASTWARD_WIND in with_wind
    assert pipeline.NORTHWARD_WIND in with_wind


def test_required_variables_matches_each_metric_declared_inputs():
    # required_variables is exactly the metric's declared inputs -- the contract
    # `_derived_inputs` relies on to know which intermediates it can build.
    for name in ALL_METRICS:
        assert pipeline.required_variables([name]) == sorted(pipeline.METRICS[name].inputs)


def test_single_input_metrics_read_only_their_own_variable():
    # `temperature` and `specific_humidity` are raw variables carried through
    # the daily reduction, so each reads exactly one AORC variable -- neither
    # pulls the others off S3.
    assert pipeline.required_variables(["temperature"]) == [pipeline.AIR_TEMPERATURE]
    assert pipeline.required_variables(["specific_humidity"]) == [pipeline.SPECIFIC_HUMIDITY]


def test_derived_metrics_read_humidity_and_pressure():
    # Every metric that derives vapour pressure needs both specific humidity and
    # surface pressure; that is all of them except the two raw-passthrough ones.
    passthrough = {"temperature", "specific_humidity"}
    for name in ALL_METRICS:
        if name in passthrough:
            continue
        variables = pipeline.required_variables([name])
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


def test_daily_metrics_temperature_matches_manual_numpy(synthetic_aorc_dataset):
    result = pipeline.daily_metrics(synthetic_aorc_dataset, ["temperature"]).compute()

    loaded = synthetic_aorc_dataset.compute()
    # Kelvin -> Celsius -> Fahrenheit, exactly once. A missing or doubled
    # conversion changes the numbers here.
    hourly = core.celsius_to_fahrenheit(core.kelvin_to_celsius(loaded["TMP_2maboveground"].values))

    for day in (0, 1):
        window = hourly[day * 24 : (day + 1) * 24]
        np.testing.assert_allclose(
            result["temperature_min"].isel(time=day).values, window.min(axis=0), rtol=1e-5
        )
        np.testing.assert_allclose(
            result["temperature_mean"].isel(time=day).values, window.mean(axis=0), rtol=1e-5
        )
        np.testing.assert_allclose(
            result["temperature_max"].isel(time=day).values, window.max(axis=0), rtol=1e-5
        )


def test_daily_metrics_relative_humidity_matches_manual_numpy(synthetic_aorc_dataset):
    result = pipeline.daily_metrics(synthetic_aorc_dataset, ["relative_humidity"]).compute()

    loaded = synthetic_aorc_dataset.compute()
    air_temperature_celsius = core.kelvin_to_celsius(loaded["TMP_2maboveground"].values)
    air_pressure_hpa = (loaded["PRES_surface"].values / 100.0).astype(np.float32)
    vapor = core.vapor_pressure(loaded["SPFH_2maboveground"].values, air_pressure_hpa)
    saturation = core.saturation_vapor_pressure(air_temperature_celsius)
    # Output is a percent, carried through with no Fahrenheit conversion.
    hourly = core.relative_humidity(vapor, saturation)

    for day in (0, 1):
        window = hourly[day * 24 : (day + 1) * 24]
        np.testing.assert_allclose(
            result["relative_humidity_min"].isel(time=day).values, window.min(axis=0), rtol=1e-5
        )
        np.testing.assert_allclose(
            result["relative_humidity_mean"].isel(time=day).values, window.mean(axis=0), rtol=1e-5
        )
        np.testing.assert_allclose(
            result["relative_humidity_max"].isel(time=day).values, window.max(axis=0), rtol=1e-5
        )


def test_relative_humidity_is_a_percent_not_fahrenheit(synthetic_aorc_dataset):
    result = pipeline.daily_metrics(synthetic_aorc_dataset, ["relative_humidity"])
    assert result["relative_humidity_mean"].attrs["units"] == "percent"


def test_temperature_is_fahrenheit(synthetic_aorc_dataset):
    result = pipeline.daily_metrics(synthetic_aorc_dataset, ["temperature"])
    assert result["temperature_mean"].attrs["units"] == "deg_F"


def test_daily_metrics_specific_humidity_matches_manual_numpy(synthetic_aorc_dataset):
    result = pipeline.daily_metrics(synthetic_aorc_dataset, ["specific_humidity"]).compute()

    loaded = synthetic_aorc_dataset.compute()
    # Carried straight through in kg/kg -- no conversion at all.
    hourly = loaded["SPFH_2maboveground"].values.astype(np.float32)

    for day in (0, 1):
        window = hourly[day * 24 : (day + 1) * 24]
        np.testing.assert_allclose(
            result["specific_humidity_min"].isel(time=day).values, window.min(axis=0), rtol=1e-5
        )
        np.testing.assert_allclose(
            result["specific_humidity_mean"].isel(time=day).values, window.mean(axis=0), rtol=1e-5
        )
        np.testing.assert_allclose(
            result["specific_humidity_max"].isel(time=day).values, window.max(axis=0), rtol=1e-5
        )


def test_specific_humidity_is_kg_per_kg(synthetic_aorc_dataset):
    result = pipeline.daily_metrics(synthetic_aorc_dataset, ["specific_humidity"])
    assert result["specific_humidity_mean"].attrs["units"] == "kg/kg"


@pytest.mark.parametrize("metric", ALL_METRICS)
def test_each_metric_computes_when_requested_alone(synthetic_aorc_dataset, metric):
    """A metric run by itself must succeed on a dataset holding only its inputs.

    `open_aorc_year` subsets the opened year to `required_variables`, so a
    single-metric run reaches `daily_metrics` with only that metric's variables
    present. This reproduces that subset -- without it, a metric that reads a
    strict subset of the raw variables (temperature, specific_humidity) crashed
    with a KeyError inside `_derived_inputs`, and no full-fixture test caught it.
    """
    subset = synthetic_aorc_dataset[pipeline.required_variables([metric])]

    result = pipeline.daily_metrics(subset, [metric]).compute()

    assert sorted(result.data_vars) == [f"{metric}_{s}" for s in ("max", "mean", "min")]
    # A cell inside the (all-True) fixture region must carry real data.
    assert np.isfinite(result[f"{metric}_mean"].values).all()


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


def test_heat_index_and_relative_humidity_clamp_supersaturated_air():
    # core.relative_humidity routinely exceeds 100% from float noise in
    # saturated air (see test_wet_bulb_temperature_supersaturated_equals_air_temperature
    # in test_core.py), and heat_index's regression is only defined over
    # 0-100% RH: at T=95 F an unclamped RH=110 swings heat_index ~16 F versus
    # RH=100. Build a dataset supersaturated to exactly RH=110% before
    # clamping, and check both that relative_humidity itself reads back
    # clamped to 100 (not 110), and that heat_index -- fed the same clamped
    # intermediate -- matches heat_index evaluated at exactly RH=100%.
    air_temperature_celsius = np.float32(35.0)
    air_temperature_kelvin = air_temperature_celsius + np.float32(273.15)
    air_pressure_hpa = np.float32(1000.0)

    saturation = core.saturation_vapor_pressure(air_temperature_celsius)
    supersaturated_vapor_pressure = saturation * np.float32(1.10)
    epsilon = core.DRY_AIR_TO_VAPOR_MOLAR_MASS_RATIO
    specific_humidity = (epsilon * supersaturated_vapor_pressure) / (
        air_pressure_hpa - (np.float32(1.0) - epsilon) * supersaturated_vapor_pressure
    )

    hours = 24
    times = xr.date_range(
        "2000-07-01", periods=hours, freq="h", use_cftime=True, calendar="standard"
    )
    shape = (hours, 1, 1)
    dataset = xr.Dataset(
        {
            "TMP_2maboveground": (
                ("time", "latitude", "longitude"),
                np.full(shape, air_temperature_kelvin, dtype=np.float32),
            ),
            "SPFH_2maboveground": (
                ("time", "latitude", "longitude"),
                np.full(shape, specific_humidity, dtype=np.float32),
            ),
            "PRES_surface": (
                ("time", "latitude", "longitude"),
                np.full(shape, air_pressure_hpa * np.float32(100.0), dtype=np.float32),
            ),
        },
        coords={"time": times, "latitude": [30.0], "longitude": [-99.0]},
    ).chunk({"time": 24})

    result = pipeline.daily_metrics(dataset, ["heat_index", "relative_humidity"]).compute()

    assert float(result["relative_humidity_mean"].item()) == pytest.approx(100.0, abs=0.1)

    expected_heat_index = core.heat_index(
        core.celsius_to_fahrenheit(air_temperature_celsius), np.float32(100.0)
    )
    assert float(result["heat_index_mean"].item()) == pytest.approx(
        float(expected_heat_index), rel=1e-4
    )


def test_prepare_dataset_crops_and_masks(synthetic_aorc_dataset):
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

    loaded = pipeline.prepare_dataset(synthetic_aorc_dataset, region).compute()
    assert not np.isnan(loaded["TMP_2maboveground"].isel(latitude=0, longitude=0)).any()
    assert bool(np.isnan(loaded["TMP_2maboveground"].isel(latitude=1, longitude=1)).all())


def test_prepare_dataset_leaves_chunking_at_the_source_native_values(synthetic_aorc_dataset):
    """`prepare_dataset` does not rechunk -- time or space.

    AORC's native 144-hour chunk already holds six midnight-aligned days, so a
    daily resample group never crosses a boundary without any rechunk;
    splitting to 24 hours only inflates the graph. Spatial alignment is the
    region's responsibility, snapped before this function is called. So whatever
    chunking the source arrives with must survive untouched -- checked here on a
    deliberately odd 6-hour time chunk and 1-cell spatial chunks, both of which
    an imposed rechunk would erase.
    """
    from aorc_heat.mask import Region

    source = synthetic_aorc_dataset.chunk(
        {"time": 6, "latitude": 1, "longitude": 1}
    )
    mask = xr.DataArray(
        np.ones((2, 2), dtype=bool),
        dims=("latitude", "longitude"),
        coords={
            "latitude": source.latitude,
            "longitude": source.longitude,
        },
    )
    region = Region(mask=mask, latitude_slice=slice(0, 2), longitude_slice=slice(0, 2))

    prepared = pipeline.prepare_dataset(source, region)

    assert prepared.chunksizes["time"] == (6,) * 8  # 48 hours / 6, untouched
    assert prepared.chunksizes["latitude"] == (1, 1)
    assert prepared.chunksizes["longitude"] == (1, 1)


def test_native_time_chunking_gives_identical_daily_reductions(synthetic_aorc_dataset):
    """The daily min/mean/max must not depend on the input's time chunking.

    This is what licenses leaving time at AORC's native 144 hours instead of
    rechunking to 24: the reduction is chunk-invariant, so the cheaper chunking
    is free. If a future change makes the reduction chunk-sensitive, this fails.
    """
    native = synthetic_aorc_dataset  # fixture is chunked at 24h
    six_day = synthetic_aorc_dataset.chunk({"time": 48})  # one chunk spanning both days

    a = pipeline.daily_metrics(native, ALL_METRICS).compute()
    b = pipeline.daily_metrics(six_day, ALL_METRICS).compute()

    for name in a.data_vars:
        np.testing.assert_array_equal(a[name].values, b[name].values)


def test_daily_metrics_rejects_a_block_not_starting_at_midnight(synthetic_aorc_dataset):
    """`coarsen(time=24)` blindly bundles 24 consecutive samples per day.

    Unlike `resample`, it has no calendar awareness to fall back on, so an
    off-midnight start would silently mislabel every day from there on
    instead of raising -- this pins the explicit guard added for that.
    """
    misaligned = synthetic_aorc_dataset.isel(time=slice(1, 25))  # 24 hours, starts 01Z
    with pytest.raises(ValueError, match="00Z"):
        pipeline.daily_metrics(misaligned, ["temperature"])


def test_daily_metrics_rejects_a_block_not_a_whole_number_of_days(synthetic_aorc_dataset):
    partial = synthetic_aorc_dataset.isel(time=slice(0, 47))  # 47 hours, starts 00Z
    with pytest.raises(ValueError, match="24-hour days"):
        pipeline.daily_metrics(partial, ["temperature"])


def test_daily_metrics_time_coordinate_matches_daily_time_axis(synthetic_aorc_dataset):
    """Pins the left-labelling: `coarsen`'s `coord_func="min"` must produce the
    same midnight timestamps `resample(time="1D")` did, since `daily_time_axis`
    and `write_block`'s store lookup both key off day *n* of the input landing
    on day *n* of the store's axis.
    """
    result = pipeline.daily_metrics(synthetic_aorc_dataset, ["temperature"])

    # The fixture covers 1-2 July 2000: days 182 and 183 of a leap-year axis
    # (see test_write_block_fills_only_its_own_days, which pins the same fact
    # from the store-write side).
    axis = pipeline.daily_time_axis(2000, 2000)
    expected = np.asarray(axis[182:184])

    np.testing.assert_array_equal(
        np.asarray(result["temperature_mean"].time.values), expected
    )


def test_prepare_dataset_casts_to_float32(synthetic_aorc_dataset):
    """Raw AORC variables are float64 on disk; everything downstream is float32."""
    from aorc_heat.mask import Region

    source = synthetic_aorc_dataset.astype(np.float64)
    assert source["TMP_2maboveground"].dtype == np.float64

    mask = xr.DataArray(
        np.ones((2, 2), dtype=bool),
        dims=("latitude", "longitude"),
        coords={"latitude": source.latitude, "longitude": source.longitude},
    )
    region = Region(mask=mask, latitude_slice=slice(0, 2), longitude_slice=slice(0, 2))

    prepared = pipeline.prepare_dataset(source, region)

    for name, variable in prepared.data_vars.items():
        assert variable.dtype == np.float32, name


def test_native_spatial_chunks_reads_the_zarr_encoding(tmp_path):
    """Alignment must come from the on-disk chunk grid, not from dask.

    xarray may coalesce several stored chunks into one dask chunk when opening,
    so reading `.chunks` would give a different -- and wrong -- answer.
    """
    store_path = tmp_path / "source.zarr"
    dataset = xr.Dataset(
        {
            "TMP_2maboveground": (
                ("time", "latitude", "longitude"),
                np.zeros((4, 16, 24), dtype=np.float32),
            )
        },
        coords={
            "time": xr.date_range("2000-01-01", periods=4, freq="h", use_cftime=True),
            "latitude": np.arange(16.0),
            "longitude": np.arange(24.0),
        },
    )
    dataset.chunk({"time": 4, "latitude": 8, "longitude": 12}).to_zarr(
        store_path, zarr_format=2, consolidated=True
    )

    reopened = xr.open_zarr(store_path, consolidated=True)
    assert pipeline.native_spatial_chunks(reopened) == (8, 12)


def test_write_block_succeeds_when_the_raw_bbox_starts_mid_chunk(tmp_path):
    """A region write must succeed when the mask's own bbox starts mid-chunk.

    Mirrors the real Texas bounding box, whose start indices into the AORC
    domain (701, 2803) are prime and so land mid-chunk for any realistic
    native chunk size. Taken raw, that produces a partial *first* dask chunk;
    `initialize_output_store` would then declare the zarr chunk size from that
    partial remainder and `write_block` would raise "would overlap multiple
    Dask chunks" on the very first write.

    The guarantee now lives in `crop_to_bounding_box`, which snaps the box
    outward to native boundaries. This test goes through that real path --
    building a mask whose true extent starts at a prime offset, then cropping
    it with alignment -- rather than hand-constructing an already-aligned
    Region, so it exercises the actual production sequence.
    """
    from aorc_heat.mask import crop_to_bounding_box

    domain_size = 320
    native_chunks = (41, 43)  # both prime, so nothing divides the offsets below
    extent = 190
    latitude_start, longitude_start = 701 % domain_size, 2803 % domain_size

    full_latitudes = np.arange(domain_size, dtype=np.float64)
    full_longitudes = np.arange(domain_size, dtype=np.float64)
    hours = 48
    times = xr.date_range("2000-07-01", periods=hours, freq="h", use_cftime=True, calendar="standard")
    shape = (hours, domain_size, domain_size)
    generator = np.random.default_rng(seed=99)

    def field(low, high):
        return generator.uniform(low, high, size=shape).astype(np.float64)

    full_domain = xr.Dataset(
        {
            "TMP_2maboveground": (("time", "latitude", "longitude"), field(295.0, 315.0)),
            "SPFH_2maboveground": (("time", "latitude", "longitude"), field(0.005, 0.018)),
            "PRES_surface": (("time", "latitude", "longitude"), field(98000.0, 101500.0)),
        },
        coords={"time": times, "latitude": full_latitudes, "longitude": full_longitudes},
    ).chunk({"time": 24, "latitude": native_chunks[0], "longitude": native_chunks[1]})

    selected = np.zeros((domain_size, domain_size), dtype=bool)
    selected[
        latitude_start : latitude_start + extent,
        longitude_start : longitude_start + extent,
    ] = True
    full_mask = xr.DataArray(
        selected,
        dims=("latitude", "longitude"),
        coords={"latitude": full_latitudes, "longitude": full_longitudes},
    )

    # The raw bbox starts mid-chunk; snapping must pull it back to a boundary.
    assert latitude_start % native_chunks[0] != 0
    assert longitude_start % native_chunks[1] != 0
    region = crop_to_bounding_box(full_mask, alignment=native_chunks)
    assert region.latitude_slice.start % native_chunks[0] == 0
    assert region.longitude_slice.start % native_chunks[1] == 0

    prepared = pipeline.prepare_dataset(full_domain, region)
    daily = pipeline.daily_metrics(prepared, ["humidex"])

    store_path = tmp_path / pipeline.OUTPUT_STORE_NAME
    axis = pipeline.daily_time_axis(2000, 2000)
    pipeline.initialize_output_store(store_path, ["humidex"], axis, daily)

    # This must not raise the zarr "would overlap multiple Dask chunks" error.
    pipeline.write_block(store_path, daily, axis)

    written = xr.open_zarr(store_path, consolidated=True).compute()
    # The synthetic data covers 1-2 July 2000: days 182 and 183 of a leap year.
    written_days = written["humidex_mean"].isel(time=slice(182, 184)).values
    inside = region.mask.values

    # Every cell the mask selects carries real data...
    assert not bool(np.isnan(written_days[:, inside]).any())
    # ...and every cell the snap widened into stays NaN, so the margin is
    # inert rather than silently contributing values outside the boundary.
    assert bool(np.isnan(written_days[:, ~inside]).all())
    assert (~inside).any(), "snapping should have widened past the mask"

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
    assert written.chunksizes["time"][0] == pipeline.OUTPUT_TIME_CHUNK_DAYS


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


# ---------------------------------------------------------------------------
# Store chunk alignment across year boundaries
# ---------------------------------------------------------------------------
def test_chunk_sizes_aligned_to_store_splits_at_boundaries():
    """Every returned chunk must fall inside exactly one store chunk."""
    # A block starting on a boundary and ending past the next one.
    assert pipeline.chunk_sizes_aligned_to_store(365, 731, 365) == (365, 1)
    # A block starting mid-chunk (the usual case after a leap year).
    assert pipeline.chunk_sizes_aligned_to_store(731, 1096, 365) == (364, 1)
    # A block exactly filling one chunk.
    assert pipeline.chunk_sizes_aligned_to_store(0, 365, 365) == (365,)
    # A block smaller than a chunk and wholly inside one.
    assert pipeline.chunk_sizes_aligned_to_store(10, 20, 365) == (10,)


@pytest.mark.parametrize(
    "start, stop, chunk",
    [(0, 365, 365), (365, 731, 365), (731, 1096, 365), (5, 900, 90), (89, 271, 90)],
)
def test_chunk_sizes_aligned_to_store_are_sound(start, stop, chunk):
    """Sizes must sum to the block length, and no chunk may cross a boundary."""
    sizes = pipeline.chunk_sizes_aligned_to_store(start, stop, chunk)
    assert sum(sizes) == stop - start

    position = start
    for size in sizes:
        # The chunk this piece lands in, at its first and last index, must match.
        assert position // chunk == (position + size - 1) // chunk
        position += size


def test_consecutive_year_writes_preserve_the_shared_boundary_chunk(tmp_path):
    """The chunk straddling two years must retain both years' data.

    With a 365-day store chunk and alternating 365/366-day years, adjacent
    years share the chunk at their boundary: one writes its tail, the next
    writes its head. Correctness depends on zarr read-modify-writing that chunk
    rather than replacing it. This is the assumption `write_block`'s contract
    rests on, so it is checked directly: each day is stamped with its own index
    in the store's axis, and every day of both years must read back as itself.
    """
    store_path = tmp_path / pipeline.OUTPUT_STORE_NAME
    axis = pipeline.daily_time_axis(1999, 2000)  # 365 + 366, boundary mid-chunk
    latitudes, longitudes = np.array([30.0, 31.0]), np.array([-100.0, -99.0])

    def block_for(year):
        days = 365 if year == 1999 else 366
        times = xr.date_range(
            f"{year}-01-01", periods=days, freq="D", use_cftime=True, calendar="standard"
        )
        offset = 0 if year == 1999 else 365
        # Value == the day's index in the full store axis, so a misplaced or
        # clobbered day is immediately visible rather than plausible.
        values = np.repeat(
            (np.arange(days, dtype=np.float32) + offset)[:, None, None], 2, axis=1
        ).repeat(2, axis=2)
        return xr.Dataset(
            {"humidex_mean": (("time", "latitude", "longitude"), values)},
            coords={"time": times, "latitude": latitudes, "longitude": longitudes},
        ).chunk({"time": 50})  # deliberately misaligned with the store's 365

    template = block_for(1999)
    pipeline.initialize_output_store(store_path, ["humidex"], axis, template)
    pipeline.write_block(store_path, block_for(1999), axis)
    pipeline.write_block(store_path, block_for(2000), axis)

    written = xr.open_zarr(store_path, consolidated=True)["humidex_mean"].compute()
    assert written.sizes["time"] == 731
    expected = np.arange(731, dtype=np.float32)
    np.testing.assert_array_equal(written.isel(latitude=0, longitude=0).values, expected)
    # Day 364 (end of 1999) and day 365 (start of 2000) share store chunk 1.
    assert float(written.isel(time=364, latitude=0, longitude=0)) == 364.0
    assert float(written.isel(time=365, latitude=0, longitude=0)) == 365.0


def test_output_store_uses_the_configured_time_chunk(tmp_path, synthetic_aorc_dataset):
    """The store must actually be allocated at OUTPUT_TIME_CHUNK_DAYS."""
    store_path = tmp_path / pipeline.OUTPUT_STORE_NAME
    axis = pipeline.daily_time_axis(2000, 2000)
    template = pipeline.daily_metrics(synthetic_aorc_dataset, ["humidex"])

    pipeline.initialize_output_store(store_path, ["humidex"], axis, template)

    written = xr.open_zarr(store_path, consolidated=True)
    assert written.chunksizes["time"][0] == pipeline.OUTPUT_TIME_CHUNK_DAYS


def test_masked_cells_do_not_emit_runtime_warnings(synthetic_aorc_dataset):
    """A masked region must compute silently.

    Masked cells are NaN by design, and the branchy kernels raise IEEE-754's
    invalid-operation flag on SIMD lanes whose results are discarded. The
    results stay correct, so the pipeline suppresses the resulting warning; this
    check keeps that suppression working end to end rather than by inspection.
    """
    import warnings

    from aorc_heat.mask import Region

    # Large enough to exceed the SIMD width where if-conversion kicks in.
    grid = 32
    hours = 48
    times = xr.date_range(
        "2000-07-01", periods=hours, freq="h", use_cftime=True, calendar="standard"
    )
    generator = np.random.default_rng(7)
    shape = (hours, grid, grid)

    def field(low, high):
        return generator.uniform(low, high, size=shape).astype(np.float64)

    dataset = xr.Dataset(
        {
            "TMP_2maboveground": (("time", "latitude", "longitude"), field(295.0, 315.0)),
            "SPFH_2maboveground": (("time", "latitude", "longitude"), field(0.005, 0.018)),
            "PRES_surface": (("time", "latitude", "longitude"), field(98000.0, 101500.0)),
            "UGRD_10maboveground": (("time", "latitude", "longitude"), field(-8.0, 8.0)),
            "VGRD_10maboveground": (("time", "latitude", "longitude"), field(-8.0, 8.0)),
        },
        coords={
            "time": times,
            "latitude": np.arange(float(grid)),
            "longitude": np.arange(float(grid)),
        },
    ).chunk({"time": 24})

    selected = np.zeros((grid, grid), dtype=bool)
    selected[8:24, 8:24] = True  # most of the region masked out, as in Texas
    mask = xr.DataArray(
        selected,
        dims=("latitude", "longitude"),
        coords={"latitude": dataset.latitude, "longitude": dataset.longitude},
    )
    region = Region(mask=mask, latitude_slice=slice(0, grid), longitude_slice=slice(0, grid))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prepared = pipeline.prepare_dataset(dataset, region)
        result = pipeline.daily_metrics(prepared, ALL_METRICS).compute()

    runtime_warnings = [
        str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)
    ]
    assert not runtime_warnings, runtime_warnings

    # And the results are still right: masked NaN, unmasked finite.
    inside = mask.values
    for name in ALL_METRICS:
        values = result[f"{name}_mean"].values
        assert np.isnan(values[:, ~inside]).all(), name
        assert np.isfinite(values[:, inside]).all(), name
