"""Unit tests for region masking.

The point-in-polygon routine is checked against a unit square, where membership
is obvious by inspection, rather than against the Texas shapefile -- which is
gitignored output and not available in a clean checkout.
"""
import numpy as np
import pytest
import xarray as xr

from aorc_heat import mask as mask_module


UNIT_SQUARE = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])


@pytest.mark.parametrize(
    "longitude, latitude, inside",
    [
        (0.5, 0.5, True),    # centre
        (0.1, 0.9, True),    # near a corner, still inside
        (1.5, 0.5, False),   # east of the square
        (0.5, 1.5, False),   # north of the square
        (-0.5, 0.5, False),  # west of the square
    ],
)
def test_point_in_polygon(longitude, latitude, inside):
    assert mask_module.point_in_polygon(longitude, latitude, UNIT_SQUARE) == inside


@pytest.mark.skipif(
    not mask_module.TEXAS_MASK_PATH.is_file() or not mask_module.TEXAS_SHAPEFILE_PATH.is_file(),
    reason="cached mask or shapefile not present in this checkout",
)
def test_build_mask_from_shapefile_matches_cached_mask_on_a_grid_subset():
    # build_mask now handles multiple features, MultiPolygon parts, and
    # interior rings (holes) -- but this is a robustness fix, not a behaviour
    # change: the bundled shapefile is (verified) 1 feature, 1 Polygon, 0
    # interior rings, so today's correct answer must come out bit-identical.
    # Point-in-polygon over the full 4201x8401 CONUS grid against ~90,405
    # vertices is slow, so this checks every 20th lat/lon rather than the
    # whole domain.
    cached = xr.open_dataset(mask_module.TEXAS_MASK_PATH)["mask"].astype(bool)
    subset_latitudes = cached.latitude.values[::20]
    subset_longitudes = cached.longitude.values[::20]

    rebuilt = mask_module.build_mask(
        subset_latitudes, subset_longitudes, mask_module.TEXAS_SHAPEFILE_PATH
    )
    cached_subset = cached.isel(
        latitude=slice(None, None, 20), longitude=slice(None, None, 20)
    )

    assert np.array_equal(rebuilt.values, cached_subset.values)


def test_mask_from_polygon_has_grid_dimensions():
    latitudes = np.linspace(-1.0, 2.0, 13, dtype=np.float64)
    longitudes = np.linspace(-1.0, 2.0, 13, dtype=np.float64)
    result = mask_module.mask_from_polygon(latitudes, longitudes, UNIT_SQUARE)

    assert result.dims == ("latitude", "longitude")
    assert result.shape == (13, 13)
    assert result.dtype == bool
    # Every selected cell must lie inside the unit square.
    selected = result.where(result, drop=True)
    assert float(selected.latitude.min()) >= 0.0
    assert float(selected.latitude.max()) <= 1.0


def test_mask_from_polygon_does_not_transpose_lat_lon_axes():
    # test_mask_from_polygon_has_grid_dimensions uses the same array for both
    # latitudes and longitudes against a symmetric unit square, so it cannot
    # detect a row/column (lat/lon) swap -- a transposed implementation would
    # produce byte-identical output on that test. Here the grid axes have
    # different lengths and different ranges, and the polygon is a rectangle
    # that is wide in longitude but narrow in latitude, with a longitude
    # range that does not overlap the latitude range at all. If latitude and
    # longitude were ever swapped internally, the selected cells would land
    # outside the ranges asserted below (or the mask would select nothing).
    latitudes = np.linspace(10.0, 12.0, 5, dtype=np.float64)
    longitudes = np.linspace(-100.0, -90.0, 21, dtype=np.float64)
    rectangle = np.array(
        [
            [-98.0, 10.5],
            [-98.0, 11.5],
            [-92.0, 11.5],
            [-92.0, 10.5],
            [-98.0, 10.5],
        ]
    )

    result = mask_module.mask_from_polygon(latitudes, longitudes, rectangle)

    assert result.dims == ("latitude", "longitude")
    assert result.shape == (5, 21)
    selected = result.where(result, drop=True)
    assert selected.latitude.size > 0
    assert selected.longitude.size > 0
    # Selected cells must fall within the rectangle's per-axis ranges.
    assert float(selected.latitude.min()) >= 10.5
    assert float(selected.latitude.max()) <= 11.5
    assert float(selected.longitude.min()) >= -98.0
    assert float(selected.longitude.max()) <= -92.0


def test_crop_to_bounding_box_trims_empty_margins():
    latitudes = np.arange(10.0, 20.0)
    longitudes = np.arange(100.0, 110.0)
    values = np.zeros((10, 10), dtype=bool)
    values[3:6, 4:8] = True
    full_mask = xr.DataArray(
        values,
        dims=("latitude", "longitude"),
        coords={"latitude": latitudes, "longitude": longitudes},
    )

    region = mask_module.crop_to_bounding_box(full_mask)

    assert region.latitude_slice == slice(3, 6)
    assert region.longitude_slice == slice(4, 8)
    assert region.mask.shape == (3, 4)
    assert bool(region.mask.all())


def test_crop_to_bounding_box_rejects_empty_mask():
    empty = xr.DataArray(
        np.zeros((4, 4), dtype=bool),
        dims=("latitude", "longitude"),
        coords={"latitude": np.arange(4.0), "longitude": np.arange(4.0)},
    )
    with pytest.raises(ValueError, match="no selected cells"):
        mask_module.crop_to_bounding_box(empty)


def test_crop_to_bounding_box_snaps_outward_to_chunk_boundaries():
    """With alignment, the box must start on a chunk boundary and still contain
    every selected cell. Zarr chunks are atomic, so the widened cells are
    already being downloaded either way -- snapping keeps them instead of
    discarding them, and makes the first chunk a full one."""
    latitudes = np.arange(100.0)
    longitudes = np.arange(200.0)
    values = np.zeros((100, 200), dtype=bool)
    values[37:61, 83:129] = True  # starts mid-chunk on both axes
    full_mask = xr.DataArray(
        values,
        dims=("latitude", "longitude"),
        coords={"latitude": latitudes, "longitude": longitudes},
    )

    region = mask_module.crop_to_bounding_box(full_mask, alignment=(16, 32))

    assert region.latitude_slice == slice(32, 64)
    assert region.longitude_slice == slice(64, 160)
    assert region.latitude_slice.start % 16 == 0
    assert region.longitude_slice.start % 32 == 0
    # Widened, never narrowed: every originally-selected cell survives.
    assert int(region.mask.sum()) == int(full_mask.sum())


def test_crop_to_bounding_box_snapping_never_runs_past_the_domain():
    """The far edge clamps to the axis length, leaving a partial last chunk --
    which is harmless, unlike a partial first chunk."""
    latitudes = np.arange(50.0)
    longitudes = np.arange(50.0)
    values = np.zeros((50, 50), dtype=bool)
    values[40:50, 40:50] = True
    full_mask = xr.DataArray(
        values,
        dims=("latitude", "longitude"),
        coords={"latitude": latitudes, "longitude": longitudes},
    )

    region = mask_module.crop_to_bounding_box(full_mask, alignment=(16, 16))

    assert region.latitude_slice == slice(32, 50)
    assert region.longitude_slice == slice(32, 50)


def test_crop_to_bounding_box_without_alignment_is_unchanged():
    """Omitting alignment must keep the exact tight bounding box."""
    latitudes = np.arange(20.0)
    longitudes = np.arange(20.0)
    values = np.zeros((20, 20), dtype=bool)
    values[3:6, 4:8] = True
    full_mask = xr.DataArray(
        values,
        dims=("latitude", "longitude"),
        coords={"latitude": latitudes, "longitude": longitudes},
    )

    region = mask_module.crop_to_bounding_box(full_mask)

    assert region.latitude_slice == slice(3, 6)
    assert region.longitude_slice == slice(4, 8)
