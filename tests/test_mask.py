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
