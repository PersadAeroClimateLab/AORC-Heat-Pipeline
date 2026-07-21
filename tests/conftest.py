"""Shared fixtures for the AORC-Heat-Pipeline test suite.

The package is installed into the image with `pip install -e .`, so tests import
`aorc_heat` directly with no path manipulation.
"""
import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def synthetic_aorc_dataset():
    """Two days of hourly AORC-shaped data on a 2x2 grid.

    Values are physically plausible for a Texas summer so that every metric
    exercises its main branch rather than an edge case. Chunked at 24 hours to
    match what `prepare_dataset` produces.
    """
    hours = 48
    times = xr.date_range("2000-07-01", periods=hours, freq="h", use_cftime=True, calendar="standard")
    latitudes = np.array([30.0, 31.0])
    longitudes = np.array([-100.0, -99.0])
    shape = (hours, latitudes.size, longitudes.size)

    generator = np.random.default_rng(seed=20260721)

    def field(low, high):
        return generator.uniform(low, high, size=shape).astype(np.float32)

    dataset = xr.Dataset(
        {
            "TMP_2maboveground": (("time", "latitude", "longitude"), field(295.0, 315.0)),
            "SPFH_2maboveground": (("time", "latitude", "longitude"), field(0.005, 0.018)),
            "PRES_surface": (("time", "latitude", "longitude"), field(98000.0, 101500.0)),
            "UGRD_10maboveground": (("time", "latitude", "longitude"), field(-8.0, 8.0)),
            "VGRD_10maboveground": (("time", "latitude", "longitude"), field(-8.0, 8.0)),
        },
        coords={"time": times, "latitude": latitudes, "longitude": longitudes},
    )
    return dataset.chunk({"time": 24})
