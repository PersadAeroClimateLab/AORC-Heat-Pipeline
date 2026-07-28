"""Spatial masking of the AORC grid to the Texas state boundary.

The mask NetCDF is gitignored, so it is generated from the bundled shapefile on
first use and cached to disk. Because the AORC domain covers the whole of CONUS
and Texas occupies a small part of it, the mask is cropped to its own bounding
box; the index slices used for that crop travel with it so the same subsetting
can be applied to a full-domain dataset.
"""

from dataclasses import dataclass
from pathlib import Path

import numba as nb
import numpy as np
import xarray as xr

TEXAS_MASK_PATH = Path("boundary_data/texas_mask.nc")
TEXAS_SHAPEFILE_PATH = Path("boundary_data/texas_shapefile/State_Boundary.shp")

MASK_ATTRIBUTES = {
    "description": "Boolean mask of AORC grid points inside the Texas state boundary",
    "shapefile_url": "https://gis-txdot.opendata.arcgis.com/datasets/texas-state-boundary/explore",
    "credit": "Point-in-polygon routines adapted from a Stack Overflow post by user 'epifanio'",
    "credit_url": "https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python",
}


@dataclass(frozen=True)
class Region:
    """A boolean mask cropped to its own bounding box.

    :param mask: Boolean DataArray over dims ("latitude", "longitude")
    :param latitude_slice: Index slice that crops a full-domain latitude axis
    :param longitude_slice: Index slice that crops a full-domain longitude axis
    """

    mask: xr.DataArray
    latitude_slice: slice
    longitude_slice: slice


# --------------------------------------------------------------------------
# Point-in-polygon. Adapted from the Stack Overflow post credited above.
# --------------------------------------------------------------------------
@nb.njit(cache=True)
def point_in_polygon(longitude, latitude, polygon):
    """Ray-casting test for a single point against a closed polygon.

    :param longitude: Point longitude in degrees east
    :param latitude: Point latitude in degrees north
    :param polygon: (N, 2) array of polygon vertices as (longitude, latitude)
    :return: True when the point lies inside the polygon
    """
    vertex_count = len(polygon)
    inside = False
    previous_longitude, previous_latitude = polygon[0]
    for index in range(vertex_count + 1):
        current_longitude, current_latitude = polygon[index % vertex_count]
        # This gate (strictly greater than the lower latitude, less-than-or-
        # equal to the upper one) can never be satisfied when the edge is
        # horizontal (previous_latitude == current_latitude), since min ==
        # max there and no value is both > and <= the same number. Horizontal
        # edges are therefore excluded by this check alone and need no
        # special-cased handling below.
        if latitude > min(previous_latitude, current_latitude):
            if latitude <= max(previous_latitude, current_latitude):
                if longitude <= max(previous_longitude, current_longitude):
                    crossing = (latitude - previous_latitude) * (
                        current_longitude - previous_longitude
                    ) / (current_latitude - previous_latitude) + previous_longitude
                    if previous_longitude == current_longitude or longitude <= crossing:
                        inside = not inside
        previous_longitude, previous_latitude = current_longitude, current_latitude
    return inside


@nb.njit(cache=True)
def _point_in_rings(longitude, latitude, vertices, ring_bounds, polygon_bounds):
    """Ray-casting test against one or more polygons, each possibly with holes.

    Within a polygon, the ray-casting parity from each of its rings (the
    exterior plus any interior/hole rings) is XORed together: ray-casting
    parity composes, so being inside the exterior but inside an odd number of
    holes cancels out to exactly "inside the exterior and not inside a hole".
    Across polygons the per-polygon results are OR-ed, since a point belongs
    to a MultiPolygon -- or to multiple shapefile features -- if it lies
    inside ANY one of its parts.

    :param longitude: Point longitude in degrees east
    :param latitude: Point latitude in degrees north
    :param vertices: (V, 2) array of all ring vertices concatenated
    :param ring_bounds: (R+1,) offsets into `vertices` bounding each ring
    :param polygon_bounds: (P+1,) offsets into `ring_bounds` bounding each
        polygon's rings
    :return: True when the point lies inside the geometry
    """
    for polygon_index in range(len(polygon_bounds) - 1):
        inside_polygon = False
        for ring_index in range(polygon_bounds[polygon_index], polygon_bounds[polygon_index + 1]):
            ring = vertices[ring_bounds[ring_index] : ring_bounds[ring_index + 1]]
            if point_in_polygon(longitude, latitude, ring):
                inside_polygon = not inside_polygon
        if inside_polygon:
            return True
    return False


@nb.njit(parallel=True, cache=True)
def _grid_in_polygon(latitudes, longitudes, polygon):
    """Point-in-polygon over a full lat/lon grid."""
    result = np.empty((latitudes.size, longitudes.size), dtype=nb.boolean)
    for row in nb.prange(latitudes.size):
        for column in range(longitudes.size):
            result[row, column] = point_in_polygon(longitudes[column], latitudes[row], polygon)
    return result


@nb.njit(parallel=True, cache=True)
def _grid_in_rings(latitudes, longitudes, vertices, ring_bounds, polygon_bounds):
    """Point-in-(multi-polygon-with-holes) over a full lat/lon grid."""
    result = np.empty((latitudes.size, longitudes.size), dtype=nb.boolean)
    for row in nb.prange(latitudes.size):
        for column in range(longitudes.size):
            result[row, column] = _point_in_rings(
                longitudes[column], latitudes[row], vertices, ring_bounds, polygon_bounds
            )
    return result


def _wrap_mask(values, latitudes, longitudes):
    """Package a boolean grid as the mask DataArray both builders return."""
    return xr.DataArray(
        values,
        dims=("latitude", "longitude"),
        coords={"latitude": latitudes, "longitude": longitudes},
        name="mask",
        attrs=MASK_ATTRIBUTES,
    )


def mask_from_polygon(latitudes, longitudes, polygon):
    """Build a boolean mask over a lat/lon grid from a single polygon ring.

    For a general geometry that may have multiple parts or holes, use
    `mask_from_rings` instead.

    :param latitudes: 1-D array of latitudes in degrees north
    :param longitudes: 1-D array of longitudes in degrees east
    :param polygon: (N, 2) array of polygon vertices as (longitude, latitude)
    :return: Boolean DataArray over dims ("latitude", "longitude")
    """
    values = _grid_in_polygon(
        np.ascontiguousarray(latitudes, dtype=np.float64),
        np.ascontiguousarray(longitudes, dtype=np.float64),
        np.ascontiguousarray(polygon, dtype=np.float64),
    )
    return _wrap_mask(values, latitudes, longitudes)


def mask_from_rings(latitudes, longitudes, vertices, ring_bounds, polygon_bounds):
    """Build a boolean mask over a lat/lon grid from multiple polygons with holes.

    :param latitudes: 1-D array of latitudes in degrees north
    :param longitudes: 1-D array of longitudes in degrees east
    :param vertices: (V, 2) array of all ring vertices concatenated, as
        (longitude, latitude) pairs
    :param ring_bounds: (R+1,) offsets into `vertices` bounding each ring;
        ring i is `vertices[ring_bounds[i]:ring_bounds[i + 1]]`
    :param polygon_bounds: (P+1,) offsets into `ring_bounds` bounding each
        polygon's rings; polygon j's rings are
        `range(polygon_bounds[j], polygon_bounds[j + 1])`. A polygon's first
        ring is its exterior; any further rings are holes.
    :return: Boolean DataArray over dims ("latitude", "longitude")
    """
    values = _grid_in_rings(
        np.ascontiguousarray(latitudes, dtype=np.float64),
        np.ascontiguousarray(longitudes, dtype=np.float64),
        np.ascontiguousarray(vertices, dtype=np.float64),
        np.ascontiguousarray(ring_bounds, dtype=np.int64),
        np.ascontiguousarray(polygon_bounds, dtype=np.int64),
    )
    return _wrap_mask(values, latitudes, longitudes)


def build_mask(latitudes, longitudes, shapefile_path):
    """Build a boundary mask by reading a shapefile.

    Every feature in the shapefile contributes, each feature's geometry may be
    a Polygon or a MultiPolygon, and each polygon's interior rings (holes) are
    subtracted rather than filled in. A point is inside the mask if it is
    inside any polygon's exterior and not inside that polygon's holes.

    :param latitudes: 1-D array of latitudes in degrees north
    :param longitudes: 1-D array of longitudes in degrees east
    :param shapefile_path: Path to a shapefile describing the boundary
    :return: Boolean DataArray over dims ("latitude", "longitude")
    """
    import geopandas

    shapefile_path = Path(shapefile_path)
    if not shapefile_path.is_file():
        raise FileNotFoundError(f"Boundary shapefile not found at {shapefile_path}")

    boundary = geopandas.read_file(shapefile_path).to_crs(crs=4326)

    vertex_arrays = []
    ring_bounds = [0]
    polygon_bounds = [0]
    for geometry in boundary.geometry:
        polygons = geometry.geoms if geometry.geom_type == "MultiPolygon" else (geometry,)
        for polygon in polygons:
            for ring in (polygon.exterior, *polygon.interiors):
                ring_vertices = np.array(ring.coords.xy).T
                vertex_arrays.append(ring_vertices)
                ring_bounds.append(ring_bounds[-1] + len(ring_vertices))
            polygon_bounds.append(len(ring_bounds) - 1)

    vertices = np.concatenate(vertex_arrays, axis=0)
    return mask_from_rings(
        latitudes,
        longitudes,
        vertices,
        np.array(ring_bounds, dtype=np.int64),
        np.array(polygon_bounds, dtype=np.int64),
    )


def _snap_outward(start, stop, alignment, limit):
    """Widen [start, stop) so it begins on a multiple of `alignment`.

    :param start: First selected index
    :param stop: One past the last selected index
    :param alignment: Chunk size to align the start to
    :param limit: Size of the axis, so the end never runs past the domain
    :return: (start, stop) widened to chunk boundaries
    """
    widened_start = (start // alignment) * alignment
    widened_stop = min(-(-stop // alignment) * alignment, limit)
    return widened_start, widened_stop


def crop_to_bounding_box(mask, alignment=None):
    """Crop a mask to the bounding box of its selected cells.

    With `alignment`, the box is widened outward so it *starts* on a multiple of
    the given chunk sizes. That matters because zarr chunks are atomic: reading
    a box whose start falls mid-chunk still downloads the whole chunk, so the
    extra cells are already being paid for. Aligning the start means the cropped
    array's first chunk is a full one, which in turn keeps every later chunk
    boundary lined up with the store's -- the condition zarr region writes
    require. A partial chunk at the far end is harmless; only a partial first
    chunk desynchronises everything after it.

    :param mask: Boolean DataArray over dims ("latitude", "longitude")
    :param alignment: Optional (latitude, longitude) chunk sizes to align to
    :return: A Region holding the cropped mask and the slices that produced it
    """
    latitude_indices = np.flatnonzero(mask.any("longitude").values)
    longitude_indices = np.flatnonzero(mask.any("latitude").values)
    if latitude_indices.size == 0 or longitude_indices.size == 0:
        raise ValueError("Cannot crop a mask with no selected cells.")

    latitude_start, latitude_stop = int(latitude_indices[0]), int(latitude_indices[-1]) + 1
    longitude_start, longitude_stop = int(longitude_indices[0]), int(longitude_indices[-1]) + 1

    if alignment is not None:
        latitude_alignment, longitude_alignment = alignment
        latitude_start, latitude_stop = _snap_outward(
            latitude_start, latitude_stop, latitude_alignment, mask.sizes["latitude"]
        )
        longitude_start, longitude_stop = _snap_outward(
            longitude_start, longitude_stop, longitude_alignment, mask.sizes["longitude"]
        )

    latitude_slice = slice(latitude_start, latitude_stop)
    longitude_slice = slice(longitude_start, longitude_stop)
    return Region(
        mask=mask.isel(latitude=latitude_slice, longitude=longitude_slice),
        latitude_slice=latitude_slice,
        longitude_slice=longitude_slice,
    )


def texas_region(
    latitudes,
    longitudes,
    mask_path=TEXAS_MASK_PATH,
    shapefile_path=TEXAS_SHAPEFILE_PATH,
    alignment=None,
):
    """Load the Texas mask, generating and caching it from the shapefile if absent.

    :param latitudes: 1-D array of AORC latitudes in degrees north
    :param longitudes: 1-D array of AORC longitudes in degrees east
    :param mask_path: Where the generated mask is cached
    :param shapefile_path: Source shapefile used when the cache is missing
    :param alignment: Optional (latitude, longitude) chunk sizes to align the
        bounding box start to; see `crop_to_bounding_box`
    :return: A Region cropped to the Texas bounding box
    """
    mask_path = Path(mask_path)
    if mask_path.is_file():
        mask = xr.open_dataset(mask_path)["mask"].astype(bool)
        if not np.array_equal(mask.latitude.values, latitudes) or not np.array_equal(
            mask.longitude.values, longitudes
        ):
            raise ValueError(
                f"Cached mask at {mask_path} does not match the AORC grid. "
                "Delete it to regenerate."
            )
    else:
        mask = build_mask(latitudes, longitudes, shapefile_path)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask.to_dataset(name="mask").to_netcdf(mask_path)

    return crop_to_bounding_box(mask, alignment=alignment)
