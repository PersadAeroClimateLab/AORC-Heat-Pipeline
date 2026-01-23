import xarray
import geopandas as gpd
import numpy as np
import numba as nb

# ======================= NOTE: Copied Code =======================
# https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
@nb.jit(nopython=True)
def pointinpolygon(x,y,poly):
    """
    https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
    """
    n = len(poly)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x,p1y = poly[0]
    for i in nb.prange(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y
    return inside


@nb.njit(parallel=True)
def parallelpointinpolygon(lats, lons, polygon):
    """
    https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
    """
    D = np.empty((lats.size, lons.size), dtype=nb.boolean)
    for i in nb.prange(0, D.shape[0]):
        for j in nb.prange(0, D.shape[1]):
            D[i, j] = pointinpolygon(lons[j], lats[i], polygon)
    return D
# ======================= End of copied code =======================

def get_data_array_mask(lats, lons, shpfile_path):
    shapefile = gpd.read_file(shpfile_path).to_crs(crs=4326)
    polygon = np.array(shapefile.geometry[0].exterior.coords.xy).T

    booleans = parallelpointinpolygon(lats, lons, polygon)

    return xarray.Dataset(
        data_vars={
            "mask":(["x","y"], booleans),
        },
        coords=dict(
            latitude=(["x"], lats),
            longitude=(["y"], lons),
        ),
        attrs={
            "description": "binary mask of points from reference dataset that are inside the boundaries of Texas",
            "shapefile_url": "https://gis-txdot.opendata.arcgis.com/datasets/texas-state-boundary/explore",
            "credit": "The nb functions used to quickly mask this data were found on a stack overflow post by the user 'epifanio'",
            "credit_url": "https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python"
        }
    )