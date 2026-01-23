from shpmask import get_data_array_mask
import xarray as xr
import s3fs
from numba import set_num_threads


if __name__ == "__main__":
    print("Starting AORC analysis pipeline")

    s3_base_path = "s3://noaa-nws-aorc-v1-1-1km/"

    fs = s3fs.S3FileSystem(anon=True)
    # datasets = []
    # for year in range(1979, 2025):
    #     mapper = fs.get_mapper(f"{s3_base_path}{year}.zarr")
    #     datasets.append(xr.open_zarr(mapper, consolidated=True))
    # aorc_ds = xr.concat(datasets, dim="time")

    year = 1979
    mapper = fs.get_mapper(f"{s3_base_path}{year}.zarr")
    aorc_ds = xr.open_zarr(mapper, consolidated=True)

    set_num_threads(12)
    mask_ds = get_data_array_mask(aorc_ds.latitude.values, aorc_ds.longitude.values, "Texas_Shapefile/State_Boundary.shp")
    mask_ds.to_netcdf("Texas_mask.nc")