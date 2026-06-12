from shpmask import get_data_array_mask
from numba import set_num_threads
from os.path import isfile, isdir
import xarray as xr
import numpy as np
import s3fs
import metrics
from datetime import datetime
from dask.distributed import LocalCluster, Client

PATH_TO_TEXAS_MASK = "boundary_data/texas_mask.nc"
PATH_TO_TEXAS_SHP = "boundary_data/texas_shapefile/State_Boundary.shp"
NUM_THREADS_NUMBA = 2

if __name__ == "__main__":
    print(f"{datetime.now().timestamp()} [init] Starting AORC heat analysis pipeline, initiating Dask cluster.")

    cluster = LocalCluster(n_workers=80, threads_per_worker=2, memory_limit="20GB", dashboard_address=":8002")
    client = cluster.get_client()
    set_num_threads(NUM_THREADS_NUMBA)
    print(client)

    s3_base_path = "s3://noaa-nws-aorc-v1-1-1km/"
    print(f"{datetime.now().timestamp()} [init] Connecting to AORC S3 bucket '{s3_base_path}'")

    fs = s3fs.S3FileSystem(anon=True)

    aorc_sample_ds = xr.open_zarr(fs.get_mapper(f"{s3_base_path}1979.zarr"), consolidated=True)

    if isfile(PATH_TO_TEXAS_MASK):
        print(f"{datetime.now().timestamp()} [init] Existing mask dataset for Texas found.")
        texas_mask = xr.open_dataset(PATH_TO_TEXAS_MASK)["mask"]
        assert np.array_equal(texas_mask.latitude.values, aorc_sample_ds.latitude.values) 
        assert np.array_equal(texas_mask.longitude.values, aorc_sample_ds.longitude.values) 
    else:
        print(f"{datetime.now().timestamp()} [init] A mask dataset for Texas must generated since none was found. This may take a while!")
        set_num_threads(NUM_THREADS_NUMBA)
        texas_mask = get_data_array_mask(aorc_ds.latitude.values, aorc_ds.longitude.values, PATH_TO_TEXAS_SHP)
        texas_mask.to_netcdf("Texas_mask.nc")

    global_attrs = {
        "description": "Heat metrics derived from NOAA NWS AORC data provided via AWS S3 Zarr Bucket",
        "source_version": "AORC Version 1.1", 
        "source_url": "https://registry.opendata.aws/noaa-nws-aorc",
    }

    var_attrs = {
        "units": "deg_F",
        "source_timestep": "hourly",
        "desc": "Statistic taken across 24 hours"
    }

    for year in range(1979, 2025):
        print(f"{datetime.now().timestamp()} [compute] Loading data and applying Texas mask for year {year}.")
        aorc_ds = xr.open_zarr(fs.get_mapper(f"{s3_base_path}{year}.zarr"), consolidated=True).astype(np.float32).where(texas_mask, drop=True)

        
        print(f"{datetime.now().timestamp()} [compute] Calculating heat index metrics for {year}.")
        aorc_hi = xr.apply_ufunc(
            metrics.get_aorc_heat_index,
            aorc_ds["TMP_2maboveground"],
            aorc_ds["SPFH_2maboveground"],
            dask="parallelized",
            output_dtypes=[float]
        )
        da = aorc_hi.resample(time="1D").mean().chunk('auto')
        da.attrs = var_attrs
        if not isdir(f"yearly_metrics_zarrs/AORC_heat_index_mean_{year}.zarr"):
            xr.Dataset(data_vars={"heat_index_mean": da}, attrs=global_attrs).to_zarr(
                f"yearly_metrics_zarrs/AORC_heat_index_mean_{year}.zarr", zarr_format=2
            )
        da = aorc_hi.resample(time="1D").min().chunk('auto')
        da.attrs = var_attrs
        if not isdir(f"yearly_metrics_zarrs/AORC_heat_index_min_{year}.zarr"):
            xr.Dataset(data_vars={"heat_index_min": da}, attrs=global_attrs).to_zarr(
                f"yearly_metrics_zarrs/AORC_heat_index_min_{year}.zarr", zarr_format=2
            )
        da = aorc_hi.resample(time="1D").max().chunk('auto')
        da.attrs = var_attrs
        if not isdir(f"yearly_metrics_zarrs/AORC_heat_index_max_{year}.zarr"):
            xr.Dataset(data_vars={"heat_index_max": da}, attrs=global_attrs).to_zarr(
                f"yearly_metrics_zarrs/AORC_heat_index_max_{year}.zarr", zarr_format=2
            )
        del aorc_hi


        print(f"{datetime.now().timestamp()} [compute] Calculating apparent temperature metrics for {year}.")
        aorc_atemp = xr.apply_ufunc(
            metrics.get_aorc_apparent_temp,
            aorc_ds["TMP_2maboveground"],
            aorc_ds["UGRD_10maboveground"],
            aorc_ds["VGRD_10maboveground"],
            aorc_ds["SPFH_2maboveground"],
            dask="parallelized",
            output_dtypes=[float]
        ).chunk('auto')
        da = aorc_atemp.resample(time="1D").mean().chunk('auto')
        da.attrs = var_attrs
        if not isdir(f"yearly_metrics_zarrs/AORC_apparent_temp_mean_{year}.zarr"):
            xr.Dataset(data_vars={"apparent_temp_mean": da}, attrs=global_attrs).to_zarr(
                f"yearly_metrics_zarrs/AORC_apparent_temp_mean_{year}.zarr", zarr_format=2
            )
        da = aorc_atemp.resample(time="1D").min().chunk('auto')
        da.attrs = var_attrs
        if not isdir(f"yearly_metrics_zarrs/AORC_apparent_temp_min_{year}.zarr"):
            xr.Dataset(data_vars={"apparent_temp_min": da}, attrs=global_attrs).to_zarr(
                f"yearly_metrics_zarrs/AORC_apparent_temp_min_{year}.zarr", zarr_format=2
            )
        da = aorc_atemp.resample(time="1D").max().chunk('auto')
        da.attrs = var_attrs
        if not isdir(f"yearly_metrics_zarrs/AORC_apparent_temp_max_{year}.zarr"):
            xr.Dataset(data_vars={"apparent_temp_max": da}, attrs=global_attrs).to_zarr(
                f"yearly_metrics_zarrs/AORC_apparent_temp_max_{year}.zarr", zarr_format=2
            )
        del aorc_atemp


        print(f"{datetime.now().timestamp()} [compute] Calculating humidex metrics for {year}.")
        aorc_hdex = xr.apply_ufunc(
            metrics.get_aorc_humidex,
            aorc_ds["TMP_2maboveground"],
            aorc_ds["SPFH_2maboveground"],
            dask="parallelized",
            output_dtypes=[float]
        ).chunk('auto')
        da = aorc_hdex.resample(time="1D").mean().chunk('auto')
        da.attrs = var_attrs
        if not isdir(f"yearly_metrics_zarrs/AORC_humidex_mean_{year}.zarr"):
            xr.Dataset(data_vars={"humidex_mean": da}, attrs=global_attrs).to_zarr(
                f"yearly_metrics_zarrs/AORC_humidex_mean_{year}.zarr", zarr_format=2
            )
        da = aorc_hdex.resample(time="1D").min().chunk('auto')
        da.attrs = var_attrs
        if not isdir(f"yearly_metrics_zarrs/AORC_humidex_min_{year}.zarr"):
            xr.Dataset(data_vars={"humidex_min": da}, attrs=global_attrs).to_zarr(
                f"yearly_metrics_zarrs/AORC_humidex_min_{year}.zarr", zarr_format=2
            )
        da = aorc_hdex.resample(time="1D").max().chunk('auto')
        da.attrs = var_attrs
        if not isdir(f"yearly_metrics_zarrs/AORC_humidex_max_{year}.zarr"):
            xr.Dataset(data_vars={"humidex_max": da}, attrs=global_attrs).to_zarr(
                f"yearly_metrics_zarrs/AORC_humidex_max_{year}.zarr", zarr_format=2
            )
        del aorc_hdex


        print(f"{datetime.now().timestamp()} [compute] Calculating simple wet-bulb globe temperature metrics for {year}.")
        aorc_swbgt = xr.apply_ufunc(
            metrics.get_aorc_swbgt,
            aorc_ds["TMP_2maboveground"],
            aorc_ds["SPFH_2maboveground"],
            dask="parallelized",
            output_dtypes=[float]
        ).chunk('auto')
        da = aorc_swbgt.resample(time="1D").mean().chunk('auto')
        da.attrs = var_attrs
        if not isdir(f"yearly_metrics_zarrs/AORC_swbgt_mean_{year}.zarr"):
            xr.Dataset(data_vars={"swbgt_mean": da}, attrs=global_attrs).to_zarr(
                f"yearly_metrics_zarrs/AORC_swbgt_mean_{year}.zarr", zarr_format=2
            )
        da = aorc_swbgt.resample(time="1D").min().chunk('auto')
        da.attrs = var_attrs
        if not isdir(f"yearly_metrics_zarrs/AORC_swbgt_min_{year}.zarr"):
            xr.Dataset(data_vars={"swbgt_min": da}, attrs=global_attrs).to_zarr(
                f"yearly_metrics_zarrs/AORC_swbgt_min_{year}.zarr", zarr_format=2
            )
        da = aorc_swbgt.resample(time="1D").max().chunk('auto')
        da.attrs = var_attrs
        if not isdir(f"yearly_metrics_zarrs/AORC_swbgt_max_{year}.zarr"):
            xr.Dataset(data_vars={"swbgt_max": da}, attrs=global_attrs).to_zarr(
                f"yearly_metrics_zarrs/AORC_swbgt_max_{year}.zarr", zarr_format=2
            )
        del aorc_swbgt

        
        print(f"{datetime.now().timestamp()} [compute] Calculating daily 2-meter temperature metrics for {year}.")
        aorc_tas = xr.apply_ufunc(
            metrics.celsius_to_fahrenheit,
            (aorc_ds["TMP_2maboveground"]-273.15),
            dask="parallelized",
            output_dtypes=[float]
        ).chunk('auto')
        da = aorc_tas.resample(time="1D").mean().chunk('auto')
        da.attrs = var_attrs
        if not isdir(f"yearly_metrics_zarrs/AORC_tas_mean_{year}.zarr"):
            xr.Dataset(data_vars={"tas_mean": da}, attrs=global_attrs).to_zarr(
                f"yearly_metrics_zarrs/AORC_tas_mean_{year}.zarr", zarr_format=2
            )
        da = aorc_tas.resample(time="1D").min().chunk('auto')
        da.attrs = var_attrs
        if not isdir(f"yearly_metrics_zarrs/AORC_tas_min_{year}.zarr"):
            xr.Dataset(data_vars={"tas_min": da}, attrs=global_attrs).to_zarr(
                f"yearly_metrics_zarrs/AORC_tas_min_{year}.zarr", zarr_format=2
            )
        da = aorc_tas.resample(time="1D").max().chunk('auto')
        da.attrs = var_attrs
        if not isdir(f"yearly_metrics_zarrs/AORC_tas_max_{year}.zarr"):
            xr.Dataset(data_vars={"tas_max": da}, attrs=global_attrs).to_zarr(
                f"yearly_metrics_zarrs/AORC_tas_max_{year}.zarr", zarr_format=2
            )


        print(f"{datetime.now().timestamp()} [compute] Calculating daily wet-bulb temperature metrics for {year}.")
        aorc_wbt = xr.apply_ufunc(
            metrics.get_aorc_wbt,
            aorc_ds["TMP_2maboveground"],
            aorc_ds["SPFH_2maboveground"],
            aorc_ds["PRES_surface"],
            dask="parallelized",
            output_dtypes=[float]
        ).chunk('auto')
        da = aorc_wbt.resample(time="1D").mean().chunk('auto')
        da.attrs = var_attrs
        if not isdir(f"yearly_metrics_zarrs/AORC_wbt_mean_{year}.zarr"):
            xr.Dataset(data_vars={"wbt_mean": da}, attrs=global_attrs).to_zarr(
                f"yearly_metrics_zarrs/AORC_wbt_mean_{year}.zarr", zarr_format=2
            )
        da = aorc_wbt.resample(time="1D").min().chunk('auto')
        da.attrs = var_attrs
        if not isdir(f"yearly_metrics_zarrs/AORC_wbt_min_{year}.zarr"):
            xr.Dataset(data_vars={"wbt_min": da}, attrs=global_attrs).to_zarr(
                f"yearly_metrics_zarrs/AORC_wbt_min_{year}.zarr", zarr_format=2
            )
        da = aorc_wbt.resample(time="1D").max().chunk('auto')
        da.attrs = var_attrs
        if not isdir(f"yearly_metrics_zarrs/AORC_wbt_max_{year}.zarr"):
            xr.Dataset(data_vars={"wbt_max": da}, attrs=global_attrs).to_zarr(
                f"yearly_metrics_zarrs/AORC_wbt_max_{year}.zarr", zarr_format=2
            )
    
    print("[final] Computations finished, shutting down Dask cluster.")

    client.shutdown()
