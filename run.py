from shpmask import get_data_array_mask
from numba import set_num_threads
from os.path import isfile
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

    cluster = LocalCluster(n_workers=30, threads_per_worker=2, memory_limit="20GB", dashboard_address=":8002")
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

    print(f"{datetime.now().timestamp()} [init] Reading Zarr metadata for AORC datasets, concatenating over time.")
    datasets = []
    for year in range(1979, 2025):
        datasets.append(xr.open_zarr(fs.get_mapper(f"{s3_base_path}{year}.zarr"), consolidated=True))

    print(f"{datetime.now().timestamp()} [init] Applying Texas mask.")
    aorc_ds = xr.concat(datasets, dim="time").where(texas_mask, drop=True)

    print(aorc_ds["TMP_2maboveground"])

    print(f"{datetime.now().timestamp()} [init] Building metric task graphs.")
    aorc_hi = xr.apply_ufunc(
        metrics.get_aorc_heat_index,
        aorc_ds["TMP_2maboveground"],
        aorc_ds["SPFH_2maboveground"],
        dask="parallelized",
        output_dtypes=[float]
    ).chunk()

    aorc_atemp = xr.apply_ufunc(
        metrics.get_aorc_apparent_temp,
        aorc_ds["TMP_2maboveground"],
        aorc_ds["UGRD_10maboveground"],
        aorc_ds["UGRD_10maboveground"],
        aorc_ds["VGRD_10maboveground"],
        dask="parallelized",
        output_dtypes=[float]
    ).chunk()

    aorc_hdex = xr.apply_ufunc(
        metrics.get_aorc_humidex,
        aorc_ds["TMP_2maboveground"],
        aorc_ds["SPFH_2maboveground"],
        dask="parallelized",
        output_dtypes=[float]
    ).chunk()

    aorc_swbgt = xr.apply_ufunc(
        metrics.get_aorc_swgbt,
        aorc_ds["TMP_2maboveground"],
        aorc_ds["SPFH_2maboveground"],
        dask="parallelized",
        output_dtypes=[float]
    ).chunk()

    print(f"{datetime.now().timestamp()} [init] Aggregating metrics and deriving mean/min/max task graphs.")
    aorc_heat_metrics = xr.Dataset(
        data_vars={
            "heat_index_mean": aorc_hi.resample(time="1D").mean(),
            "heat_index_min": aorc_hi.resample(time="1D").min(),
            "heat_index_max": aorc_hi.resample(time="1D").max(),
            "apparent_temp_mean": aorc_atemp.resample(time="1D").mean(),
            "apparent_temp_min": aorc_atemp.resample(time="1D").min(),
            "apparent_temp_max": aorc_atemp.resample(time="1D").max(),
            "humidex_mean": aorc_hdex.resample(time="1D").mean(),
            "humidex_min": aorc_hdex.resample(time="1D").min(),
            "humidex_max": aorc_hdex.resample(time="1D").max(),
            "swbgt_mean": aorc_swbgt.resample(time="1D").mean(),
            "swbgt_min": aorc_swbgt.resample(time="1D").min(),
            "swbgt_max": aorc_swbgt.resample(time="1D").max()
        },
        attrs={
            "description": "Heat metrics derived from NOAA NWS AORC data provided via AWS S3 Zarr Bucket",
            "source_version": "AORC Version 1.1", 
            "source_url": "https://registry.opendata.aws/noaa-nws-aorc",
        }
    )

    aorc_heat_metrics["heat_index_mean"].attrs = {
        "units": "deg_F",
        "long_name": "Daily Mean Heat Index",
        "source_timestep": "hourly",
        "desc": "National Weather Service regression for heat index. Mean over 24 hours."
    }
    aorc_heat_metrics["heat_index_max"].attrs = {
        "units": "deg_F",
        "long_name": "Daily Maximum Heat Index",
        "source_timestep": "hourly",
        "desc": "National Weather Service regression for heat index. Maximum over 24 hours."
    }
    aorc_heat_metrics["heat_index_min"].attrs = {
        "units": "deg_F",
        "long_name": "Daily Minimum Heat Index",
        "source_timestep": "hourly",
        "desc": "National Weather Service regression for heat index. Minimum over 24 hours."
    }

    aorc_heat_metrics["apparent_temp_mean"].attrs = {
        "units": "deg_F",
        "long_name": "Daily Mean Apparent Temperature",
        "source_timestep": "hourly",
        "desc": "(temp + 0.33*vp - 0.7*sfcwind - 4.00) Mean over 24 hours."
    }
    aorc_heat_metrics["apparent_temp_max"].attrs = {
        "units": "deg_F",
        "long_name": "Daily Maximum Apparent Temperature",
        "source_timestep": "hourly",
        "desc": "(temp + 0.33*vp - 0.7*sfcwind - 4.00) Maximum over 24 hours."
    }
    aorc_heat_metrics["apparent_temp_min"].attrs = {
        "units": "deg_F",
        "long_name": "Daily Minimum Apparent Temperature",
        "source_timestep": "hourly",
        "desc": "(temp + 0.33*vp - 0.7*sfcwind - 4.00) Minimum over 24 hours."
    }

    aorc_heat_metrics["humidex_mean"].attrs = {
        "units": "deg_F",
        "long_name": "Daily Mean Humidex",
        "source_timestep": "hourly",
        "desc": "(temp + 5/9*(6.112 * 10**(7.5*temp/(237.7 + temp)) * vp / 100 - 10)) Mean over 24 hours."
    }
    aorc_heat_metrics["humidex_max"].attrs = {
        "units": "deg_F",
        "long_name": "Daily Maximum Humidex",
        "source_timestep": "hourly",
        "desc": "(temp + 5/9*(6.112 * 10**(7.5*temp/(237.7 + temp)) * vp / 100 - 10)) Maximum over 24 hours."
    }
    aorc_heat_metrics["humidex_min"].attrs = {
        "units": "deg_F",
        "long_name": "Daily Minimum Humidex",
        "source_timestep": "hourly",
        "desc": "(temp + 5/9*(6.112 * 10**(7.5*temp/(237.7 + temp)) * vp / 100 - 10)) Minimum over 24 hours."
    }

    aorc_heat_metrics["swbgt_mean"].attrs = {
        "units": "deg_F",
        "long_name": "Daily Mean Simple Wet-Bulb Globe Temperature",
        "source_timestep": "hourly",
        "desc": "((0.567*(temp) + 0.393*(vp) + 3.94)) Mean over 24 hours."
    }
    aorc_heat_metrics["swbgt_max"].attrs = {
        "units": "deg_F",
        "long_name": "Daily Maximum Simple Wet-Bulb Globe Temperature",
        "source_timestep": "hourly",
        "desc": "((0.567*(temp) + 0.393*(vp) + 3.94)) Maximum over 24 hours."
    }
    aorc_heat_metrics["swbgt_min"].attrs = {
        "units": "deg_F",
        "long_name": "Daily Minimum Simple Wet-Bulb Globe Temperature",
        "source_timestep": "hourly",
        "desc": "((0.567*(temp) + 0.393*(vp) + 3.94)) Minimum over 24 hours."
    }

    aorc_heat_metrics["apparent_temp_mean"].attrs = {
        "units": "deg_F",
        "long_name": "Daily Mean Apparent Temperature",
        "source_timestep": "hourly",
        "desc": "(temp + 0.33*vp - 0.7*sfcwind - 4.00) Mean over 24 hours."
    }
    aorc_heat_metrics["apparent_temp_max"].attrs = {
        "units": "deg_F",
        "long_name": "Daily Maximum Apparent Temperature",
        "source_timestep": "hourly",
        "desc": "(temp + 0.33*vp - 0.7*sfcwind - 4.00) Maximum over 24 hours."
    }
    aorc_heat_metrics["apparent_temp_min"].attrs = {
        "units": "deg_F",
        "long_name": "Daily Minimum Apparent Temperature",
        "source_timestep": "hourly",
        "desc": "(temp + 0.33*vp - 0.7*sfcwind - 4.00) Minimum over 24 hours."
    }

    aorc_heat_metrics["humidex_mean"].attrs = {
        "units": "deg_F",
        "long_name": "Daily Mean Humidex",
        "source_timestep": "hourly",
        "desc": "(temp + 5/9*(6.112 * 10**(7.5*temp/(237.7 + temp)) * vp / 100 - 10)) Mean over 24 hours."
    }
    aorc_heat_metrics["humidex_max"].attrs = {
        "units": "deg_F",
        "long_name": "Daily Maximum Humidex",
        "source_timestep": "hourly",
        "desc": "(temp + 5/9*(6.112 * 10**(7.5*temp/(237.7 + temp)) * vp / 100 - 10)) Maximum over 24 hours."
    }
    aorc_heat_metrics["humidex_min"].attrs = {
        "units": "deg_F",
        "long_name": "Daily Minimum Humidex",
        "source_timestep": "hourly",
        "desc": "(temp + 5/9*(6.112 * 10**(7.5*temp/(237.7 + temp)) * vp / 100 - 10)) Minimum over 24 hours."
    }

    aorc_heat_metrics["swbgt_mean"].attrs = {
        "units": "deg_F",
        "long_name": "Daily Mean Simple Wet-Bulb Globe Temperature",
        "source_timestep": "hourly",
        "desc": "((0.567*(temp) + 0.393*(vp) + 3.94)) Mean over 24 hours."
    }
    aorc_heat_metrics["swbgt_max"].attrs = {
        "units": "deg_F",
        "long_name": "Daily Maximum Simple Wet-Bulb Globe Temperature",
        "source_timestep": "hourly",
        "desc": "((0.567*(temp) + 0.393*(vp) + 3.94)) Maximum over 24 hours."
    }
    aorc_heat_metrics["swbgt_min"].attrs = {
        "units": "deg_F",
        "long_name": "Daily Minimum Simple Wet-Bulb Globe Temperature",
        "source_timestep": "hourly",
        "desc": "((0.567*(temp) + 0.393*(vp) + 3.94)) Minimum over 24 hours."
    }

    aorc_heat_metrics.to_zarr("AORC_heat_metrics.zarr")