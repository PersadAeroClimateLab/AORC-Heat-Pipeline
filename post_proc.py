from os import listdir
import xarray
from dask.distributed import LocalCluster, Client


if __name__ == "__main__":    
    cluster = LocalCluster(n_workers=90, threads_per_worker=1, memory_limit="4GB", dashboard_address=":8002")
    client = cluster.get_client()
    
    dataset_names = {}
    
    for file_name in listdir("./yearly_metrics_zarrs"):
        if ".zarr" in file_name:
            aorc, var_name, stat, yr = file_name.split("_")
            tag = f"AORC_{var_name}_{stat}.zarr"
            if tag in dataset_names:
                dataset_names[tag].append(f"yearly_metrics_zarrs/{file_name}")
            else:
                dataset_names[tag] = [f"yearly_metrics_zarrs/{file_name}"]
    
    for output_path in dataset_names:
        print(output_path)
        dataset_zarrs = []
        dataset_names[output_path].sort()
        for zarr_input_path in dataset_names[output_path]:
            dataset_zarrs.append(xarray.open_zarr(zarr_input_path).convert_calendar("standard", use_cftime=True))
        ds = xarray.concat(dataset_zarrs, dim="time")
        ds.to_zarr(f"output_zarrs/{output_path}", mode="w", zarr_format=2, consolidated=True, write_empty_chunks=True, align_chunks=True)

    client.shutdown()