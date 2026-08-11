# AORC-Heat-Pipeline

Daily minimum, mean, and maximum heat metrics derived from the
[NOAA AORC](https://registry.opendata.aws/noaa-nws-aorc) 1 km hourly archive,
masked to the state of Texas.

## Metrics

| Name | Description | Units |
| --- | --- | --- |
| `heat_index` | NWS Rothfusz regression | °F |
| `apparent_temperature` | Steadman apparent temperature | °F |
| `humidex` | Canadian humidex | °F |
| `simplified_wbgt` | ACSM simplified wet-bulb globe temperature | °F |
| `wet_bulb_temperature` | Thermodynamic wet-bulb temperature | °F |

## Structure

- `src/aorc_heat/core.py` — scalar scientific formulations in literature units,
  compiled with numba and computed in float32. Knows nothing about AORC.
- `src/aorc_heat/pipeline.py` — reads AORC from S3, builds shared intermediate
  quantities once, applies the core functions, reduces each 24-hour chunk to
  daily statistics, and writes the result into a single zarr store.
- `src/aorc_heat/cli.py` — argument parsing and Dask cluster lifecycle.
- `src/aorc_heat/mask.py` — generates and caches the Texas boundary mask.

## Running

Build the image once:

```bash
docker build -t aorc .
```

Compute every metric across the full record:

```bash
docker run --rm -v "$PWD":/project -w /project aorc \
    aorc-heat --all --output-dir output_zarrs --cores 80 --memory-limit 10GB
```

Compute a subset, or a shorter period:

```bash
docker run --rm -v "$PWD":/project -w /project aorc \
    aorc-heat --metrics heat_index humidex \
              --output-dir output_zarrs --cores 80 --memory-limit 10GB \
              --start-year 2020 --end-year 2021
```

Metrics already present in the store are skipped rather than recomputed, so a
later run that adds `--metrics wet_bulb_temperature` appends those variables
without touching what is already there. The year range must match the range the
store was built for.

## Output

A single store at `<output-dir>/aorc_heat_metrics.zarr` with dimensions
`(time, latitude, longitude)`, one entry per day, and a
`{metric}_{min,mean,max}` variable for each computed metric. Cells outside the
Texas boundary are NaN.

## Tests

```bash
docker run --rm -v "$PWD":/project -w /project aorc python -m pytest tests/ -v
```
