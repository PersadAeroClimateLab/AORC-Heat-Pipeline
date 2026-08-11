# Tests

Unit tests for the heat-stress formulations in
[`src/aorc_heat/core.py`](../src/aorc_heat/core.py), the region masking in
[`src/aorc_heat/mask.py`](../src/aorc_heat/mask.py), the dataset orchestration in
[`src/aorc_heat/pipeline.py`](../src/aorc_heat/pipeline.py), and argument parsing
in [`src/aorc_heat/cli.py`](../src/aorc_heat/cli.py).

Expected values in `test_core.py` are taken from published references — the NWS
heat-index chart and Rothfusz regression, standard Tetens saturation-vapour-
pressure tables, the Steadman apparent-temperature, Canadian humidex and ACSM
simplified-WBGT definitions, and Stull's 2011 wet-bulb reference — so that a
formula transcription error is caught rather than reproduced. Tolerances are
loose enough to absorb float32 rounding, which the kernels use by design.

`test_pipeline.py` runs against an in-memory synthetic dataset and a `tmp_path`
zarr store. Nothing in the suite touches S3 or the network.

## Running

```bash
docker build -t aorc .
docker run --rm -v "$PWD":/project -w /project aorc python -m pytest tests/ -v
```
