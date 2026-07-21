# Tests

Unit tests for the heat-stress metrics in [`metrics.py`](../metrics.py).

Expected values are taken from published references (NWS heat-index chart and
Rothfusz regression, standard Tetens saturation-vapour-pressure tables, the
Steadman apparent-temperature / Canadian humidex / ACSM simplified-WBGT
definitions, and Stull's 2011 wet-bulb reference) so that a formula transcription
error is caught rather than masked.

## Running

`pytest` is not part of the runtime image, so install it on top of the project
deps. From the project root:

```bash
docker run --rm -v "$PWD":/project -w /project aorc \
    bash -c "pip install -q pytest && python -m pytest tests/ -v"
```

(Or `pip install -r requirements-dev.txt` once in a persistent environment.)
