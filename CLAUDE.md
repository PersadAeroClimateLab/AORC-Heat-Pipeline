# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

**There is no local Python environment with the dependencies. Everything runs in Docker.** Invoking `python` or `pytest` on the host fails with `ModuleNotFoundError`.

```bash
docker build -t aorc .                                                       # once, and after dependency changes
docker run --rm -v "$PWD":/project -w /project aorc python -m pytest tests/ -q
docker run --rm -v "$PWD":/project -w /project aorc python -m pytest tests/test_core.py::test_humidex -v
docker run --rm -v "$PWD":/project -w /project aorc python -m pytest tests/ -k "wet_bulb" -v
docker run --rm -v "$PWD":/project -w /project aorc aorc-heat --help
```

The package is installed editable at image build time, so the bind mount means source edits take effect without rebuilding.

Numba caches compiled kernels to `.nbi`/`.nbc` beside the source. Those files end up **root-owned** from container runs, so host-side `find -delete` fails silently. If runtime behaviour contradicts the source you are reading, clear them as root:

```bash
docker run --rm -v "$PWD":/project -w /project aorc \
    bash -c "find /project -name '*.nbi' -delete; find /project -name '*.nbc' -delete"
```

## Architecture

Four modules with boundaries that are enforced by convention, not by tooling:

- **`core.py`** — scalar science in literature units (°C, hPa, kg/kg, %). Imports only numpy and numba. Knows nothing about AORC. Every function is independently checkable against published values.
- **`pipeline.py`** — translates AORC's units into core's, applies the kernels, reduces 24 hourly values to daily min/mean/max, and owns the zarr store. Contains **no science**.
- **`mask.py`** — builds and caches the Texas boundary mask, and decides the region's index slices.
- **`cli.py`** — argparse plus Dask cluster lifecycle. Nothing else.

A formula in `pipeline.py` is a bug: it cannot be tested against literature values there. A reference to `TMP_2maboveground` in `core.py` is the same bug in the other direction.

### The NaN contract spans three modules

`prepare_dataset` masks with `.where(region.mask)`, so **roughly half the cells reaching the kernels are NaN by design**. Three consequences that are invisible from any single file:

1. A `core.py` function that branches on a possibly-NaN value must use `NAN_SAFE_FASTMATH`, not `fastmath=True`. The latter sets LLVM's `nnan`, which folds NaN guards away and may treat an `if/elif` as exhaustive. This produced a production-corrupting bug: `wet_bulb_temperature`'s clamp collapsed into an unconditional `else` and turned every out-of-region cell into a finite `T − 40`.
2. Above the SIMD width (8 float32 lanes) LLVM if-converts branches into selects and evaluates both sides for all lanes, so NaN lanes raise IEEE-754's invalid-operation flag even though their results are discarded. `pipeline._apply` wraps every kernel in `errstate(invalid="ignore")` for this reason. That suppression is only safe because `test_finite_inputs_never_produce_non_finite_outputs` would catch a genuine bad computation — if that test fails, remove the suppression rather than adjusting it.
3. Masked cells are the *expensive* ones in iterative kernels unless guarded. `wet_bulb_temperature` early-returns NaN because without it a masked cell ran the full 50-iteration cap at ~14× the cost of a real cell.

### Chunk alignment spans mask.py and pipeline.py

Zarr chunks are atomic, so a crop starting mid-chunk downloads the whole chunk anyway. `mask.crop_to_bounding_box(mask, alignment=...)` therefore snaps the region **outward** to the source store's native chunk boundaries (AORC: `time=144, latitude=128, longitude=256`). This keeps cells already being transferred instead of discarding them, and makes the cropped array's first chunk a full one.

That single decision is what lets `prepare_dataset` and `write_block` skip spatial rechunking entirely: source, dask graph, and output store share one chunk grid. The Texas bbox starts at latitude 701 and longitude 2803 — both prime — so without snapping the first chunk is a partial remainder and the first region write fails zarr's alignment check after a full year of compute.

Time is left native too. AORC's 144-hour chunk is exactly six midnight-aligned days, so a daily `resample` group never crosses a boundary and `prepare_dataset` does no time rechunk either. Splitting to 24 hours (an earlier version did) inflated the per-year task graph ~6x for bit-identical output — the reduction just regroups what the split divided.

### Writes must stay sequential

`write_block` passes `safe_chunks=False`, because year lengths alternate 365/366 and no fixed time chunk divides every year boundary. It is safe only because of two things that must both hold:

- Within a write, `chunk_sizes_aligned_to_store` splits the block at the store's own boundaries, so no two dask tasks target one chunk.
- Across writes, adjacent years share the chunk at their boundary and rely on zarr read-modify-writing it. **Parallelising `run`'s year loop would silently corrupt one day at each year boundary.**

### Store semantics

One store, `aorc_heat_metrics.zarr`, holding only the metrics actually computed — not NaN-padded placeholders. Re-running with an added metric appends its variables and leaves existing ones bit-identical.

Completion is recorded per year in the `completed_years` store attribute *after* each successful write. `pending_metrics` treats a metric as done only when the record covers every requested year, so an interrupted run recomputes instead of reporting success over NaN. Recovery granularity is the metric, not the year.

### Source data gaps and what NaN means in the output

The upstream NOAA AORC archive itself has gaps, measured directly rather than assumed: each is exactly one native 128×256 chunk tile, out for a contiguous run of whole hours, a different tile on each occasion (`(10,4)` in Oct 1979, `(7,3)` in Jun 1981), and — on every occasion checked — every affected cell is missing for all 24 hours of the day, never partially. Separately, AORC's archive starts 1 February 1979; January 1979 does not exist upstream, and the pipeline writes that short year at store index 31 without padding a fake January in front of it.

Before this was handled, that meant NaN in the output was overloaded three ways — out-of-region, source gap, and never-computed were indistinguishable — and a day covered by only part of its 24 hours would silently reduce over the surviving hours and write an ordinary-looking, silently biased value (a daily max missing the afternoon reads low with nothing recording it).

Every metric now also emits `{metric}_valid_hours` (uint8, 0–24): the count of that metric's own hourly values that were non-NaN, from that metric's own hourly series so metrics with different inputs can disagree about which days are incomplete. `MINIMUM_VALID_HOURS` (in `pipeline.py`) is the coverage floor below which a day's min/mean/max are suppressed to NaN — strict (24) by default, because every gap measured so far is whole-day so a looser floor changes nothing yet, and because daily min/max are order statistics that a partial day biases rather than just makes noisier. `{metric}_valid_hours` itself is **never** suppressed, even on a day whose statistics were: that is what lets NaN in the store mean, unambiguously, "no value," while `valid_hours` on that same cell-day still answers why — 0 for out-of-region or a fully-missing day, a number below `MINIMUM_VALID_HOURS` for a partial one, 24 for a real value.

## Conventions

- All metric outputs are degrees Fahrenheit (`units: "deg_F"`).
- `core.py` computes in float32; coerce each argument with `np.float32(...)` first. Test tolerances of `rel=1e-5` / `abs=1e-4` reflect that — tightening them means something is circular.
- Function names describe what is computed; parameter names carry the unit (`vapor_pressure_hpa`, `air_temperature_celsius`). Putting the unit in the function name collides with other functions' parameter names.
- `core.vapor_pressure` takes ambient pressure as a second argument. It does not assume 1013.25 hPa, so every metric requires `PRES_surface`.
- `core.heat_index` takes Fahrenheit **and returns Fahrenheit**. Converting its result again is a silent, plausible-looking bug.

## Skills

Task-specific guidance lives in `.claude/skills/`: `adding-a-core-function`, `adding-a-pipeline-metric`, `writing-core-unit-tests`, `testing-pipeline-metrics`. Each is written against failures actually observed in this repository.

## Design history

`docs/superpowers/specs/` and `docs/superpowers/plans/` record the refactor's design and the decisions implementation revised. The plan is marked executed and notes where it disagrees with the code; **where they conflict, the code is right.**

## Out of scope

The Lu & Romps (2022) heat index and its lookup tables were deliberately removed and will be revisited separately. Do not reintroduce them. They are recoverable from git history at `0bb254e`.
