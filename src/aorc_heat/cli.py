"""Command-line front end for the AORC heat pipeline.

Owns argument parsing and the Dask cluster lifecycle, and nothing else. All
science lives in `core` and all dataset handling in `pipeline`.
"""

import argparse
from pathlib import Path

from aorc_heat import pipeline

DEFAULT_START_YEAR = 1979
DEFAULT_END_YEAR = 2024
DEFAULT_DASHBOARD_ADDRESS = ":0"

#: Threads in each worker process, when `--threads-per-worker` is not given.
#:
#: 1 preserves the historical shape exactly -- `--cores N` has always meant N
#: single-threaded processes, and `--memory-limit` is per *process*, so
#: defaulting to anything else would quietly divide the cluster's total memory
#: by that factor for every existing job script. Raising it is therefore an
#: explicit choice the caller makes together with a matching `--memory-limit`.
#:
#: It is, however, usually the wrong value to leave it at. See the
#: `--threads-per-worker` help text.
DEFAULT_THREADS_PER_WORKER = 1


def _positive_integer(value):
    """Validate that an argument is a positive integer.

    :param value: The value to validate
    :return: The parsed integer
    :raises argparse.ArgumentTypeError: If the value is not a positive integer
    """
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {ivalue}")
    return ivalue


def cluster_shape(arguments):
    """Worker processes and threads-per-worker for the requested core count.

    `--cores` is the total number of worker threads the cluster will run, so
    the process count is that divided by the threads each process gets. Kept
    out of `main` so it is reachable from a test without starting a cluster.

    :param arguments: Parsed arguments carrying `cores` and `threads_per_worker`
    :return: (worker processes, threads per worker)
    """
    return arguments.cores // arguments.threads_per_worker, arguments.threads_per_worker


class _ValidatingParser(argparse.ArgumentParser):
    """Parser that rejects incoherent argument *combinations* at parse time.

    argparse validates each option on its own; these two checks are about how
    options relate. Both live here rather than in `main` so that they apply to
    every caller, including the tests, and so a bad invocation fails before any
    cluster is started -- which for a 46-year job means before an hour of S3
    reads rather than after.
    """

    def parse_args(self, args=None, namespace=None):
        arguments = super().parse_args(args, namespace)
        if arguments.end_year < arguments.start_year:
            self.error(
                f"--end-year {arguments.end_year} precedes "
                f"--start-year {arguments.start_year}"
            )
        # Floor division would silently drop the remainder -- `--cores 40
        # --threads-per-worker 3` would start 13 processes running 39 threads,
        # one fewer than asked for, with nothing in the log to say so. On an
        # HPC allocation the core count is exact, so this is worth rejecting.
        if arguments.cores % arguments.threads_per_worker:
            self.error(
                f"--cores {arguments.cores} is not a multiple of "
                f"--threads-per-worker {arguments.threads_per_worker}; "
                f"that would run "
                f"{(arguments.cores // arguments.threads_per_worker) * arguments.threads_per_worker}"
                f" threads rather than {arguments.cores}"
            )
        return arguments


def build_parser():
    """Build the argument parser.

    :return: A configured ArgumentParser
    """
    parser = _ValidatingParser(
        prog="aorc-heat",
        description=(
            "Compute daily minimum, mean, and maximum heat metrics from the NOAA "
            "AORC hourly archive and write them to a zarr store. Metrics already "
            "present in the store are skipped, so a store can be extended one "
            "metric at a time."
        ),
    )

    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--all",
        action="store_true",
        help="Compute every available metric.",
    )
    selection.add_argument(
        "--metrics",
        nargs="+",
        choices=sorted(pipeline.METRICS),
        metavar="METRIC",
        help=f"Metrics to compute. One or more of: {', '.join(sorted(pipeline.METRICS))}.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory the output zarr store is written to.",
    )
    parser.add_argument(
        "--cores",
        required=True,
        type=_positive_integer,
        help=(
            "Total Dask worker threads across the cluster. Divided by "
            "--threads-per-worker to give the number of worker processes."
        ),
    )
    parser.add_argument(
        "--threads-per-worker",
        type=_positive_integer,
        default=DEFAULT_THREADS_PER_WORKER,
        help=(
            "Threads in each worker process. Default "
            f"{DEFAULT_THREADS_PER_WORKER}, which reproduces the historical "
            "one-process-per-core shape, but a higher value is usually right: "
            "the numba kernels release the GIL (measured ~5.6x on 8 threads), "
            "so threads are real parallelism, and a process costs ~390 MB of "
            "interpreter, numba and JIT state before it holds any data. Five "
            "threads per process turns 40 processes into 8 and recovers "
            "~12 GB, while letting tasks on one worker share arrays by "
            "pointer instead of serialising them. Scale --memory-limit up by "
            "the same factor."
        ),
    )
    parser.add_argument(
        "--memory-limit",
        required=True,
        help=(
            "Memory limit per worker *process*, for example '10GB'. Because "
            "raising --threads-per-worker lowers the process count, it also "
            "lowers the cluster total unless this is raised to match. The "
            "shape and the resulting total are printed at startup."
        ),
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=DEFAULT_START_YEAR,
        help=f"First year to compute, inclusive. Default {DEFAULT_START_YEAR}.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=DEFAULT_END_YEAR,
        help=f"Last year to compute, inclusive. Default {DEFAULT_END_YEAR}.",
    )
    parser.add_argument(
        "--dashboard-address",
        default=DEFAULT_DASHBOARD_ADDRESS,
        help=(
            "Address the Dask dashboard binds to, for example ':8787'. Default "
            f"'{DEFAULT_DASHBOARD_ADDRESS}' picks an ephemeral free port so that "
            "concurrent runs on one node cannot collide; the chosen port is "
            "printed at startup."
        ),
    )

    return parser


def selected_metrics(arguments):
    """Resolve `--all` or `--metrics` into a concrete metric list.

    :param arguments: Parsed arguments
    :return: Metric names to compute
    """
    if arguments.all:
        return sorted(pipeline.METRICS)
    return list(arguments.metrics)


def main(argv=None):
    """Parse arguments, start a Dask cluster, and run the pipeline.

    :param argv: Argument list, defaulting to sys.argv[1:]
    :return: Process exit code
    """
    from dask.distributed import LocalCluster

    arguments = build_parser().parse_args(argv)
    metric_names = selected_metrics(arguments)
    workers, threads = cluster_shape(arguments)

    cluster = LocalCluster(
        n_workers=workers,
        threads_per_worker=threads,
        memory_limit=arguments.memory_limit,
        dashboard_address=arguments.dashboard_address,
        host="127.0.0.1",
    )
    try:
        # Printed explicitly because --memory-limit is per process: the same
        # --cores and --memory-limit give a different cluster total at every
        # --threads-per-worker, and that is not something to leave a reader to
        # work out from the Client repr.
        print(
            f"Cluster: {workers} process(es) x {threads} thread(s) = "
            f"{arguments.cores} worker threads, "
            f"{arguments.memory_limit} per process."
        )
        print(cluster.get_client())
        print(f"Dask dashboard: {cluster.dashboard_link}")
        store_path = pipeline.run(
            output_dir=arguments.output_dir,
            metric_names=metric_names,
            start_year=arguments.start_year,
            end_year=arguments.end_year,
        )
        print(f"Metrics written to {store_path}")
    finally:
        cluster.close()
    return 0
