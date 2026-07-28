"""Command-line front end for the AORC heat pipeline.

Owns argument parsing and the Dask cluster lifecycle, and nothing else. All
science lives in `core` and all dataset handling in `pipeline`.
"""

import argparse
from pathlib import Path

from aorc_heat import pipeline

DEFAULT_START_YEAR = 1979
DEFAULT_END_YEAR = 2024
DASHBOARD_ADDRESS = ":8002"


def _positive_integer(value):
    """Validate that cores is a positive integer.

    :param value: The value to validate
    :return: The parsed integer
    :raises argparse.ArgumentTypeError: If the value is not a positive integer
    """
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(
            f"cores must be at least 1, got {ivalue}"
        )
    return ivalue


class _YearRangeParser(argparse.ArgumentParser):
    """Parser that rejects an inverted year range at parse time.

    The check lives here rather than in `main` so that it applies to every
    caller, including the tests, and so a bad range fails before any cluster is
    started.
    """

    def parse_args(self, args=None, namespace=None):
        arguments = super().parse_args(args, namespace)
        if arguments.end_year < arguments.start_year:
            self.error(
                f"--end-year {arguments.end_year} precedes "
                f"--start-year {arguments.start_year}"
            )
        return arguments


def build_parser():
    """Build the argument parser.

    :return: A configured ArgumentParser
    """
    parser = _YearRangeParser(
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
        help="Number of Dask worker processes, one thread each.",
    )
    parser.add_argument(
        "--memory-limit",
        required=True,
        help="Memory limit per Dask worker, for example '10GB'.",
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

    cluster = LocalCluster(
        n_workers=arguments.cores,
        threads_per_worker=1,
        memory_limit=arguments.memory_limit,
        dashboard_address=DASHBOARD_ADDRESS,
    )
    try:
        print(cluster.get_client())
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
