"""Unit tests for argument parsing.

No Dask cluster is started here; `main` is not exercised because it opens a
network connection to S3.
"""
import pytest

from aorc_heat import cli, pipeline


BASE_ARGUMENTS = ["--output-dir", "/tmp/out", "--cores", "8", "--memory-limit", "10GB"]


def test_all_selects_every_registered_metric():
    arguments = cli.build_parser().parse_args(BASE_ARGUMENTS + ["--all"])
    assert cli.selected_metrics(arguments) == sorted(pipeline.METRICS)


def test_metrics_selects_only_what_was_asked_for():
    arguments = cli.build_parser().parse_args(
        BASE_ARGUMENTS + ["--metrics", "humidex", "heat_index"]
    )
    assert cli.selected_metrics(arguments) == ["humidex", "heat_index"]


def test_metric_selection_is_required():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(BASE_ARGUMENTS)


def test_all_and_metrics_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(BASE_ARGUMENTS + ["--all", "--metrics", "humidex"])


def test_unknown_metric_is_rejected():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(BASE_ARGUMENTS + ["--metrics", "not_a_metric"])


def test_year_range_defaults_to_the_full_aorc_record():
    arguments = cli.build_parser().parse_args(BASE_ARGUMENTS + ["--all"])
    assert arguments.start_year == cli.DEFAULT_START_YEAR
    assert arguments.end_year == cli.DEFAULT_END_YEAR


def test_year_range_can_be_narrowed():
    arguments = cli.build_parser().parse_args(
        BASE_ARGUMENTS + ["--all", "--start-year", "2010", "--end-year", "2011"]
    )
    assert (arguments.start_year, arguments.end_year) == (2010, 2011)


def test_inverted_year_range_is_rejected():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            BASE_ARGUMENTS + ["--all", "--start-year", "2011", "--end-year", "2010"]
        )


def test_zero_cores_is_rejected():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            ["--output-dir", "/tmp/out", "--cores", "0", "--memory-limit", "10GB", "--all"]
        )


def test_negative_cores_is_rejected():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            ["--output-dir", "/tmp/out", "--cores", "-1", "--memory-limit", "10GB", "--all"]
        )


def test_positive_cores_is_accepted():
    arguments = cli.build_parser().parse_args(
        BASE_ARGUMENTS + ["--all"]
    )
    assert arguments.cores == 8


def test_dashboard_address_defaults_to_an_ephemeral_port():
    arguments = cli.build_parser().parse_args(BASE_ARGUMENTS + ["--all"])
    assert arguments.dashboard_address == cli.DEFAULT_DASHBOARD_ADDRESS == ":0"


def test_dashboard_address_can_be_set_explicitly():
    arguments = cli.build_parser().parse_args(
        BASE_ARGUMENTS + ["--all", "--dashboard-address", ":8787"]
    )
    assert arguments.dashboard_address == ":8787"


@pytest.mark.parametrize(
    "incomplete_arguments",
    [
        ["--cores", "8", "--memory-limit", "10GB"],               # no --output-dir
        ["--output-dir", "/tmp/out", "--memory-limit", "10GB"],   # no --cores
        ["--output-dir", "/tmp/out", "--cores", "8"],             # no --memory-limit
    ],
)
def test_required_arguments(incomplete_arguments):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(incomplete_arguments + ["--all"])
