"""Confirms the package is installed and importable in the test environment."""


def test_package_imports():
    import aorc_heat

    assert aorc_heat.__all__ == ["core", "pipeline", "cli", "mask"]
