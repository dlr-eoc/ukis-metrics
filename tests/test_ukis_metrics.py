from pathlib import Path

import toml
from ukis_metrics import __version__


def test_version():
    with open(Path.cwd().joinpath("pyproject.toml")) as f:
        assert __version__ == toml.load(f).get("tool").get("poetry").get("version")
