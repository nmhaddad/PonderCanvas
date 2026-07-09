import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("PONDERCANVAS_RUN_LIVE_TESTS") == "1":
        return
    skip_live = pytest.mark.skip(
        reason="live test: set PONDERCANVAS_RUN_LIVE_TESTS=1 (and real API keys) to run"
    )
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
