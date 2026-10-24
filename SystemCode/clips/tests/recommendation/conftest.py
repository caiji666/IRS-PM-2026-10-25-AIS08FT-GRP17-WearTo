"""Keep the default recommendation tests offline.

The server environment selects the real preview provider (`PREVIEW_PROVIDER=openai`). So that
a developer machine or CI runner that happens to have these variables set can never reach the
paid image API from a default test, every test except those marked `live_preview` runs with the
four variables Student 5 reads (`PREVIEW_ENV_KEYS`) removed; other environment variables are
left alone. A test that needs a configuration sets it explicitly in its own body, which runs
after this fixture. Live tests are additionally gated by RUN_LIVE_PREVIEW_TESTS=1 (see
test_recommendation_live_preview.py).
"""

import pytest

from travel_agent.recommendation.student5.preview_adapter import PREVIEW_ENV_KEYS


@pytest.fixture(autouse=True)
def _no_server_preview_configuration(request, monkeypatch):
    if request.node.get_closest_marker("live_preview") is None:
        for name in PREVIEW_ENV_KEYS:
            monkeypatch.delenv(name, raising=False)
