from __future__ import annotations

import pytest
from tiny_stack import build_stack


@pytest.fixture
def stack():
    model, probe, steering, hooked = build_stack()
    yield model, probe, steering, hooked
    hooked.close()
