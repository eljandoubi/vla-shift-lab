import pytest

from vlashift import runtime
from vlashift.config import KNOWN_ENV


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Study variables of the shell must not reach the tests, nor leak state between them."""
    for name in KNOWN_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(runtime, "_runtime", None)
