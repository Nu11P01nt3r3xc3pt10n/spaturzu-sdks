import pytest
import spaturzu
pytest.importorskip("google.genai")
from google.genai import Client as RealClient
from spaturzu.google import Client


@pytest.fixture(autouse=True)
def _reset():
    spaturzu.__reset_default_for_tests()
    yield
    spaturzu.__reset_default_for_tests()


def test_google_dropin_instruments():
    client = Client(api_key="test")
    assert callable(getattr(client, "with_agent", None))
    assert callable(client.models.generate_content)
