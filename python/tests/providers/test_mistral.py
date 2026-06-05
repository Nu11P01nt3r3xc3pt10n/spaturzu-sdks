import pytest
import spaturzu
pytest.importorskip("mistralai")
from spaturzu.mistral import Mistral


@pytest.fixture(autouse=True)
def _reset():
    spaturzu.__reset_default_for_tests()
    yield
    spaturzu.__reset_default_for_tests()


def test_mistral_dropin_instruments():
    client = Mistral(api_key="test")
    assert callable(getattr(client, "with_agent", None))
    assert callable(client.chat.complete)
    assert callable(client.chat.stream)
