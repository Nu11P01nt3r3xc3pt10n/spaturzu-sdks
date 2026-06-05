import pytest
import spaturzu
pytest.importorskip("anthropic")
from anthropic import Anthropic as RealAnthropic
from spaturzu.anthropic import Anthropic, AsyncAnthropic


@pytest.fixture(autouse=True)
def _reset():
    spaturzu.__reset_default_for_tests()
    yield
    spaturzu.__reset_default_for_tests()


def test_anthropic_dropin_instruments():
    client = Anthropic(api_key="sk-ant-test")
    assert callable(getattr(client, "with_agent", None))
    assert callable(client.messages.create)


def test_async_anthropic_dropin():
    client = AsyncAnthropic(api_key="sk-ant-test")
    assert callable(getattr(client, "with_agent", None))
    assert callable(client.messages.create)
