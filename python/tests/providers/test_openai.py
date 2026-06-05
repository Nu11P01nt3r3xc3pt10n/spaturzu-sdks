import pytest
import spaturzu
pytest.importorskip("openai")
from openai import OpenAI as RealOpenAI, AsyncOpenAI as RealAsyncOpenAI
from spaturzu.openai import OpenAI, AsyncOpenAI


@pytest.fixture(autouse=True)
def _reset():
    spaturzu.__reset_default_for_tests()
    yield
    spaturzu.__reset_default_for_tests()


def test_openai_dropin_instruments_and_preserves_call_surface():
    client = OpenAI(api_key="sk-test")
    assert callable(getattr(client, "with_agent", None))
    assert callable(client.chat.completions.create)


def test_async_openai_dropin():
    client = AsyncOpenAI(api_key="sk-test")
    assert callable(getattr(client, "with_agent", None))
    assert callable(client.chat.completions.create)
