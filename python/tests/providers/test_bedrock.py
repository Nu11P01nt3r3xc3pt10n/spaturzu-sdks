import pytest
import spaturzu
pytest.importorskip("boto3")
from spaturzu.bedrock import BedrockRuntime


@pytest.fixture(autouse=True)
def _reset():
    spaturzu.__reset_default_for_tests()
    yield
    spaturzu.__reset_default_for_tests()


def test_bedrock_dropin_instruments():
    client = BedrockRuntime(region_name="us-east-1")
    assert callable(getattr(client, "with_agent", None))
    assert callable(client.converse)
    assert callable(client.converse_stream)
