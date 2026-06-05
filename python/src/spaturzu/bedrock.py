"""Drop-in replacement for boto3's bedrock-runtime client. Swap::

    - import boto3
    - client = boto3.client("bedrock-runtime", region_name="us-east-1")
    + from spaturzu.bedrock import BedrockRuntime
    + client = BedrockRuntime(region_name="us-east-1")

Targets the Converse API (client.converse / converse_stream); v1 is sync-only.
"""
from __future__ import annotations
from typing import Any, Optional

import boto3
from . import get_default_spaturzu


def BedrockRuntime(*args: Any, spaturzu: Optional[dict] = None, **kwargs: Any) -> Any:
    return get_default_spaturzu().wrap_bedrock(
        boto3.client("bedrock-runtime", *args, **kwargs), **(spaturzu or {})
    )
