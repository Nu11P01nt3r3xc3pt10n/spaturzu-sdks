"""Cross-provider translation, organised by source provider.

Mirror of ``sdks/typescript/src/translate/`` (TS). Each source-provider
module owns the translators FROM that provider TO each target provider,
plus the inverse response translators. 5 providers × 4 other-providers
= 20 directional pairs; openai↔anthropic ship today, Plans 3/4/5 add
the remaining 18.

Importing surface is preserved 1:1 with the old single-module shape:
    from spaturzu._translate import chat_to_anthropic_params  # etc.
"""

from ._anthropic import (
    anthropic_params_to_chat,
    anthropic_response_to_chat,
    anthropic_to_bedrock_params,
    anthropic_to_gemini_params,
    anthropic_to_mistral_params,
)
from ._bedrock import (
    anthropic_response_from_bedrock,
    bedrock_to_anthropic_params,
    bedrock_to_chat_params,
    bedrock_to_gemini_params,
    bedrock_to_mistral_params,
    chat_response_from_bedrock,
)
from ._bedrock_to_others import (
    bedrock_response_to_anthropic,
    bedrock_response_to_chat,
)
from ._gemini import (
    anthropic_response_from_gemini,
    bedrock_response_from_gemini,
    chat_response_from_gemini,
    gemini_to_anthropic_params,
    gemini_to_bedrock_params,
    gemini_to_chat_params,
    gemini_to_mistral_params,
)
from ._gemini_to_others import (
    gemini_response_to_anthropic,
    gemini_response_to_bedrock,
    gemini_response_to_chat,
)
from ._mistral import (
    anthropic_response_from_mistral,
    bedrock_response_from_mistral,
    chat_response_from_mistral,
    gemini_response_from_mistral,
    mistral_to_anthropic_params,
    mistral_to_bedrock_params,
    mistral_to_chat_params,
    mistral_to_gemini_params,
)
from ._mistral_to_others import (
    mistral_response_to_anthropic,
    mistral_response_to_bedrock,
    mistral_response_to_chat,
    mistral_response_to_gemini,
)
from ._openai import (
    chat_response_to_anthropic,
    chat_to_anthropic_params,
    chat_to_bedrock_params,
    chat_to_gemini_params,
    chat_to_mistral_params,
)

__all__ = [
    "anthropic_params_to_chat",
    "anthropic_response_from_bedrock",
    "anthropic_response_from_gemini",
    "anthropic_response_from_mistral",
    "anthropic_response_to_chat",
    "anthropic_to_bedrock_params",
    "anthropic_to_gemini_params",
    "anthropic_to_mistral_params",
    "bedrock_response_from_gemini",
    "bedrock_response_from_mistral",
    "bedrock_response_to_anthropic",
    "bedrock_response_to_chat",
    "bedrock_to_anthropic_params",
    "bedrock_to_chat_params",
    "bedrock_to_gemini_params",
    "bedrock_to_mistral_params",
    "chat_response_from_bedrock",
    "chat_response_from_gemini",
    "chat_response_from_mistral",
    "chat_response_to_anthropic",
    "chat_to_anthropic_params",
    "chat_to_bedrock_params",
    "chat_to_gemini_params",
    "chat_to_mistral_params",
    "gemini_response_from_mistral",
    "gemini_response_to_anthropic",
    "gemini_response_to_bedrock",
    "gemini_response_to_chat",
    "gemini_to_anthropic_params",
    "gemini_to_bedrock_params",
    "gemini_to_chat_params",
    "gemini_to_mistral_params",
    "mistral_response_to_anthropic",
    "mistral_response_to_bedrock",
    "mistral_response_to_chat",
    "mistral_response_to_gemini",
    "mistral_to_anthropic_params",
    "mistral_to_bedrock_params",
    "mistral_to_chat_params",
    "mistral_to_gemini_params",
]
