"""Package for supporting audio compliance evals."""

from inspect_tts._text_to_speech import text_to_speech
from inspect_tts._tts_provider import TTSProvider, azure_openai_tts_provider

__all__ = [
    "text_to_speech",
    "TTSProvider",
    "azure_openai_tts_provider",
]
