"""An Inspect extension for text-to-speech."""

from inspect_tts._audio import AudioFormat
from inspect_tts._text_to_speech import text_to_speech

__all__ = [
    "AudioFormat",
    "text_to_speech",
]
