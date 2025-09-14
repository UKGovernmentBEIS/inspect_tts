import os
from typing import Any, Protocol, runtime_checkable

import httpx
from elevenlabs import AsyncElevenLabs
from inspect_ai._util.httpx import httpx_should_retry, log_httpx_retry_attempt
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from inspect_tts._audio import AudioFormat


@runtime_checkable
class TTSProvider(Protocol):
    """Protocol for a Text-to-Speech provider."""

    async def __call__(
        self, model: str, input: str, voice: str, format: AudioFormat, **kwargs: Any
    ) -> bytes:
        """Call the TTS provider to generate audio from text."""
        ...


def azure_openai_tts_provider() -> TTSProvider:
    """
    Creates a TTSProvider that uses Azure OpenAI for text-to-speech.

    Returns:
        TTSProvider: A callable that performs text-to-speech using Azure OpenAI.
    """
    azure_openai_api_key = os.environ.get("AZURE_OPENAI_API_KEY", None)
    azure_openai_tts_endpoint = os.environ.get("AZURE_OPENAI_TTS_ENDPOINT", None)
    if not azure_openai_api_key:
        raise ValueError(
            "Azure OpenAI API key is not set. "
            "Please set the AZURE_OPENAI_API_KEY environment variable."
        )
    if not azure_openai_tts_endpoint:
        raise ValueError(
            "Azure OpenAI TTS endpoint is not set. "
            "Please set the AZURE_OPENAI_TTS_ENDPOINT environment variable."
        )

    # Use an HTTP client instead of the openai Python package to support models hosted
    # on Azure with older API versions such as 2024-05-01-preview. See
    # https://learn.microsoft.com/en-us/azure/ai-services/openai/text-to-speech-quickstart
    client = httpx.AsyncClient()

    async def tts(
        model: str, input: str, voice: str, format: str, **kwargs: Any
    ) -> bytes:
        url = azure_openai_tts_endpoint
        headers = {"api-key": azure_openai_api_key, "Content-Type": "application/json"}
        payload = {
            "model": model,
            "input": input,
            "voice": voice,
            "response_format": format,
        }
        for key, value in kwargs.items():
            payload[key] = value

        @retry(
            wait=wait_exponential_jitter(initial=3, exp_base=2, max=60),
            stop=stop_after_attempt(500),  # 500 attempts = ~8.5 hours
            retry=retry_if_exception(httpx_should_retry),
            before_sleep=log_httpx_retry_attempt("Azure OpenAI TTS"),
        )
        async def call_api() -> bytes:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.content

        return await call_api()

    return tts


def elevenlabs_tts_provider() -> TTSProvider:
    """
    Creates a TTSProvider that uses ElevenLabs for text-to-speech.

    Returns:
        TTSProvider: A callable that performs text-to-speech using ElevenLabs.
    """
    client = AsyncElevenLabs()

    def to_specific_audio_format(format: AudioFormat) -> str:
        if format == "mp3":
            return "mp3_44100_96"
        elif format == "wav":
            return "pcm_44100"
        else:
            raise ValueError(f"Unsupported audio format for elevenlabs: {format}")

    async def tts(
        model: str, input: str, voice: str, format: AudioFormat, **kwargs: Any
    ) -> bytes:
        result = b""
        stream = client.text_to_speech.convert(
            text=input,
            model_id=model,
            voice_id=voice,
            output_format=to_specific_audio_format(format),
            # TODO: Voice settings etc. from kwargs
        )
        async for value in stream:
            result += value
        return result

    return tts
