# Inspect TTS

An Inspect extension providing a text-to-speech (TTS) solver for audio-based evaluations.

## Getting Started

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. To install
the dependencies, run:

```sh
uv sync
```

To use the Azure OpenAI TTS Provider, set `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_BASE_URL` and
`AZURE_OPENAI_TTS_ENDPOINT` in your environment (which can be done by putting them in a `.env` file.)

To use the ElevenLabs TTS Provider, set `ELEVENLABS_API_KEY`.

Then, to make convert your text input into audio before passing it to a model, add the `text_to_speech` solver in your task before calling `generate`. For example, your task might look something like:

```py
from typing import Any, Literal

from inspect_ai import Task, task
from inspect_ai.dataset import csv_dataset
from inspect_ai.scorer import answer
from inspect_ai.solver import generate

from inspect_tts._audio import AudioFormat
from inspect_tts._text_to_speech import text_to_speech


@task
def audio_eval(
    provider: Literal["azure_openai", "elevenlabs"],
    model: str,
    voice: str,
    format: AudioFormat,
    audio_dir: str | None = None,
    save_dir: str | None = None,
    **tts_provider_kwargs: Any,
) -> Task:
    """
    Evaluates AI responses to audio inputs converted from text.

    Args:
        provider (Literal["azure_openai", "elevenlabs"]): The TTS provider.
        model (str): The model from the TTS provider to use for TTS.
        voice (str): The voice rom the TTS provider to use for TTS.
        format (AudioFormat): The format for the generated audio,
            either 'wav' or 'mp3'.
        audio_dir (str | None): Optional directory in which to search for
            existing audio files.
        save_dir (str | None): Optional directory in which to save audio files.
        **tts_provider_kwargs: Additional arguments to pass to the TTS provider.
    """
    return Task(
        dataset = csv_dataset("text_inputs.csv"),
        solver = [
            text_to_speech(
                provider,
                model=model,
                voice=voice,
                format=format,
                audio_dir=audio_dir,
                save_dir=save_dir,
                **tts_provider_kwargs,
            ),
            generate(),
        ],
        scorer=answer(),
    )
```

Assuming this in a file called `task.py`, we could run against it `gpt-4o-audio-preview` using:

```sh
uv run inspect eval task.py --model openai/gpt-4o-audio-preview
```
