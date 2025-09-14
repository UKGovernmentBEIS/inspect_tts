from typing import Literal

from inspect_ai import Task, task, task_with
from inspect_ai.model import Model
from inspect_ai.solver import generate, system_message
from inspect_evals.xstest import xstest

from inspect_tts._text_to_speech import text_to_speech


@task
def xstest_tts(
    subset: Literal["safe", "unsafe"] = "safe",
    scorer_model: str | Model | None = "openai/gpt-4o",
    tts_voice: str = "alloy",
    tts_format: Literal["wav", "mp3"] = "wav",
    tts_audio_dir: str | None = None,
    tts_save_dir: str | None = None,
    tts_kwargs: dict[str, str] | None = None,
) -> Task:
    solver = [
        # System message from xstest's default solver.
        system_message("You are a helpful assistant."),
        text_to_speech(
            "azure_openai",
            model="tts-1",
            voice=tts_voice,
            format=tts_format,
            audio_dir=tts_audio_dir,
            save_dir=tts_save_dir,
            **(tts_kwargs or {}),
        ),
        generate(),
    ]
    return task_with(
        xstest(subset=subset, scorer_model=scorer_model),
        solver=solver,
    )
