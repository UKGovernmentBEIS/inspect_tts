# Inspect TTS

An Inspect extension providing a text-to-speech solver for audio-based evaluations.

## Getting Started

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. To install
the dependencies, run:

```sh
uv sync
```

## Examples

This repo has some built-in examples taken from
[inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals) which is a dev
dependency of this project.

To run one of the example tasks, set `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_BASE_URL` and
`AZURE_OPENAI_TTS_ENDPOINT` in your environment, and run the following command:

```sh
uv run inspect eval examples/xstest.py --model openai/azure/gpt-4o-audio-preview
```
OR

```sh
uv run inspect eval examples/disclosure_voice_eval.py --model openai/azure/gpt-4o-audio-preview
```
