# Qwen-ASR Backend Notes

This project uses the official `qwen-asr` package through `provider: qwen-local`.

## Install

```bash
pip install -r requirements.txt
pip install -U qwen-asr
```

Qwen-ASR also requires PyTorch. Install a PyTorch build that matches your CUDA/driver from the official PyTorch instructions.

## Model Path

In `config.example.yaml`:

```yaml
asr_backends:
  - name: qwen-asr-1.7b
    provider: qwen-local
    model_path: Qwen/Qwen3-ASR-1.7B
    device: cuda:0
    dtype: bfloat16
    language: null
```

`model_path` can be:

- `Qwen/Qwen3-ASR-1.7B`, letting HuggingFace/qwen-asr download or resolve the model.
- `/path/to/local/Qwen3-ASR-1.7B`, if you have downloaded the checkpoint yourself.

Do not commit checkpoints into this repository. Keep them outside the repo or under an ignored directory such as `checkpoints/`.

## Manifest Input

Each JSONL row passed to `asr_eval.py` should look like:

```json
{"id": "sample_001", "audio_path": "/path/to/pred.wav", "reference": "目标文本", "lang": "zh", "benchmark": "cv3", "tts_system": "my-tts-system", "meta": {}}
```

Most users should not write this manifest manually. Instead, put generated TTS files under:

```text
<tts_root>/<task>/<utt_id>/pred.wav
```

Then set `TTS_ROOT` and `BUILD_MANIFEST=1` in `run_eval.sh`.

## Run

```bash
sh run_eval.sh
```

Outputs are written to:

```text
outputs/asr_eval/<benchmark>/<tts_system>/<asr_backend>/
```

## Metrics

Use these headline metrics:

- English: `avg_wer`
- Chinese: `avg_cer`, `avg_pinyin_cer`, `avg_pinyin_notone_cer`

`avg_wer` is not recommended as the main Chinese metric because Chinese text has no natural whitespace tokenization.
