# ASR Evaluation Framework

This repository evaluates whether TTS-generated speech matches its reference text.

```text
TTS generated wav + reference text
  -> ASR transcription
  -> text normalization
  -> CER / WER / pinyin error rates
  -> predictions + metrics + errors
```

The default ASR backend is `qwen-asr-1.7b` via the official `qwen-asr` package.

## What Is Included

```text
asr_eval.py              # Run ASR and compute item-level metrics
build_tts_manifest.py    # Build ASR manifest from benchmark jsonl + TTS output directory
export_eval_outputs.py   # Export predictions.jsonl / metrics.json / errors.csv / summary.csv
run_eval.sh              # Main entry; edit a few variables and run sh run_eval.sh
config.example.yaml      # ASR backend and normalization config
requirements.txt         # Python dependencies
data/benchmarks/         # cv3 and seed-tts-eval benchmark jsonl files
test_dummy/              # Tiny no-model smoke test
```

Large files are intentionally not included:

- Qwen-ASR model weights / checkpoints
- TTS generated audio
- ASR output results
- optional speaker-similarity checkpoints

For GitHub, do not commit model checkpoints. Qwen-ASR can load from the HuggingFace repo id `Qwen/Qwen3-ASR-1.7B`, or from a local checkpoint path that each user prepares locally.

## Installation

```bash
git clone <your-repo-url>
cd asr_eval_framework

conda create -n asr-eval python=3.10 -y
conda activate asr-eval

pip install -r requirements.txt
```

Qwen-ASR requires PyTorch. Install the PyTorch build that matches your CUDA/driver from the official PyTorch instructions:

```bash
# Example only. Pick the correct command for your machine from pytorch.org.
pip install torch torchvision torchaudio
```

Then install Qwen-ASR if it was not installed by `requirements.txt` in your environment:

```bash
pip install -U qwen-asr
```

## Quick Smoke Test

The default `run_eval.sh` uses `RUN_MODE="dummy"`, which does not load any ASR model. It only checks that the pipeline can read a manifest and write outputs.

```bash
sh run_eval.sh
```

Expected outputs are written under `test_dummy/`.

## Real Evaluation

For a new TTS system, edit only the **User settings** block at the top of `run_eval.sh`.

```sh
# Options: dummy, eval, batch, custom
RUN_MODE="eval"

# Options: cv3, seed-tts-eval
BENCHMARK="cv3"

# Any name you choose for this TTS system.
TTS_SYSTEM="my-tts-system"

# Options: qwen-asr-1.7b
ASR_BACKEND="qwen-asr-1.7b"

# Root directory of generated TTS wavs for this benchmark.
TTS_ROOT="/path/to/my-tts-system/cv3"

# Set to 1 the first time, so the script builds data/manifests/<tts_system>/<benchmark>.asr.jsonl.
BUILD_MANIFEST=1
```

Then run:

```bash
sh run_eval.sh
```

After the manifest is generated, you can set `BUILD_MANIFEST=0` for repeat runs.

## TTS Output Directory Layout

`build_tts_manifest.py` expects generated audio in this layout:

```text
<tts_root>/<task>/<utt_id>/pred.wav
```

Examples:

```text
/path/to/my-tts-system/cv3/cv3-zeroshot-en/zero_shot_uttid_1/pred.wav
/path/to/my-tts-system/seed-tts/seed-tts-zeroshot-en/zero_shot_uttid_000001/pred.wav
```

The benchmark jsonl files in `data/benchmarks/` provide `id`, `task`, and reference `text`. The script combines those fields with `TTS_ROOT` to locate each `pred.wav`.

`BENCHMARK` uses the evaluation-set name:

- `cv3`
- `seed-tts-eval`

`TTS_ROOT` uses your actual directory name. It can be named `cv3`, `seed-tts`, or anything else, as long as the internal layout matches `<task>/<utt_id>/pred.wav`.

## Manual Manifest Generation

`run_eval.sh` can build manifests automatically. If you prefer to build one manually:

```bash
python build_tts_manifest.py \
  --input-jsonl data/benchmarks/cv3-eval.jsonl \
  --tts-root /path/to/my-tts-system/cv3 \
  --benchmark cv3 \
  --tts-system my-tts-system \
  --out data/manifests/my-tts-system/cv3.asr.jsonl
```

For `seed-tts-eval`:

```bash
python build_tts_manifest.py \
  --input-jsonl data/benchmarks/seed-tts-eval.jsonl \
  --tts-root /path/to/my-tts-system/seed-tts \
  --benchmark seed-tts-eval \
  --tts-system my-tts-system \
  --out data/manifests/my-tts-system/seed-tts-eval.asr.jsonl
```

The builder skips tasks whose names contain `hard`, and keeps only Chinese/English eval tasks.

## Config

Edit `config.example.yaml` if needed.

```yaml
asr_backends:
  - name: qwen-asr-1.7b
    provider: qwen-local
    model_path: Qwen/Qwen3-ASR-1.7B
    device: cuda:0
    dtype: bfloat16
    language: null
    max_inference_batch_size: 1
    max_new_tokens: 256
```

`model_path` can be either:

- HuggingFace repo id: `Qwen/Qwen3-ASR-1.7B`
- local checkpoint directory: `/path/to/local/Qwen3-ASR-1.7B`

If you use a local checkpoint, keep it outside this git repository or under an ignored directory such as `checkpoints/`.

## Output

Outputs are written to:

```text
outputs/asr_eval/<benchmark>/<tts_system>/<asr_backend>/
```

Each output directory contains:

```text
results.jsonl       # Full per-item results, useful for debugging
predictions.jsonl   # Compact per-item predictions
summary.csv         # Aggregated metrics by asr_backend + benchmark + tts_system + lang
metrics.json        # JSON version of summary.csv
errors.csv          # Failed samples; header only if no failures
```

Recommended headline metrics:

- English: `avg_wer`
- Chinese: `avg_cer`, `avg_pinyin_cer`, `avg_pinyin_notone_cer`

Chinese `avg_wer` is not recommended as the main metric because Chinese text has no natural whitespace tokenization.

## Batch Mode

`batch` mode runs multiple rows in `BATCH_TARGETS` sequentially, not in parallel.

```sh
RUN_MODE="batch"

BATCH_TARGETS='''
my-tts-system cv3
my-tts-system seed-tts-eval
'''
```

For parallel execution, launch multiple terminal sessions or jobs manually, and write each run to a separate output directory.

## Custom Mode

Use `custom` when you already have a manifest/config/output directory and do not want to follow the default path convention.

```sh
RUN_MODE="custom"
CUSTOM_MANIFEST="/path/to/manifest.jsonl"
CUSTOM_CONFIG="config.example.yaml"
CUSTOM_OUT_DIR="/path/to/output_dir"
```

## Checkpoint Policy

Qwen-ASR checkpoint files should not be committed to GitHub:

- They are large.
- They may have model-license or redistribution constraints.
- GitHub repositories are easier to clone when weights are downloaded separately.

Keep only `config.example.yaml` and documentation in the repo. Users can either use `model_path: Qwen/Qwen3-ASR-1.7B` to download via HuggingFace at runtime, or set `model_path` to their own local checkpoint directory.
