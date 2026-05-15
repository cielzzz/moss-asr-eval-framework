# ASR Evaluation Framework / ASR 评测框架

这个项目用于评估 TTS 生成音频和目标文本的一致性。It evaluates whether TTS-generated speech matches its reference text.

```text
TTS generated wav + reference text
  -> ASR transcription
  -> text normalization
  -> CER / WER / pinyin error rates
  -> predictions + metrics + errors
```

默认 ASR backend 是 `qwen-asr-1.7b`，通过官方 `qwen-asr` package 调用。The default ASR backend is `qwen-asr-1.7b` via the official `qwen-asr` package.

## 目录结构 / Repository Layout

```text
asr_eval.py              # 主评测脚本：跑 ASR + 计算逐条指标 / main evaluation script
build_tts_manifest.py    # 从 benchmark jsonl + TTS 音频目录生成 ASR manifest
export_eval_outputs.py   # 从 results.jsonl 导出 predictions/metrics/errors/summary
run_eval.sh              # 主入口：修改顶部少数参数后直接 sh run_eval.sh
config.example.yaml      # ASR backend 和 normalization 配置示例
requirements.txt         # Python dependencies
data/benchmarks/         # cv3 和 seed-tts-eval benchmark jsonl
test_dummy/              # 不加载模型的小型 smoke test
```

不包含大文件 / Large files are intentionally not included:

- Qwen-ASR model weights / checkpoints
- TTS generated audio / TTS 推理音频
- ASR output results / 评测输出
- optional speaker-similarity checkpoints / 可选说话人相似度权重

Qwen-ASR 权重不建议放进 GitHub。用户可以使用 `model_path: Qwen/Qwen3-ASR-1.7B` 让环境自行下载，也可以把 `model_path` 改成本地 checkpoint 路径。

## 安装 / Installation

```bash
git clone https://github.com/cielzzz/moss-asr-eval-framework.git
cd moss-asr-eval-framework

conda create -n asr-eval python=3.10 -y
conda activate asr-eval

pip install -r requirements.txt
```

Qwen-ASR 需要 PyTorch。请根据你的 CUDA/driver 从 PyTorch 官网选择合适安装命令。Qwen-ASR requires PyTorch; install the build matching your CUDA/driver.

```bash
# Example only. Pick the correct command for your machine from pytorch.org.
pip install torch torchvision torchaudio
```

如果 `requirements.txt` 安装后环境里仍没有 `qwen-asr`，再执行：

```bash
pip install -U qwen-asr
```

## 快速自测 / Quick Smoke Test

默认 `run_eval.sh` 使用：

```sh
RUN_MODE="dummy"
```

`dummy` 不加载真实 ASR 模型，只读取测试样例同名 `.txt`，用于确认 pipeline 能运行。The dummy mode does not load an ASR model; it only verifies the pipeline.

```bash
sh run_eval.sh
```

输出会写到 `test_dummy/`。

## 真实评测 / Real Evaluation

对于一个新的 TTS 系统，通常只需要修改 `run_eval.sh` 顶部的 **User settings** 区域。For a new TTS system, edit only the **User settings** block at the top of `run_eval.sh`.

```sh
# Options: dummy, eval, batch, custom
RUN_MODE="eval"

# Options: cv3, seed-tts-eval
BENCHMARK="cv3"

# 自定义 TTS 系统名称，会用于 manifest 和输出路径。
# Any name you choose for this TTS system.
TTS_SYSTEM="my-tts-system"

# Options: qwen-asr-1.7b
ASR_BACKEND="qwen-asr-1.7b"

# 当前 benchmark 的 TTS 推理输出根目录。
# Root directory of generated TTS wavs for this benchmark.
TTS_ROOT="/path/to/my-tts-system/cv3"

# 新 TTS 系统第一次跑时设为 1，用来生成 data/manifests/<tts_system>/<benchmark>.asr.jsonl。
# Set to 1 the first time so the script builds the manifest before ASR evaluation.
BUILD_MANIFEST=1
```

然后运行 / Then run:

```bash
sh run_eval.sh
```

manifest 生成后，重复跑同一组可以改回：

```sh
BUILD_MANIFEST=0
```

## TTS 输出目录格式 / TTS Output Layout

`build_tts_manifest.py` 期望 TTS 音频目录满足：

```text
<tts_root>/<task>/<utt_id>/pred.wav
```

示例 / Examples:

```text
/path/to/my-tts-system/cv3/cv3-zeroshot-en/zero_shot_uttid_1/pred.wav
/path/to/my-tts-system/seed-tts/seed-tts-zeroshot-en/zero_shot_uttid_000001/pred.wav
```

`data/benchmarks/` 里的 jsonl 提供 `id`、`task` 和目标文本 `text`。脚本会用这些字段和 `TTS_ROOT` 拼出每条样本的 `pred.wav` 路径。

`BENCHMARK` 是评测集名称 / evaluation-set name:

- `cv3`
- `seed-tts-eval`

`TTS_ROOT` 是你的真实推理目录，可以叫 `cv3`、`seed-tts` 或其它名字，只要内部满足 `<task>/<utt_id>/pred.wav` 即可。

## 手动生成 Manifest / Manual Manifest Generation

`run_eval.sh` 可以自动生成 manifest。如果想手动生成，可以运行：

```bash
python build_tts_manifest.py \
  --input-jsonl data/benchmarks/cv3-eval.jsonl \
  --tts-root /path/to/my-tts-system/cv3 \
  --benchmark cv3 \
  --tts-system my-tts-system \
  --out data/manifests/my-tts-system/cv3.asr.jsonl
```

`seed-tts-eval` 示例：

```bash
python build_tts_manifest.py \
  --input-jsonl data/benchmarks/seed-tts-eval.jsonl \
  --tts-root /path/to/my-tts-system/seed-tts \
  --benchmark seed-tts-eval \
  --tts-system my-tts-system \
  --out data/manifests/my-tts-system/seed-tts-eval.asr.jsonl
```

该脚本会跳过 task 名称中包含 `hard` 的样本，只保留中文和英文评测样本。The builder skips tasks whose names contain `hard`, and keeps only Chinese/English eval tasks.

## 配置 / Config

主要配置在 `config.example.yaml`。The main config is `config.example.yaml`.

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

`model_path` 可以是：

- HuggingFace repo id: `Qwen/Qwen3-ASR-1.7B`
- 本地 checkpoint 目录: `/path/to/local/Qwen3-ASR-1.7B`

如果使用本地 checkpoint，请放在仓库外，或者放在被 `.gitignore` 忽略的 `checkpoints/` 目录下。

## 输出 / Output

输出路径 / Output path:

```text
outputs/asr_eval/<benchmark>/<tts_system>/<asr_backend>/
```

每个输出目录包含 / Each output directory contains:

```text
results.jsonl       # 完整逐条结果，字段最多，适合 debug / full per-item results
predictions.jsonl   # 简化逐条预测 / compact predictions
summary.csv         # 按 asr_backend + benchmark + tts_system + lang 聚合
metrics.json        # summary.csv 的 JSON 版本
errors.csv          # 失败样本；无失败时只有表头
```

推荐主指标 / Recommended headline metrics:

- English: `avg_wer`
- Chinese: `avg_cer`, `avg_pinyin_cer`, `avg_pinyin_notone_cer`

中文 `avg_wer` 不建议作为主指标，因为中文文本没有天然空格分词。Chinese `avg_wer` is not recommended as the main metric because Chinese has no natural whitespace tokenization.

错误率公式 / Error-rate formula:

```text
error_rate = (substitution + deletion + insertion) / reference_length
```

因此 CER/WER/PER 没有严格上限；插入很多内容时可能大于 1。

## Batch Mode

`batch` 模式会按 `BATCH_TARGETS` 顺序跑多组，不并行。Batch mode is sequential, not parallel.

```sh
RUN_MODE="batch"

BATCH_TARGETS='
my-tts-system cv3
my-tts-system seed-tts-eval
'
```

如需并行，请启动多个终端或任务，并保证不同进程写到不同输出目录。

## Custom Mode

当你已经有自己的 manifest/config/output_dir，不想走默认目录规则时使用 `custom`。

```sh
RUN_MODE="custom"
CUSTOM_MANIFEST="/path/to/manifest.jsonl"
CUSTOM_CONFIG="config.example.yaml"
CUSTOM_OUT_DIR="/path/to/output_dir"
```

## Checkpoint Policy / 权重文件说明

不要把 Qwen-ASR checkpoint 提交到 GitHub：

- 文件体积很大。
- 可能涉及模型 license 或再分发限制。
- 仓库保持轻量，其他用户 clone 更方便。

仓库只保留 `config.example.yaml` 和文档。用户可以用 `model_path: Qwen/Qwen3-ASR-1.7B` 让环境下载，也可以改成本地 checkpoint 路径。
