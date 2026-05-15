# ASR 评测框架

这个项目用于评估 TTS 生成音频和目标文本的一致性。

```text
TTS generated wav + reference text
  -> ASR transcription
  -> text normalization
  -> CER / WER / pinyin error rates
  -> predictions + metrics + errors
```

默认 ASR backend 是 `qwen-asr-1.7b`，通过官方 `qwen-asr` package 调用。

## 目录结构

```text
asr_eval.py              # 主评测脚本：跑 ASR + 计算逐条指标
build_tts_manifest.py    # 从 benchmark jsonl + TTS 音频目录生成 ASR manifest
export_eval_outputs.py   # 从 results.jsonl 导出 predictions/metrics/errors/summary
run_eval.sh              # 主入口：修改顶部少数参数后直接 sh run_eval.sh
config.example.yaml      # ASR backend 和 normalization 配置示例
requirements.txt         # Python 依赖
data/benchmarks/         # cv3 和 seed-tts-eval benchmark jsonl
test_dummy/              # 不加载模型的小型 smoke test
```


Qwen-ASR 权重：用户可以使用 `model_path: Qwen/Qwen3-ASR-1.7B` 让环境自行下载，也可以把 `model_path` 改成本地 checkpoint 路径。

## 安装

最小安装流程：

```bash
git clone https://github.com/cielzzz/moss-asr-eval-framework.git
cd moss-asr-eval-framework

conda create -n asr-eval python=3.10 -y
conda activate asr-eval
```

Qwen-ASR 需要 PyTorch。请根据你的 CUDA/driver 从 PyTorch 官网选择合适安装命令；如果你已经有可用的 PyTorch 环境，可以跳过这一步。

```bash
# 这里只是示例，请根据你的机器从 pytorch.org 选择正确命令。
pip install torch torchvision torchaudio
```

然后安装本项目依赖：

```bash
pip install -r requirements.txt
```

如果 `requirements.txt` 安装后环境里仍没有 `qwen-asr`，再执行：

```bash
pip install -U qwen-asr
```

## 快速自测

默认 `run_eval.sh` 使用：

```sh
RUN_MODE="dummy"
```

`dummy` 不加载真实 ASR 模型，只读取测试样例同名 `.txt`，用于确认 pipeline 能运行。

```bash
sh run_eval.sh
```

输出会写到 `test_dummy/`。

## 真实评测

对于一个新的 TTS 系统，通常只需要修改 `run_eval.sh` 顶部的 **User settings** 区域。

```sh
# 可选项: dummy, eval, batch, custom
RUN_MODE="eval"

# 可选项: cv3, seed-tts-eval
BENCHMARK="cv3"

# 自定义 TTS 系统名称，会用于 manifest 和输出路径。
TTS_SYSTEM="my-tts-system"

# 可选项: qwen-asr-1.7b
ASR_BACKEND="qwen-asr-1.7b"

# 当前 benchmark 的 TTS 推理输出根目录。
TTS_ROOT="/path/to/my-tts-system/cv3"

# 新 TTS 系统第一次跑时设为 1，用来生成 data/manifests/<tts_system>/<benchmark>.asr.jsonl。
BUILD_MANIFEST=1
```

然后运行:

```bash
sh run_eval.sh
```

manifest 生成后，重复跑同一组可以改回：

```sh
BUILD_MANIFEST=0
```

## Qwen-TTS 示例

假设你已经 `git clone` 了本仓库，并且拿到了 qwen-tts 的推理输出：

```text
your folder/Qwen-TTS
```

其中包含两个 benchmark 的输出目录：

```text
your folder/Qwen-TTS/cv3
your folder/Qwen-TTS/seed-tts
```

先安装依赖：

```bash
cd moss-asr-eval-framework
conda activate asr-eval
pip install -r requirements.txt
```

### 运行 cv3

打开 `run_eval.sh`，只改顶部这些参数：

```sh
RUN_MODE="eval"
BENCHMARK="cv3"
TTS_SYSTEM="qwen-tts"
ASR_BACKEND="qwen-asr-1.7b"
TTS_ROOT="your folder/Qwen-TTS/cv3"
BUILD_MANIFEST=1
ACTIVATE_ENV=0
```

然后运行：

```bash
sh run_eval.sh
```

输出会写到：

```text
outputs/asr_eval/cv3/qwen-tts/qwen-asr-1.7b/
```

### 运行 seed-tts-eval

同样打开 `run_eval.sh`，把 benchmark 和 TTS_ROOT 改成 seed-tts：

```sh
RUN_MODE="eval"
BENCHMARK="seed-tts-eval"
TTS_SYSTEM="qwen-tts"
ASR_BACKEND="qwen-asr-1.7b"
TTS_ROOT="your folder/Qwen-TTS/seed-tts"
BUILD_MANIFEST=1
ACTIVATE_ENV=0
```

然后运行：

```bash
sh run_eval.sh
```

输出会写到：

```text
outputs/asr_eval/seed-tts-eval/qwen-tts/qwen-asr-1.7b/
```

第一次跑新 TTS 系统时使用 `BUILD_MANIFEST=1`。manifest 生成后，如果只是重跑 ASR，可以改成：

```sh
BUILD_MANIFEST=0
```

## TTS 输出目录格式

`build_tts_manifest.py` 期望 TTS 音频目录满足：

```text
<tts_root>/<task>/<utt_id>/pred.wav
```

示例：

```text
/path/to/my-tts-system/cv3/cv3-zeroshot-en/zero_shot_uttid_1/pred.wav
/path/to/my-tts-system/seed-tts/seed-tts-zeroshot-en/zero_shot_uttid_000001/pred.wav
```

`data/benchmarks/` 里的 jsonl 提供 `id`、`task` 和目标文本 `text`。脚本会用这些字段和 `TTS_ROOT` 拼出每条样本的 `pred.wav` 路径。

`BENCHMARK` 是评测集名称：

- `cv3`
- `seed-tts-eval`

`TTS_ROOT` 是你的真实推理目录，可以叫 `cv3`、`seed-tts` 或其它名字，只要内部满足 `<task>/<utt_id>/pred.wav` 即可。

## 手动生成 Manifest

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

该脚本会跳过 task 名称中包含 `hard` 的样本，只保留中文和英文评测样本。

## 配置

主要配置在 `config.example.yaml`。

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

## 输出

输出路径：

```text
outputs/asr_eval/<benchmark>/<tts_system>/<asr_backend>/
```

每个输出目录包含：

```text
results.jsonl       # 完整逐条结果，字段最多，适合 debug
predictions.jsonl   # 简化逐条预测
summary.csv         # 按 asr_backend + benchmark + tts_system + lang 聚合
metrics.json        # summary.csv 的 JSON 版本
errors.csv          # 失败样本；无失败时只有表头
```

推荐主指标：

- 英文：`avg_wer`
- 中文：`avg_cer`, `avg_pinyin_cer`, `avg_pinyin_notone_cer`

中文 `avg_wer` 不建议作为主指标，因为中文文本没有天然空格分词。

错误率公式：

```text
error_rate = (substitution + deletion + insertion) / reference_length
```

因此 CER/WER/PER 没有严格上限；插入很多内容时可能大于 1。

## 批量模式

`batch` 模式会按 `BATCH_TARGETS` 顺序跑多组，不并行。

```sh
RUN_MODE="batch"

BATCH_TARGETS='
my-tts-system cv3
my-tts-system seed-tts-eval
'
```

如需并行，请启动多个终端或任务，并保证不同进程写到不同输出目录。

## 自定义模式

当你已经有自己的 manifest/config/output_dir，不想走默认目录规则时使用 `custom`。

```sh
RUN_MODE="custom"
CUSTOM_MANIFEST="/path/to/manifest.jsonl"
CUSTOM_CONFIG="config.example.yaml"
CUSTOM_OUT_DIR="/path/to/output_dir"
```
