# TTS ASR 评测指标计算说明

## 1. 背景

之前对 TTS 模型做客观评测时，ASR 识别器按语言区分：

| 语言 | 旧 ASR 方案 | 用途 |
| --- | --- | --- |
| 中文 | Paraformer-zh | 将中文生成音频转写为文本，再计算错误率 |
| 英文 | Whisper-large-v3 | 将英文生成音频转写为文本，再计算错误率 |

现在统一改为：

| 语言 | 新 ASR 方案 | 配置 |
| --- | --- | --- |
| 中文 | Qwen-ASR-1.7B | `qwen-asr-1.7b` |
| 英文 | Qwen-ASR-1.7B | `qwen-asr-1.7b` |

因此，当前所有 TTS 模型、所有 benchmark 的 ASR 评测结果都应放在：

```text
outputs/asr_eval/<benchmark>/<tts_system>/qwen-asr-1.7b/
```

每个结果目录包含：

```text
results.jsonl        # 每条样本的完整评测结果
predictions.jsonl    # 便于人工检查的预测结果
summary.csv          # 按 benchmark / tts_system / lang 聚合后的平均指标
metrics.json         # summary 的 json 版本
errors.csv           # 推理或评测失败样本
```

## 2. 当前评测流程

整体流程如下：

```text
TTS 生成音频 + reference 文本
  -> Qwen-ASR-1.7B 转写生成 hyp 文本
  -> 文本归一化
  -> 计算 CER / WER / pinyin error rate
  -> 输出逐条结果和聚合 summary
```

输入 manifest 每行是一条样本：

```json
{
  "id": "sample_001",
  "audio_path": "/path/to/pred.wav",
  "reference": "目标文本",
  "lang": "zh",
  "benchmark": "seed-tts-eval",
  "tts_system": "moss-tts-local-v1-tokenscontrol0",
  "meta": {}
}
```

当前 ASR backend 配置在 `config.example.yaml`：

```yaml
asr_backends:
  - name: qwen-asr-1.7b
    provider: qwen-local
    model_path: Qwen/Qwen3-ASR-1.7B
    device: cuda:0
    dtype: bfloat16
    language: null
```

`language: null` 表示让 Qwen-ASR 自动识别语言。

## 3. 文本归一化逻辑

计算指标前，会先对 reference 和 ASR hypothesis 做归一化。

当前归一化配置：

```yaml
normalization:
  lowercase: true
  remove_punctuation: true
  remove_spaces_for_cer: true
  chinese_to_pinyin_style: "TONE3"
```

具体规则：

| 步骤 | 说明 | 示例 |
| --- | --- | --- |
| 去首尾空白 | `strip()` | `" hello "` -> `"hello"` |
| 全角空格转半角 | `\u3000` -> `" "` | `"你　好"` -> `"你 好"` |
| 连续空白合并 | 多个空格变成一个空格 | `"a   b"` -> `"a b"` |
| 小写化 | 英文统一小写 | `"Hello"` -> `"hello"` |
| 去标点 | 删除中英文标点 | `"你好！"` -> `"你好"` |
| CER 去空格 | 仅 CER 计算时去掉空格 | `"hello world"` -> `"helloworld"` |

需要注意：CER 和 WER 使用不同的归一化文本。

| 指标 | 是否去空格 | 原因 |
| --- | --- | --- |
| CER | 去空格 | 按字符比较，空格通常不是中文语义内容 |
| WER | 保留空格 | 英文需要依赖空格做 word token 切分 |

## 4. CER 计算逻辑

CER 是 Character Error Rate，字符错误率。

公式：

```text
CER = (S + D + I) / N
```

其中：

| 符号 | 含义 |
| --- | --- |
| S | substitution，替换错误数 |
| D | deletion，删除错误数 |
| I | insertion，插入错误数 |
| N | reference 字符数 |

当前实现逻辑：

1. 对 reference 和 hypothesis 分别做 CER 归一化。
2. 把字符串拆成单个字符。
3. 用空格把字符重新拼成 token 序列。
4. 调用 `jiwer.wer()` 复用编辑距离逻辑计算字符级错误率。

等价示例：

```text
ref = "今天天气真好"
hyp = "今天天气很好"

ref chars = "今 天 天 气 真 好"
hyp chars = "今 天 天 气 很 好"

S = 1  # 真 -> 很
D = 0
I = 0
N = 6

CER = 1 / 6 = 0.1667
```

空 reference 的特殊情况：

| reference | hypothesis | CER |
| --- | --- | --- |
| 空 | 空 | 0.0 |
| 空 | 非空 | 1.0 |

## 5. WER 计算逻辑

WER 是 Word Error Rate，词错误率。

公式：

```text
WER = (S + D + I) / N
```

区别在于这里的 `N` 是 reference 的 word 数，而不是字符数。

当前实现逻辑：

1. 对 reference 和 hypothesis 分别做 WER 归一化。
2. 保留空格。
3. 调用 `jiwer.wer(ref, hyp)`。

英文示例：

```text
ref = "the weather is good"
hyp = "the weather good"

ref words = ["the", "weather", "is", "good"]
hyp words = ["the", "weather", "good"]

D = 1  # 删除 is
N = 4

WER = 1 / 4 = 0.25
```

中文需要谨慎解读 WER，因为中文句子通常没有自然空格。例如：

```text
ref = "今天天气真好"
hyp = "今天天气很好"
```

如果没有分词，整句话可能被当成一个 token，因此中文主要看：

```text
avg_cer
avg_pinyin_cer
avg_pinyin_notone_cer
```

## 6. PER / pinyin error rate 计算逻辑

这里的 PER 指当前代码里的拼音错误率，也就是对中文样本额外计算的 pinyin error rate。

当前输出字段有两套：

| 字段 | 是否带声调 | 说明 |
| --- | --- | --- |
| `pinyin_cer` | 带声调 | 使用 `TONE3`，如 `zhong1 guo2` |
| `pinyin_wer` | 带声调 | 以拼音音节为 token 计算 |
| `pinyin_notone_cer` | 不带声调 | 使用 `NORMAL`，如 `zhong guo` |
| `pinyin_notone_wer` | 不带声调 | 以无声调拼音音节为 token 计算 |

中文样本才会计算这些字段；英文样本这些字段为 `nan` 或 `None`。

### 6.1 带声调 pinyin CER / WER

先把中文转成带数字声调的拼音：

```text
中国 -> zhong1 guo2
重庆 -> chong2 qing4
银行 -> yin2 hang2
```

然后分别计算：

```text
pinyin_wer = 对拼音 token 序列计算 WER
pinyin_cer = 拼音字符串拼接后按字符计算 CER
```

示例：

```text
ref = "重庆"
hyp = "重情"

ref_pinyin = "chong2 qing4"
hyp_pinyin = "zhong4 qing2"
```

如果从拼音 token 看：

```text
ref tokens = ["chong2", "qing4"]
hyp tokens = ["zhong4", "qing2"]

S = 2
N = 2
pinyin_wer = 1.0
```

如果从拼音字符看，会按 `"chong2qing4"` 和 `"zhong4qing2"` 做字符级编辑距离，因此得到 `pinyin_cer`。

### 6.2 不带声调 pinyin CER / WER

无声调版本会忽略声调差异：

```text
ref = "买"
hyp = "卖"

ref_pinyin = "mai3"
hyp_pinyin = "mai4"

pinyin_cer 会计入声调错误。
```

但无声调时：

```text
ref_pinyin_notone = "mai"
hyp_pinyin_notone = "mai"

pinyin_notone_cer = 0.0
pinyin_notone_wer = 0.0
```

所以：

| 指标 | 能否反映声调错误 |
| --- | --- |
| `pinyin_cer` / `pinyin_wer` | 能 |
| `pinyin_notone_cer` / `pinyin_notone_wer` | 不能 |

## 7. 为什么 pinyin CER 可能比 CER 低

中文 CER 是按汉字比较，只要字不同就算错。

但 pinyin CER 是按读音比较，如果两个字读音相同或非常接近，拼音层面的错误会更小。

例子 1：同音字

```text
ref = "今天"
hyp = "金天"
```

汉字层面：

```text
今 == 金 ? 不同
天 == 天 ? 相同

CER = 1 / 2 = 0.5
```

拼音层面：

```text
今天 -> jin1 tian1
金天 -> jin1 tian1

pinyin_cer = 0.0
```

例子 2：只错字形，不错读音

```text
ref = "语音合成"
hyp = "语音和成"
```

汉字层面：

```text
合 -> 和，字不同
CER > 0
```

拼音层面：

```text
合 -> he2
和 -> he2

拼音相同，所以 pinyin 错误更小。
```

例子 3：声调错但无声调 PER 不错

```text
ref = "买东西"
hyp = "卖东西"
```

带声调：

```text
买 -> mai3
卖 -> mai4

pinyin_cer / pinyin_wer 会计入错误。
```

不带声调：

```text
mai == mai

pinyin_notone_cer / pinyin_notone_wer 不计入这个错误。
```

因此，通常可能出现：

```text
avg_pinyin_notone_cer <= avg_pinyin_cer <= avg_cer
```

但这不是数学上必然成立。因为拼音长度、分词、非汉字过滤、多音字转换等因素都可能影响结果。

## 8. 与旧 seed-tts-eval 逻辑的关系

seed-tts-eval 原始说明里，WER 使用：

| 语言 | 原始 ASR |
| --- | --- |
| 英文 | Whisper-large-v3 |
| 中文 | Paraformer-zh |

现在改为统一使用 Qwen-ASR-1.7B 后，变化点是：

| 项目 | 旧逻辑 | 当前逻辑 |
| --- | --- | --- |
| 英文 ASR | Whisper-large-v3 | Qwen-ASR-1.7B |
| 中文 ASR | Paraformer-zh | Qwen-ASR-1.7B |
| ASR 输出 | 不同语言来自不同模型 | 中英文都来自同一个模型 |
| 指标计算 | ASR 输出后再算 WER 等指标 | ASR 输出后统一算 CER/WER/pinyin error rate |
| 可比性 | 与 seed-tts-eval 原始论文配置一致 | 更适合同一批 TTS 系统在同一 ASR 下横向比较 |

需要注意：更换 ASR 模型后，绝对数值不能直接和旧 Paraformer / Whisper 结果严格对齐比较，因为 ASR 识别误差分布已经变化。

更合理的比较方式是：

```text
在同一个 ASR backend 下，比较不同 TTS 系统或不同参数配置之间的相对差异。
```

## 9. summary.csv 字段说明

`summary.csv` 按以下 key 分组：

```text
asr_backend + benchmark + tts_system + lang
```

字段说明：

| 字段 | 含义 |
| --- | --- |
| `asr_backend` | ASR 模型，例如 `qwen-asr-1.7b` |
| `benchmark` | 测试集，例如 `seed-tts-eval` 或 `cv3` |
| `tts_system` | TTS 系统或参数配置名称 |
| `lang` | `zh` 或 `en` |
| `n` | 当前分组样本数 |
| `avg_cer` | 当前分组所有样本 CER 的算术平均 |
| `avg_wer` | 当前分组所有样本 WER 的算术平均 |
| `avg_pinyin_cer` | 中文样本带声调 pinyin CER 平均值 |
| `avg_pinyin_wer` | 中文样本带声调 pinyin WER 平均值 |
| `avg_pinyin_notone_cer` | 中文样本无声调 pinyin CER 平均值 |
| `avg_pinyin_notone_wer` | 中文样本无声调 pinyin WER 平均值 |
| `avg_sim` | speaker similarity 平均值；当前默认未开启 |

聚合方式是直接平均逐条样本指标：

```text
avg_metric = sum(metric_i) / count(metric_i)
```

如果某个字段在该语言下不存在，例如英文没有 pinyin 指标，则 summary 中显示为 `nan`。

## 10. 当前推荐解读方式

中文建议优先看：

```text
avg_cer
avg_pinyin_cer
avg_pinyin_notone_cer
```

英文建议优先看：

```text
avg_wer
avg_cer
```

跨 TTS 系统比较时，建议固定：

```text
同一个 benchmark
同一个 ASR backend
同一个 lang
```

例如比较 MOSS-TTS-Local-v1 四个参数配置时，应分别比较：

```text
seed-tts-eval / zh / qwen-asr-1.7b
seed-tts-eval / en / qwen-asr-1.7b
cv3 / zh / qwen-asr-1.7b
cv3 / en / qwen-asr-1.7b
```

不要把不同 ASR backend 的绝对值直接混在一起排序。
