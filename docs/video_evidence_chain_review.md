# 视频证据链阶段评估报告

## 1. 当前结论

当前视频链路已经完成一个小样本真实闭环：

```text
视频文件
  → 多关键帧抽取
  → PaddleOCR 本地 OCR
  → Qwen-VL 真实视觉理解
  → DeepSeek 真实文本分析
  → results / model_calls / batch_report 统一记录
```

但它还不是完整视频理解平台。原因是：真实 ASR 目前只有受保护调用入口，尚未产生经过复核的小样本转写批次；`audio_transcript` 仍然不是已验证的可信音频证据，所以最终文件状态仍应保守看作 `partial_success`。

## 2. 使用的证据批次

| 批次 | 作用 | 结论 |
|---|---|---|
| `output/batch_video_qwen_vl_4videos_review` | 四视频关键帧 OCR 与 Qwen-VL 视觉理解批次 | 4 个视频、20 张关键帧、20 次 PaddleOCR、20 次 Qwen-VL |
| `output/batch_video_deepseek_text_reanalysis_review` | 复用 OCR / 视觉证据，重跑 DeepSeek 文本分析 | 3 个视频成功，1 个视频因 DeepSeek 空响应失败 |
| `output/batch_video_deepseek_text_reanalysis_file0001_retry` | 只补跑 `例子.mp4` | 成功输出 `other` |
| `evaluation/video_topic_gold.csv` | 用户确认的视频人工分类基准 | 记录 3 个边界样本 |

## 3. 人工基准与 DeepSeek 结果

| 文件 | 人工基准 | DeepSeek 结果 | 判断 |
|---|---|---|---|
| `例子.mp4` | `other` | `other` | 通过 |
| `例子2.mp4` | `technology`，无 `entertainment` 副分类 | `technology`，副分类为空 | 通过 |
| `例子3.mp4` | `finance_business`，无 `technology` 副分类 | `finance_business`，副分类为空 | 通过 |

这说明此前分类错误主要来自 mock 文本分析，不是 Qwen-VL 视觉理解本身，也不是 DeepSeek 规则必然失败。

## 4. 成本与延迟口径

| 项目 | 数值 | 说明 |
|---|---:|---|
| Qwen-VL 视觉理解估算成本 | 0.023587 元 | 来自四视频批次的 20 次真实 Qwen-VL 调用 |
| DeepSeek 重分析估算成本，含失败调用 | 0.015877 元 | 首次四文件重分析 0.013258 元 + `例子.mp4` 补跑 0.002619 元 |
| 当前视频链路真实 API 估算成本，含失败调用 | 0.039464 元 | Qwen-VL + DeepSeek；不含本地 PaddleOCR CPU 成本 |
| PaddleOCR 外部 API 成本 | 0 元 | 本地模型运行，但消耗本机 CPU 时间 |
| Qwen-VL P95 延迟 | 5907 ms | 来自四视频批次 |
| PaddleOCR P95 延迟 | 17672 ms | 来自四视频批次，是当前本地 OCR 明显慢点 |
| DeepSeek 文本分析 P95 延迟 | 7910 ms | 来自首次四文件重分析批次 |
| `例子.mp4` 补跑 DeepSeek 延迟 | 3998 ms | 单文件补跑成功 |

注意：`cost_cny` 是基于本地价格表和模型用量估算的成本，不等于供应商后台真实扣费。真实扣费仍需要成本对账闭环验证。

## 5. 当前能证明什么

1. 多视频输入可以进入同一批处理入口。
2. 每个视频可以抽取最多 5 张等距关键帧。
3. 每张关键帧可以分别进入 PaddleOCR 和 Qwen-VL。
4. `model_calls.jsonl` 能记录每次模型调用的供应商、模型、成本、延迟和状态。
5. `results_readable.md` 能展开 OCR 文字和视觉理解描述，支持人工复核。
6. DeepSeek 可以基于 OCR 与视觉理解证据修正 mock 分类错误。

## 6. 当前不能证明什么

1. 不能证明完整视频理解质量，因为真实 ASR 尚未完成小样本验证。
2. 不能证明大规模稳定性，因为当前只有 4 个视频样本。
3. 不能证明供应商真实扣费误差，因为当前仍是本地价格表估算。
4. 不能证明 OCR 已满足生产延迟，因为 PaddleOCR P95 延迟仍明显偏高。
5. 不能证明所有视频分类都稳定，因为人工基准只有 3 个边界样本。

## 7. 下一步闸门

进入真实 ASR 前，当前视频证据链已经达到最低条件：

- 关键帧抽取可复现；
- OCR 和视觉理解都能产出证据；
- DeepSeek 能基于证据产出正确边界分类；
- 成本和延迟能记录到批次报告；
- 真实 / mock 边界已经写清楚。

下一步可以进入真实 ASR 小样本闭环，但范围应限制为：

1. 只选 1 到 2 个视频；
2. 只接一个 ASR 后端；
3. 只验证音频转写是否能进入 `audio_transcript`；
4. 不同时扩展新供应商、动态路由或更大样本。

## 8. DashScope ASR 受保护入口

当前代码已经补入 DashScope Paraformer 录音文件识别入口，但默认仍保持 mock，不会自动发起网络请求。

受保护规则：

1. 必须显式传入 `--speech-backend dashscope_asr`。
2. 必须显式传入 `--allow-live-api`。
3. 程序会优先使用 `--asr-audio-url-map` 中已有的远端音频 URL；如果没有映射，会自动把本地 wav 上传到 DashScope 临时存储，再把返回的 `oss://` URL 交给 ASR。
4. 如果 DashScope SDK/CLI 缺失、上传失败或 ASR 返回异常，系统会把 `audio_transcript` 记录为缺失证据。

如需跳过自动上传，也可以手动准备一个本地不提交版本库的映射文件，例如 `config/asr_audio_url_map.local.json`：

```json
{
  "例子.mp4": "https://你的可访问音频地址/例子_audio.wav"
}
```

受控运行命令：

```powershell
.\.venv\Scripts\python.exe .\src\main.py --input-dir .\input\sample_videos --include-files "例子.mp4" --ocr-backend mock --vision-backend mock --speech-backend dashscope_asr --asr-audio-url-map .\config\asr_audio_url_map.local.json --allow-live-api --batch-id batch_video_dashscope_asr_file0001_trial
```

默认自动上传本地音频的受控运行命令：

```powershell
.\.venv\Scripts\python.exe .\src\main.py --input-dir .\input\sample_videos --include-files "例子.mp4" --ocr-backend mock --vision-backend mock --speech-backend dashscope_asr --allow-live-api --batch-id batch_video_dashscope_asr_file0001_trial
```

这条命令只用于验证真实 ASR 能否把音频转写写入 `audio_transcript`。OCR、视觉理解和文本分析建议先保持 mock，避免一次运行混入多个真实模型变量。

## 9. 字段说明

| 字段 | 含义与作用 |
|---|---|
| `batch_id` | 批次编号，用于区分每次处理或重分析输出 |
| `source_batch_id` | 来源批次编号，用于说明重分析结果复用了哪批历史 OCR / 视觉证据 |
| `topic` | 主分类，表示内容最主要的业务归属 |
| `secondary_topics` | 副分类，表示真实交叉领域；不能来自平台 UI、搜索页或点赞评论数据 |
| `ocr_text` | OCR 识别文字，是视频关键帧文字证据 |
| `visual_description` | 视觉理解模型生成的画面描述，是视频画面语义证据 |
| `audio_transcript` | 音频转写文本；当前已有真实 ASR 入口，但尚未完成真实小样本验证，因此仍是缺口 |
| `speech_to_text_backend` | 语音识别后端选择，用于决定继续使用 mock 还是调用 DashScope ASR |
| `asr_audio_url_map` | 可选音频 URL 映射，用于在已有远端音频地址时跳过自动上传 |
| `missing_evidence` | 缺失证据列表，用于解释为什么结果仍是部分成功 |
| `processing_status` | 文件级处理状态；缺少真实音频证据时应保守使用 `partial_success` |
| `model_calls.jsonl` | 模型调用明细文件，用于追踪每次调用的模型、成本、延迟和错误 |
| `cost_cny` | 单次模型调用估算成本，仍需供应商账单对账验证 |
| `latency_ms` | 单次模型调用耗时，用于分析延迟瓶颈 |
