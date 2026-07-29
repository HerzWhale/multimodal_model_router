# 路由策略预检查报告

生成时间：2026-07-29T15:33:04+08:00

说明：本报告只读取本地配置和可选估算输入，不触发 DeepSeek、PaddleOCR 或任何外部模型调用。

策略名称：`balanced`
预检查状态：`fail`

## 0. 运行前规模画像

| 指标 | 值 | 含义 |
|---|---:|---|
| 输入文件总数 | 12 | 本次预检查纳入的文件数量 |
| 文本文件数 | 1 | 会直接进入文本读取与文本分析的文件数 |
| 图片文件数 | 10 | 会触发 OCR 和视觉理解的图片数 |
| 视频文件数 | 1 | 会触发抽帧、语音识别、OCR 和视觉理解的视频数 |
| 文件总大小 | 52200254 bytes | 用于粗略判断批次规模 |
| 估算文本 token | 189 | 基于文本字符数粗估，不等于供应商真实计费 token |

预估任务用量：

| 任务类型 | 预估用量 | 含义 |
|---|---|---|
| ocr | image_count=13 | 用于运行前预算估算的任务单位 |
| visual_understanding | frame_count=13 | 用于运行前预算估算的任务单位 |
| speech_to_text | audio_seconds=60 | 用于运行前预算估算的任务单位 |
| text_analysis | input_tokens=3989；output_tokens=3600 | 用于运行前预算估算的任务单位 |
| summary_merge | input_tokens=0；output_tokens=0 | 用于运行前预算估算的任务单位 |

画像风险提示：
- 图片 OCR 与视觉理解的单位按图片张数估算；真实耗时还会受分辨率、小字号和版面复杂度影响。
- 文本 token 数使用字符数粗估；真实供应商计费 token 可能存在偏差。

## 0.1 历史延迟画像

| 任务类型 | 调用数 | 真实调用数 | mock调用数 | 平均延迟 | P95延迟 | 最大延迟 | 模型 |
|---|---:|---:|---:|---:|---:|---:|---|
| ocr | 5 | 3 | 2 | 10804 ms | 28261 ms | 28261 ms | doubao/mock-ocr, paddlepaddle/PP-OCRv5_mobile |
| speech_to_text | 1 | 0 | 1 | 0 ms | 0 ms | 0 ms | doubao/mock-asr |
| text_analysis | 23 | 20 | 3 | 3214 ms | 7112 ms | 8141 ms | deepseek/deepseek-v4-flash, deepseek/mock-text |
| visual_understanding | 5 | 0 | 5 | 0 ms | 0 ms | 0 ms | qwen/mock-vision |

延迟画像风险提示：
- 历史延迟来自多个批次，批次输入规模和运行环境可能不同，不能等同于下一批真实延迟。
- 以下任务的历史延迟包含 mock 调用：ocr, speech_to_text, text_analysis, visual_understanding；这些延迟不能代表真实供应商性能。

## 1. 当前路由摘要

| 指标 | 值 | 含义 |
|---|---:|---|
| 预期任务类型数 | 5 | 本次预检查覆盖的任务类型数量 |
| 真实任务类型 | ocr, text_analysis, summary_merge | 当前配置中非 mock 的任务类型 |
| mock 任务类型 | visual_understanding, speech_to_text | 当前仍为占位流程的任务类型 |
| 缺失路由任务 | 当前数据未提供 | 没有配置供应商和模型的任务类型 |
| 真实模型覆盖率 | 60.00% | 非 mock 任务占预期任务的比例 |
| 预估总成本 | 0.171279 元 | 基于传入预估用量计算；缺数据则不硬算 |
| 最大预估 P95 延迟 | 28261 ms | 当前可用延迟数据中的最高 P95 |

## 2. 路由明细

| 任务类型 | 供应商 | 模型 | mock? | 价格状态 | 预估成本 | P95 延迟 | 风险说明 |
|---|---|---|---|---|---:|---:|---|
| ocr | paddlepaddle | PP-OCRv5_mobile | 否 | known | 0.000000 元 | 28261 ms | 无 |
| visual_understanding | qwen | mock-vision | 是 | known | 0.130000 元 | 0 ms | 当前路由仍是 mock，只能证明流程可走通，不能证明真实供应商质量、成本或延迟。；该上游证据提取任务仍是多模态质量瓶颈，不能把下游文本分析结果解释为完整真实多模态能力。 |
| speech_to_text | doubao | mock-asr | 是 | known | 0.030000 元 | 0 ms | 当前路由仍是 mock，只能证明流程可走通，不能证明真实供应商质量、成本或延迟。；该上游证据提取任务仍是多模态质量瓶颈，不能把下游文本分析结果解释为完整真实多模态能力。 |
| text_analysis | deepseek | deepseek-v4-flash | 否 | known | 0.011279 元 | 7112 ms | 无 |
| summary_merge | deepseek | deepseek-v4-flash | 否 | known | 0.000000 元 | 当前数据未提供 | 当前没有提供本任务的历史或目标前估 P95 延迟，不能判断延迟约束。 |

## 3. 约束检查

| 约束 | 观测值 | 限制值 | 状态 | 说明 |
|---|---:|---:|---|---|
| routing_rules_complete | 0 | 0 | pass | 满足约束。 |
| budget_limit_cny | 0.171279 | 50 | pass | 满足约束。 |
| p95_latency_limit_ms | 28261 | 3500 | fail | 未满足约束，运行前需要处理。 |
| min_real_coverage_rate | 0.6 | 0.4 | pass | 满足约束。 |

## 4. 阻塞与风险

阻塞原因：
- p95_latency_limit_ms 未满足：未满足约束，运行前需要处理。

风险提示：
- 当前仍有 mock 任务：visual_understanding, speech_to_text。这些任务不能证明真实模型能力。

## 5. 受控小样本试跑建议

决策：`shrink_scope_before_running`

原因：当前预算约束通过，但 P95 延迟约束失败；不应直接跑完整 input，应先缩小范围定位慢点。

| 范围项 | 建议值 | 含义 |
|---|---:|---|
| 最大总文件数 | 3 | 延迟问题未定位前，本轮最多处理的文件数 |
| 最大文本文件数 | 1 | 用于观察文本分析链路和 DeepSeek 延迟 |
| 最大图片文件数 | 2 | 用于观察本地 OCR 延迟和错误状态 |
| 最大视频文件数 | 0 | 当前先不纳入视频，避免混入视频预处理和 mock 边界 |
| 最大真实 API 文件数 | 1 | 如需调用 DeepSeek，本轮最多纳入的文件数 |

范围理由：延迟阻塞未解除前，先用最多3个文件验证链路；暂不纳入视频，避免把 OCR、视频预处理和 mock 边界混在一起。

建议 include-files：`ai_content_sample.txt, img.png, img_1.png`

慢任务证据：

| 任务类型 | P95延迟 | 真实调用数 | mock调用数 | 含义 |
|---|---:|---:|---:|---|
| ocr | 28261 ms | 3 | 2 | 用于判断该任务是否是继续扩大运行前的延迟阻塞 |
| text_analysis | 7112 ms | 20 | 3 | 用于判断该任务是否是继续扩大运行前的延迟阻塞 |

建议命令：

- `offline_mock_trial`（不需要真实 API）：先验证指定文件范围、文件分流、结果写入和报告生成，不触发真实模型。

```powershell
python .\src\main.py --input-dir .\input --include-files ai_content_sample.txt,img.png,img_1.png --ocr-backend mock --text-analysis-backend mock --batch-id batch_controlled_mock_trial
```

- `local_ocr_trial`（不需要真实 API）：只放开本地 PaddleOCR，观察 OCR 延迟和错误状态，不触发 DeepSeek API。

```powershell
python .\src\main.py --input-dir .\input --include-files ai_content_sample.txt,img.png,img_1.png --ocr-backend paddleocr --text-analysis-backend mock --batch-id batch_controlled_paddleocr_trial
```

- `deepseek_text_trial`（需要真实 API 授权）：只用少量文本文件验证 DeepSeek 文本分析延迟；必须单独授权 API 调用。

```powershell
python .\src\main.py --input-dir .\input --include-files ai_content_sample.txt --ocr-backend mock --text-analysis-backend deepseek --allow-live-api --batch-id batch_controlled_deepseek_text_trial
```

不要做：
- 不要直接处理完整 input 目录。
- 不要在同一轮试跑里同时放开 PaddleOCR 和 DeepSeek 大量真实调用。
- 不要把 visual_understanding 或 speech_to_text 的 0ms mock 延迟解释为真实供应商性能。

成功标准：
- 受控试跑能生成 batch_metadata、results、model_calls 和 batch_report。
- model_calls 中能分清真实调用和 mock 调用。
- OCR 慢点和 DeepSeek 慢点能分别观察，不能混成一个总耗时结论。
- 如果小批量仍超过 P95 目标，不扩大运行范围。

试跑后的判断：
- 如果 OCR 仍明显超过延迟目标，保留本地 OCR 基线，但不要把它写成生产可用 OCR。
- 如果 DeepSeek 文本分析仍超过延迟目标，文本链路可以继续做质量评估，但不承诺在线低延迟。
- 如果受控试跑通过，再逐步增加文件数；每次只增加一个变量。

## 6. 建议动作

暂不建议直接扩大运行；先处理失败约束：p95_latency_limit_ms。

## 7. 边界说明

- 路由预检查不调用真实模型，因此不能产生新的质量结论。
- 预算检查只有在提供预估用量时才有意义；缺少用量时不会硬算总成本。
- 延迟检查只有在提供历史或目标前估 P95 延迟时才有意义；缺少延迟数据时只给出未知状态。
- 本模块不会自动修改 routing_rules.yaml，也不会替代运行时模型路由器。

## 8. 字段说明

| 字段 | 含义与作用 |
|---|---|
| `workload_profile` | 运行前规模画像，用于在不调用模型的情况下统计输入文件规模，并估算各任务会消耗的计量单位。 |
| `latency_profile` | 历史延迟画像，用于从已有模型调用记录中提取任务级 P95 延迟，帮助运行前判断延迟约束。 |
| `expected_units_by_task` | 按任务类型整理的预估用量，用于把单位价格转换成整批预算预估；缺少该字段时不会硬算总成本。 |
| `historical_p95_latency_by_task_ms` | 按任务类型整理的历史 P95 延迟，用于把历史调用经验带入运行前延迟预检查。 |
| `budget_limit_cny` | 本次预算上限，用于判断当前模型组合在预估用量下是否可能超出人民币预算。 |
| `p95_latency_limit_ms` | P95 延迟限制，用于判断最慢的高分位任务延迟是否超过业务目标。 |
| `min_real_coverage_rate` | 最低真实模型覆盖率，用于判断当前路线中真实模型任务占比是否过低。 |
| `current_route` | 当前每个任务类型会走向哪个供应商和模型，用于运行前核对实际模型组合。 |
| `preflight_status` | 预检查总状态；pass 表示未发现阻塞，warning 表示可试跑但有未知或 mock 风险，fail 表示不建议直接运行。 |
| `blocking_reasons` | 阻塞原因列表，用于说明为什么当前配置不应直接进入扩大运行。 |
| `warning_messages` | 风险提示列表，用于说明哪些地方可以继续试跑但不能过度解读。 |
| `estimated_cost_scope` | 成本估算范围说明，用于区分单位价格检查、预估成本和真实账单。 |
| `controlled_trial_plan` | 受控小样本试跑建议，用于在预算可接受但延迟失败或仍有 mock 边界时，说明下一轮应缩小到哪些范围、怎么试跑、哪些结论不能越界。 |
| `suggested_include_files` | 建议传给 `--include-files` 的文件名列表，用于只处理少量代表性文件，避免误跑完整输入目录。 |
| `trial_commands` | 受控试跑命令建议，只作为人工执行参考；报告生成本身不会执行这些命令，也不会触发模型调用。 |
