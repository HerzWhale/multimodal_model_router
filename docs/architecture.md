# Architecture

这份文档说明当前系统如何拆分模块、如何处理文本/图片/视频、如何记录模型调用，以及当前架构的真实边界。

## 1. 项目整体架构

```text
config/
  ├─ settings.yaml          运行配置
  ├─ routing_rules.yaml     任务到供应商/模型的路由配置
  └─ model_prices.yaml      模型计价配置

input/
  ├─ sample_text/           文本样例
  ├─ sample_images/         图片样例
  └─ sample_videos/         视频样例

src/
  ├─ main.py                批处理入口
  ├─ file_loader.py         文件扫描与文件记录生成
  ├─ preprocessor.py        文本、图片、视频预处理
  ├─ model_router.py        按任务类型选择模型
  ├─ model_clients.py       mock 模型与 DeepSeek 文本分析调用
  ├─ pipeline_runner.py     单文件流水线调度
  ├─ cost_latency_tracker.py 成本和延迟记录生成
  ├─ result_writer.py       输出文件写入
  ├─ report_generator.py    批次报告生成
  └─ model_strategy_advisor.py 模型组合策略报告生成

output/
  └─ batch_xxx/             每次批处理的输出目录
```

整体数据流：

```text
输入文件
  ↓
file_loader 生成文件清单
  ↓
pipeline_runner 根据 media_type 选择流水线
  ↓
preprocessor 准备文本、图片路径、视频关键帧和音频占位
  ↓
model_router / model_clients 完成模型选择与调用
  ↓
cost_latency_tracker 记录调用级成本和延迟
  ↓
result_writer 写入结果、调用明细和错误索引
  ↓
report_generator 汇总批次统计报告
  ↓
model_strategy_advisor 基于已有批次生成模型组合建议
```

## 2. 核心模块说明

| 模块 | 作用 | 当前边界 |
|---|---|---|
| `main.py` | 批处理入口，读取配置、运行文件清单、写入输出文件 | 命令行入口，没有 Web 页面 |
| `file_loader.py` | 扫描输入目录，生成文件级元数据 | 支持本地文件，不支持云存储 |
| `preprocessor.py` | 根据文件类型做基础预处理 | 视频关键帧和音频处理仍是占位 |
| `model_router.py` | 根据任务类型选择供应商和模型 | 当前是固定路由，不是动态推荐 |
| `model_clients.py` | 封装 mock 模型和 DeepSeek 文本分析 | 真实接入只覆盖 DeepSeek 文本分析 |
| `pipeline_runner.py` | 调度单个文件的处理流程，生成文件级结果 | 当前主要验证流水线和追踪能力 |
| `cost_latency_tracker.py` | 生成模型调用成本和延迟记录 | 成本依赖本地价格配置 |
| `result_writer.py` | 写入 JSON、JSONL 和 Markdown 输出 | 当前采用本地文件输出 |
| `report_generator.py` | 汇总文件、成本、延迟、错误和质量统计 | 当前报告以批次为粒度 |
| `model_strategy_advisor.py` | 基于既有批次报告和模型调用明细生成模型组合策略报告 | 只做离线分析，不触发外部 API，不自动调度真实多供应商模型 |

## 3. 文本、图片、视频三条处理流程

### 文本流程

```text
文本文件
  ↓
文本读取
  ↓
raw_text
  ↓
DeepSeek 文本分析
  ↓
topic / secondary_topics / tags / summary / business_use
```

字段说明：

| 字段 | 含义与作用 |
|---|---|
| `raw_text` | 文本文件原文，是文本分析模型的主要证据来源 |
| `topic` | 主分类，表示内容最主要的业务归属 |
| `secondary_topics` | 副分类，表示最多两个交叉领域 |
| `tags` | 关键词，用于搜索、筛选和素材管理 |
| `summary` | 内容摘要，用于快速理解文件内容 |
| `business_use` | 业务用途说明，用来解释结构化结果能支持什么动作 |

### 图片流程

```text
图片文件
  ↓
OCR mock ─────────┐
视觉理解 mock ────┤
                  ↓
          DeepSeek 文本分析
                  ↓
topic / secondary_topics / tags / summary / business_use
```

字段说明：

| 字段 | 含义与作用 |
|---|---|
| `ocr_text` | OCR 识别出的文字证据；当前为 mock |
| `visual_description` | 视觉理解模型生成的画面描述；当前为 mock |
| `evidence_used` | 最终分析实际使用的证据列表，用来判断结果依据是否完整 |
| `missing_evidence` | 缺失证据列表，用来解释结果风险 |

### 视频流程

```text
视频文件
  ↓
视频预处理占位
  ↓
关键帧 / 音频
  ↓
OCR mock ─────────────┐
视觉理解 mock ────────┤
语音识别 mock ────────┤
                      ↓
              DeepSeek 文本分析
                      ↓
topic / secondary_topics / tags / summary / business_use
```

字段说明：

| 字段 | 含义与作用 |
|---|---|
| `duration_ms` | 视频或音频时长，单位毫秒，用于估算音频相关任务成本和耗时 |
| `audio_transcript` | 语音识别得到的音频转写；当前为 mock |
| `visual_description` | 关键帧视觉描述；当前为 mock |
| `ocr_text` | 关键帧 OCR 文字；当前为 mock |

## 4. 模型路由逻辑

当前模型路由是 MVP 版本：按 `task_type` 固定选择供应商和模型。

| `task_type` 的含义 | 当前供应商/模型 | 真实状态 |
|---|---|---|
| OCR 任务，用于从图片或关键帧提取文字 | `doubao / mock-ocr` | mock |
| 视觉理解任务，用于生成图片或关键帧描述 | `qwen / mock-vision` | mock |
| 语音识别任务，用于把音频转成文本 | `doubao / mock-asr` | mock |
| 文本分析任务，用于生成分类、标签、摘要和业务用途 | `deepseek / deepseek-v4-flash` | 真实调用 |

需要注意：

- `routing_rules.yaml` 中仍保留文本分析的 mock 路由配置，这是为了让本地 mock 流程可运行。
- 实际是否调用 DeepSeek，由 `config/settings.yaml` 中的 `text_analysis_backend` 控制。
- 当前 Demo 中，`text_analysis_backend` 为 `deepseek`，因此文本分析阶段会调用 DeepSeek。
- 当前尚未实现按预算、质量、延迟动态选择模型的路由策略。

阶段 4 新增的策略报告不是替代 `model_router.py` 的实时路由器，而是一个离线决策层：

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `model_router.py` | 单个 `task_type`，也就是当前要做的任务类型 | 当前任务使用的供应商和模型 | 在批处理运行时决定这一步调用哪个模型 |
| `model_strategy_advisor.py` | 已有 `batch_report.json` 和 `model_calls.jsonl` | `model_strategy_report.md` 和 `model_strategy_report.json` | 在批处理完成后分析成本、延迟、真实/mock 边界，并给出下一步模型组合建议 |

相关字段说明：

| 字段 | 含义与作用 |
|---|---|
| `model_call_count` | 模型调用次数，用来衡量本批次任务链路规模 |
| `is_mock` | 是否为 mock 调用，用来区分真实模型证据和占位流程证据 |
| `cost_share` | 成本占比，用来判断某个任务、供应商或模型是否构成主要成本来源 |

## 5. 成本与延迟追踪逻辑

系统将成本和延迟拆成两个层级：

| 层级 | 记录位置 | 用途 |
|---|---|---|
| 单次模型调用 | `model_calls.jsonl` | 追踪每次调用用了哪个模型、多少输入输出、多少钱、耗时多久 |
| 单个文件结果 | `results.jsonl` / `results_readable.md` | 汇总该文件所有调用成本和处理耗时 |
| 整个批次 | `batch_report.json` | 汇总批次总成本、平均成本、P95 延迟、成功率和错误质量统计 |

关键字段说明：

| 字段 | 含义与作用 |
|---|---|
| `call_id` | 单次模型调用唯一标识，用于排查和追踪调用链路 |
| `input_units` | 输入用量及单位，用于成本估算，例如输入 token、图片数量、帧数或音频秒数 |
| `output_units` | 输出用量及单位，用于成本估算，例如输出 token 或文本字符数 |
| `cost_cny` | 单次模型调用成本，单位人民币 |
| `latency_ms` | 单次模型调用耗时，单位毫秒 |
| `processing_cost_cny` | 文件级总成本，由该文件关联的多次调用成本累加得到 |
| `processing_time_ms` | 文件级总处理耗时，用于判断用户等待时间 |
| `budget_used_rate` | 批次预算使用率，用于判断本次处理是否接近预算上限 |

当前 Demo 的重要统计：

- 总成本：0.042107 元。
- 平均每文件成本：0.014036 元。
- DeepSeek 文本分析成本：0.002107 元。
- 平均文件处理耗时：2154.33 ms。
- 模型调用 P95 延迟：3425 ms。

这些数字能展示成本和延迟追踪能力，但不能用于声称多个真实供应商之间的性能对比，因为首期没有真实接入多个供应商。

## 6. 输出文件生成逻辑

每次运行都会生成一个批次目录：

```text
output/{batch_id}/
```

其中 `batch_id` 是批次唯一标识，用来把同一次处理中的多个文件和模型调用归在一起。

输出文件说明：

| 文件 | 作用 |
|---|---|
| `batch_metadata.json` | 保存批次元数据，例如批次编号、创建时间、创建人、预算和输出格式 |
| `results.jsonl` | 保存文件级最终结构化结果，适合机器读取和下游导入 |
| `results_readable.md` | 保存人工可读的结果说明，适合 Demo 展示 |
| `model_calls.jsonl` | 保存模型调用明细，是成本、延迟和调用链追踪的来源 |
| `errors.jsonl` | 保存错误索引，用于集中排查失败或部分成功 |
| `batch_report.json` | 保存批次级统计报告，用于查看成本、延迟、成功率和质量风险 |
| `model_strategy_report.md` | 保存人工可读的模型组合策略报告，用于解释成本、延迟、真实/mock 边界和下一步建议 |
| `model_strategy_report.json` | 保存结构化模型组合策略报告，方便后续接入前端、数据库或自动化分析 |

文件级结果和模型调用之间的关系：

```text
batch_id：一次批处理
  └── file_id：批处理中的单个文件
        └── call_id：该文件触发的某次模型调用
```

字段说明：

| 字段 | 含义与作用 |
|---|---|
| `batch_id` | 批次唯一标识，用来聚合同一次处理中的文件和调用记录 |
| `file_id` | 文件唯一标识，用来关联文件结果、模型调用和错误记录 |
| `call_ids` | 文件级结果关联的多个模型调用 ID，用来从结果反查调用链 |
| `processing_status` | 文件级处理状态，用来判断成功、失败、部分成功或跳过 |
| `error_message` | 技术错误信息，用于开发者排查失败原因 |
| `warning_messages` | 面向使用者的风险提示，用于解释结果可信度受到什么影响 |

## 7. 决策层报告生成逻辑

`model_strategy_advisor.py` 是阶段 4 新增的离线决策层模块。它不重新运行批处理，也不调用 DeepSeek 或其他供应商 API，只读取已有 Demo 批次中的两个文件：

| 输入文件 | 含义与作用 |
|---|---|
| `batch_report.json` | 批次级统计结果，用来读取文件数、成功率、总成本、平均处理耗时和 P95 延迟 |
| `model_calls.jsonl` | 单次模型调用明细，用来读取每次调用的供应商、模型、任务类型、成本、延迟和状态 |

生成输出：

| 输出文件 | 含义与作用 |
|---|---|
| `model_strategy_report.md` | 给人看的策略报告，适合 README、作品集和面试展示 |
| `model_strategy_report.json` | 给程序看的结构化策略报告，便于后续接入前端或数据库 |

分析逻辑：

1. 从批次报告读取文件数、成功率、总成本、平均延迟和 P95 延迟。
2. 从模型调用明细按任务类型、供应商和模型汇总成本。
3. 按 `latency_ms` 识别最慢调用，并判断 P95 延迟是否由少数调用拉高。
4. 按 `model_name` 是否以 `mock` 开头识别真实模型调用和 mock 调用边界。
5. 输出预算敏感、延迟敏感、质量优先三类模型组合建议。

字段说明：

| 字段 | 含义与作用 |
|---|---|
| `latency_ms` | 单次模型调用延迟，单位毫秒，用来识别最慢调用和瓶颈 |
| `model_name` | 具体模型名称，用来识别真实模型和 mock 模型 |
| `real_model_calls` | 真实模型调用汇总，用来说明当前 Demo 可作为真实证据的范围 |
| `mock_model_calls` | mock 调用汇总，用来说明当前 Demo 只能证明流程跑通的范围 |
| `missing_data_notes` | 数据缺失说明，用来保证字段缺失时不硬算、不编造 |

## 8. 当前架构限制

- 真实模型能力只覆盖 DeepSeek 文本分析。
- OCR、视觉理解和语音识别是 mock，不能代表真实图片或视频理解质量。
- 视频预处理仍是占位，尚未进行真实关键帧抽取和音频转写。
- 模型路由是固定规则，尚未根据预算、质量要求、延迟目标动态选择模型。
- 决策层报告基于已有批次数据生成，不能替代真实多供应商 live test。
- 当前没有数据库层，输出主要写入本地 JSON、JSONL 和 Markdown 文件。
- 当前没有前端页面，展示主要依赖输出文件和后续截图材料。
- 当前 Demo 样本较小，还不足以证明大规模稳定性。

## 9. 后续可扩展方向

| 方向 | 价值 |
|---|---|
| 接入真实 OCR 或视觉理解模型 | 让图片流程从流程验证变成真实多模态能力展示 |
| 接入真实语音识别 | 让视频流程具备更可信的音频证据 |
| 增加失败和部分成功样例 | 展示证据缺失、上游失败和错误追踪能力 |
| 增加动态模型路由 | 把当前离线策略报告升级为运行时按预算、延迟和质量要求选择模型组合 |
| 增加轻量前端或展示页 | 降低招聘方理解门槛 |
| 增加更大样本 Demo | 展示批量处理、成本汇总和延迟统计更有说服力 |
| 整理 GitHub 发布物 | 增加 `.gitignore`、截图、README 链接和可复现说明 |

## 10. 相关文档

| 文档 | 作用 |
|---|---|
| `docs/demo_walkthrough.md` | 解释 Demo 输入、运行方式、输出读法、成本延迟和真实/mock 边界 |
| `docs/portfolio_showcase.md` | 给招聘方快速浏览的 3 分钟展示版 |
| `docs/tests.md` | 说明已有离线测试、测试命令、覆盖范围、风险缺口和后续测试计划 |
| `docs/release_checklist.md` | GitHub 发布前检查清单，用于避免误提交缓存、密钥或错误输出 |
