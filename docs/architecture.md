# Architecture

这份文档说明当前系统如何拆分模块、如何处理文本 / 图片 / 视频、如何记录模型调用，以及当前架构的真实边界。

项目定位保持为：面向内容平台 AI 团队技术负责人的多模态批处理与模型路由 MVP。文本主分类评估只是当前最扎实的一条验证链路，不代表项目被收缩成“只能做文本分类”。

## 1. 项目整体架构

```text
config/
  ├─ settings.yaml                 运行配置
  ├─ routing_rules.yaml            任务到供应商 / 模型的固定路由配置
  ├─ model_prices.yaml             模型计价配置，包含价格来源、更新时间和可信度
  └─ routing_policy_config.yaml    离线路由策略约束配置

evaluation/
  ├─ text_topic_small_set/         文本主分类小样本评估集
  └─ text_topic_gold.csv           文本主分类人工标准答案

src/
  ├─ main.py                       批处理入口
  ├─ file_loader.py                文件扫描与文件记录生成
  ├─ preprocessor.py               文本、图片、视频预处理
  ├─ model_router.py               按任务类型选择模型
  ├─ model_clients.py              mock 模型、本地 PaddleOCR、Qwen-VL 图片/关键帧理解与 DeepSeek 文本分析
  ├─ pipeline_runner.py            单文件流水线调度
  ├─ cost_latency_tracker.py       成本和延迟记录生成
  ├─ result_writer.py              输出文件写入
  ├─ report_generator.py           批次报告生成
  ├─ model_strategy_advisor.py     模型组合策略报告生成
  ├─ model_catalog.py              从调用明细聚合模型目录
  ├─ routing_policy.py             离线路由策略与约束判断
  ├─ routing_preflight.py          批处理前路由策略预检查
  ├─ offline_regression_check.py   受保护离线回归检查入口
  ├─ strategy_simulator.py         路由策略离线模拟报告生成
  ├─ cost_reconciliation.py        多供应商成本对账模板和报告生成
  ├─ price_catalog_updater.py      官方公开价格页抓取与本地价格目录刷新
  ├─ cost_repricing.py             按当前价格目录重算历史批次成本
  └─ text_topic_evaluator.py       文本主分类评估报告生成

output/
  └─ batch_xxx/                    每次批处理的输出目录
```

目录边界：

| 目录 | 含义 | 当前规则 |
|---|---|---|
| `input/` | 默认业务输入目录，用来模拟用户交给系统处理的一批文本、图片或视频文件 | 默认运行 `python .\src\main.py` 时读取 |
| `evaluation/` | 评估样本目录，用来验证模型输出是否命中人工标准答案 | 不会被默认流程自动读取，必须用 `--input-dir` 显式指定 |
| `output/` | 批处理输出目录，用来保存结果、调用记录、批次报告和评估报告 | 每次运行生成或复用指定的 `batch_id` |
| `local_notes/` | 本地整理记录目录，用来保存历史清理报告和文件盘点说明 | 不参与程序运行，不作为测试输入，后续公开仓库可整体排除 |

核心配置边界：

| 配置文件 | 含义 | 作用 |
|---|---|---|
| `config/settings.yaml` | 运行环境配置 | 管理输入输出目录、默认后端、API地址、模型名、token上限和预算 |
| `config/runtime_policy.yaml` | 运行策略配置 | 管理文件类型白名单、后端白名单、视频预处理参数、OCR质量闸门、topic体系和DeepSeek提示词 |
| `config/model_prices.yaml` | 模型价格配置 | 管理模型计价单位、价格来源、更新时间和价格可信度 |
| `config/routing_rules.yaml` | 固定路由配置 | 管理不同任务类型默认选择哪个供应商和模型 |

字段说明：

| 字段 | 含义与作用 |
|---|---|
| `input_dir` | 本次批处理读取的输入目录，用来区分普通业务输入和受控评估样本 |
| `batch_id` | 一次批处理任务的唯一标识，用于把本次输入文件、模型调用和输出结果归到同一批次 |
| `gold_topic` | 人工标注的正确主分类，只存在于评估流程中，用来计算文本主分类准确率 |
| `ocr_backend` | 图片或视频关键帧 OCR 后端配置，用来决定使用 mock 还是本地 PaddleOCR |
| `vision_understanding_backend` | 图片或视频关键帧视觉理解后端配置，用来决定使用 mock 还是 Qwen-VL API |
| `text_analysis_backend` | 文本分析后端配置，用来决定使用本地 mock 还是 DeepSeek 真实 API |
| `--allow-live-api` | 真实 API 调用授权开关；没有该开关时，即使配置选择 DeepSeek 或 Qwen-VL，也会在网络请求前停止 |
| `--max-api-retries` | 可重试错误的最大重试次数；默认0，只允许显式设为1，避免意外增加费用 |
| `--include-files` | 指定本次只处理哪些文件名，用于受控评估少量图片，避免误跑整个输入目录 |

PaddleOCR 在本地运行，不使用外部 API 授权开关。选择 PaddleOCR 时，未显式选择的文本分析后端强制使用 mock；选择 DeepSeek 或 Qwen-VL 时，未显式选择的其他后端同样强制使用 mock，避免一次命令意外执行多条非默认路径。

整体数据流：

```text
输入文件
  ↓
file_loader 生成文件清单
  ↓
pipeline_runner 根据媒体类型选择流水线
  ↓
preprocessor 准备文本、图片路径、视频元信息和最多 5 张等距关键帧
  ↓
model_router / model_clients 完成模型选择与调用
  ↓
cost_latency_tracker 记录调用级成本和延迟
  ↓
result_writer 写入结果、调用明细和错误索引
  ↓
report_generator 汇总批次统计报告
  ↓
model_strategy_advisor / strategy_simulator 做离线策略分析
  ↓
text_topic_evaluator 基于人工标准答案做文本主分类评估
```

## 2. 核心模块说明

| 模块 | 作用 | 当前边界 |
|---|---|---|
| `main.py` | 读取配置、检查 PaddleOCR 本地依赖、执行 DeepSeek 和 Qwen-VL API 安全校验并运行批处理 | 默认使用 mock；PaddleOCR 必须显式选择，DeepSeek 和 Qwen-VL 还必须显式授权 |
| `file_loader.py` | 扫描输入目录，生成文件级元数据 | 支持本地文件，不支持云存储 |
| `preprocessor.py` | 根据文件类型做基础预处理 | 视频 V1 已能读取元信息并等距抽取最多 5 张关键帧；本机存在 ffmpeg 时可抽取单声道 16kHz wav 音频文件；缺少依赖或提取失败时记录明确状态 |
| `model_router.py` | 根据任务类型选择供应商和模型 | 当前是固定路由，不是运行时动态推荐 |
| `model_clients.py` | 封装 mock、本地 PaddleOCR、Qwen-VL 图片/关键帧理解和 DeepSeek 文本分析 | Qwen-VL 入口已完成离线测试、`img_1.png` 单图真实 API 验证和可重试错误最多1次重试；视频关键帧质量仍需后续受控复测 |
| `pipeline_runner.py` | 调度文件流程，记录真实或 mock 调用、证据、成本、延迟、错误和质量风险 | PaddleOCR 和 Qwen-VL 可用于图片和视频 V1 抽出的多张关键帧；Qwen-VL 会把每次尝试分别记入模型调用；默认仍为 mock；真实OCR会经过低质量文本闸门 |
| `cost_latency_tracker.py` | 生成模型调用成本、价格目录元数据和延迟记录 | 成本依赖本地价格配置；模型调用记录会带出价格来源、更新时间和可信度 |
| `price_catalog_updater.py` | 从官方公开价格页抓取 Qwen-VL 和 DeepSeek 人民币价格，并生成刷新报告；带 `--apply` 时写回本地价格目录 | 不登录控制台，不读取真实账单，不调用付费模型API；写回前会检查必要计价单位、正数价格和大幅价格变化，官方页面结构变化或网络失败时会失败并输出错误 |
| `cost_repricing.py` | 读取历史 `model_calls.jsonl` 和当前 `model_prices.yaml`，重新计算历史批次在当前价格目录下的成本 | 不改写历史调用记录，不重跑模型；用于解释价格目录更新后历史成本口径变化 |
| `result_writer.py` | 写入 JSON、标准 JSONL 和 Markdown 输出 | 新 JSONL 每行一条完整记录；历史批次不重写 |
| `report_generator.py` | 汇总文件、成本、延迟、错误和质量统计 | 当前报告以批次为粒度 |
| `model_strategy_advisor.py` | 基于既有批次生成模型组合策略报告，可选择使用历史成本或当前价格目录重算成本 | 离线分析，不触发外部 API；传入成本重算报告时使用 `current_repriced` 口径 |
| `model_catalog.py` | 从调用明细聚合模型目录 | 只基于已有调用记录，不编造未知模型数据 |
| `routing_policy.py` | 定义成本、延迟、质量和平衡策略 | 离线策略判断，不改变运行时模型调用 |
| `routing_preflight.py` | 在批处理前读取当前路由、价格目录、策略约束、可选输入目录和已有模型调用记录，生成运行前风险检查、规模画像、历史延迟画像、任务级延迟目标检查、价格目录画像、延迟阻塞归因和受控小样本试跑建议 | 不调用模型，不自动换模型，不联网刷新价格；预算可基于输入规模估算，P95延迟可基于已有 `model_calls.jsonl` 判断；当配置 `task_latency_targets_ms` 时优先按 OCR、视觉理解、文本分析等任务分别检查，避免用同一个全局目标误杀不同耗时结构的任务；延迟阻塞归因会区分真实外部 API、本地运行和 mock 占位；价格目录画像只判断本地价格是否过期或可信度不足；试跑建议只生成命令参考，不会自动执行 |
| `cost_reconciliation.py` | 根据 `model_calls.jsonl` 生成多供应商手工账单模板，并把估算成本与供应商实际扣费对比 | 不接平台账单 API，不访问网络；首版只支持人工填入统一格式账单金额，并拒绝非法金额和重复账单记录 |
| `strategy_simulator.py` | 基于既有批次生成路由策略模拟报告 | 不触发外部 API，不重新跑批处理 |
| `text_topic_evaluator.py` | 生成文本评估模板并计算 Accuracy、Macro-F1 和分类级指标 | 只评估文本主分类，不评估图片、视频、摘要或标签质量 |
| `image_ocr_evaluator.py` | 按人工业务文字块计算OCR精确召回率、字符错误率、错误归因和批次级闸门 | 只读取已有图片结果，不运行OCR或外部API；当前正式基准有5张图 |
| `image_ocr_preprocessing_experiment.py` | 围绕 `img_9.jpg` 生成最小预处理实验和延迟拆分报告 | 只服务于图片OCR闸门判断，不进入主业务流水线，不新增模型能力 |
| `ocr_backend_advisor.py` | 基于OCR闸门、延迟拆分和候选评估报告生成后端取舍判断 | 只生成离线建议，不接入RapidOCR、Tesseract或云OCR |
| `rapidocr_candidate_evaluator.py` | 对 RapidOCR 候选后端执行可选评估 | 当前已用三张关键帧完成本地实测；依赖缺失时仍会安全输出 `dependency_missing`，不编造指标 |

## 3. 三条处理流程

### 文本流程

```text
文本文件
  ↓
读取 raw_text
  ↓
DeepSeek 文本分析
  ↓
topic / secondary_topics / tags / summary / business_use
```

### 图片流程

```text
图片文件
  ↓
OCR mock / 本地 PaddleOCR ─┐
视觉理解 mock / Qwen-VL ───┤
                  ↓
          OCR 证据质量闸门
                  ↓
          DeepSeek 文本分析
                  ↓
topic / secondary_topics / tags / summary / business_use
```

图片流程中的 OCR 证据质量闸门只处理“调用成功但文字明显不可用”的情况。此时 `ocr_text` 仍会保留在输出中，方便人工复核；但该证据不会进入 `evidence_used`，下游文本分析也不会把它当作可靠输入。文件级 `processing_status` 会变为 `partial_success`，并写入 `quality_flags=low_quality_ocr_text` 和对应 `warning_messages`。

### 视频流程

```text
视频文件
  ↓
视频 V1 本地预处理
  ├─ 读取时长、帧数、FPS、分辨率
  ├─ 等距抽取最多 5 张关键帧，写入 preprocess_artifacts
  └─ 音频提取：本机 ffmpeg 可用时写出 wav，缺失或失败时记录状态
        ↓
多张关键帧
  ├─ OCR mock / 本地 PaddleOCR ─────┐
  └─ 视觉理解 mock / Qwen-VL ───────┤
音频文件 → mock speech_to_text ────┤
音频缺失 → speech_to_text failed ─┤
                                  ↓
                          文本分析
                                  ↓
topic / secondary_topics / tags / summary / business_use
```

字段说明：

| 字段 | 含义与作用 |
|---|---|
| `raw_text` | 文本原文，是文本分析模型的主要证据来源 |
| `ocr_text` | OCR 提取的文字证据；图片和视频 V1 多张关键帧可选择本地 PaddleOCR；无文字时允许为空 |
| `audio_transcript` | 音频转写文字；当前真实 ASR 尚未接入，只有本地音频提取成功后才会进入 mock 语音识别占位分支 |
| `visual_description` | 图片或关键帧的视觉描述；图片和视频 V1 多张关键帧可显式选择 Qwen-VL API |
| `preprocessing_artifacts` | 文件预处理产物摘要，用于记录视频元信息、关键帧抽取状态、音频提取状态和预处理风险 |
| `keyframe_paths` | 视频 V1 抽出的关键帧路径列表，用于追踪视频画面证据来自哪些本地产物 |
| `keyframe_metadata` | 每张关键帧的帧号、时间位置和路径，用于说明视频画面证据覆盖了哪些时间点 |
| `audio_extraction_status` | 音频提取状态，用于说明当前是否已经生成真实音频文件；可能值包括 `extracted`、`dependency_missing`、`failed`、`timeout`、`empty_output` 和 `not_attempted_no_artifact_dir` |
| `evidence_used` | 最终分析实际使用的证据列表；低质量 OCR 文本会保留在 `ocr_text`，但不会进入该列表 |
| `quality_flags` | 机器可读质量风险标签，用于标记用途降级、低质量OCR等可统计问题 |
| `warning_messages` | 面向使用者的风险提示，用于解释结果可信度受到什么影响 |
| `processing_status` | 文件级处理状态；低质量OCR会让图片结果从完全成功降为部分成功 |
| `topic` | 主分类，表示内容最主要的业务归属 |
| `secondary_topics` | 副分类，表示最多两个交叉领域 |
| `tags` | 关键词，用于搜索、筛选和素材管理 |
| `summary` | 内容摘要，用于快速理解文件内容 |
| `business_use` | 业务用途说明，用来解释结构化结果能支持什么动作 |

## 4. 模型路由逻辑

当前运行时路由是 MVP 版本：按 `task_type` 固定选择供应商和模型。

| 任务类型 | 当前供应商 / 模型 | 真实状态 |
|---|---|---|
| 图片 OCR 默认任务 | `doubao / mock-ocr` | mock |
| 图片 OCR 可选任务 | `paddlepaddle / PP-OCRv5_mobile` | 已完成五张正式图片的本地 CPU 推理与分段评估 |
| 视频 OCR 默认任务 | `doubao / mock-ocr` | mock |
| 视频 OCR 可选任务 | `paddlepaddle / PP-OCRv5_mobile` | 可处理视频 V1 抽出的最多 5 张关键帧；尚未形成视频批次质量评估 |
| 图片视觉理解默认任务 | `qwen / mock-vision` | mock |
| 图片视觉理解可选任务 | `qwen / qwen-vl-plus` | 受保护 API 入口已实现，并已完成 `img_1.png` 单图真实验证 |
| 视频关键帧视觉理解可选任务 | `qwen / qwen-vl-plus` | 可处理视频 V1 抽出的最多 5 张关键帧；需要 `--allow-live-api`；支持可重试错误最多重试1次，并逐次记录调用；已有一次视频真实试跑但需复测可靠性 |
| 语音识别任务 | `doubao / mock-asr` | 真实 ASR 未接入；本地音频提取成功时会进入 mock 语音识别分支，音频缺失时记录失败且不虚算 ASR 用量 |
| 文本分析任务 | `deepseek / deepseek-v4-flash` | 真实调用 |

离线决策层不替代运行时路由器：

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `model_router.py` | 当前 `task_type` | 当前任务使用的供应商和模型 | 批处理运行时选择模型 |
| `model_strategy_advisor.py` | 已有 `batch_report.json` 和 `model_calls.jsonl` | 策略报告 | 事后解释成本、延迟、真实 / mock 边界 |
| `strategy_simulator.py` | 已有批次和策略配置 | 路由策略模拟报告 | 离线比较不同业务目标的取舍 |
| `routing_preflight.py` | 路由规则、价格表、策略约束、可选输入目录、可选预估用量和历史 `model_calls.jsonl` | 预检查报告 | 运行前判断输入规模、历史P95延迟、任务级P95目标、价格目录新鲜度、路由完整性、预算/P95延迟/真实覆盖率约束；当延迟异常或存在mock延迟证据时，进一步解释慢因来自真实 API、本地 OCR 还是 mock 占位，并给出受控小样本试跑建议 |

字段说明：

| 字段 | 含义与作用 |
|---|---|
| `task_type` | 模型调用任务类型，用于区分 OCR、视觉理解、语音识别和文本分析 |
| `provider` | 模型供应商，用于按厂商统计成本和延迟 |
| `model_name` | 请求模型名称，用于记录系统向供应商请求调用哪个模型 |
| `response_model_name` | 服务端响应模型名称，用于在供应商返回该字段时核对实际响应来自哪个模型或模型别名 |
| `is_mock` | 是否为 mock 调用，用于区分真实证据和占位流程 |
| `policy_name` | 离线路由策略名称，用于区分成本优先、延迟优先、质量优先和平衡策略 |
| `constraint_status` | 约束满足状态，用于判断当前批次是否符合策略目标 |
| `preflight_status` | 运行前预检查状态，用于区分当前配置是可继续、存在风险还是不建议直接运行 |
| `workload_profile` | 运行前规模画像，用于统计输入文件数量、媒体类型分布和预估任务用量 |
| `price_catalog_profile` | 价格目录画像，用于检查本次路由涉及模型的价格来源、更新时间和可信度 |
| `max_price_age_days` | 价格过期阈值，用于判断价格目录是否需要在成本敏感动作前刷新 |
| `price_freshness_status` | 价格新鲜度状态，用于区分价格是新鲜、过期、缺少更新时间、日期异常还是未来日期 |
| `price_confidence_status` | 价格可信度状态，用于区分官方价格、可解释的mock或本地零成本假设、未知价格和未验证价格 |
| `expected_units_by_task` | 每种任务的预估计量单位，用于把单位价格转换为运行前预算估算 |
| `latency_profile` | 历史延迟画像，用于从已有模型调用记录中汇总任务级平均延迟、P95延迟和最大延迟 |
| `latency_bottleneck_analysis` | 延迟阻塞归因，用于把慢因拆成真实外部 API、本地运行和 mock 占位三类 |
| `real_api_slow_tasks` | 真实外部 API 慢任务列表，用于判断哪些真实网络调用超过当前 P95 目标 |
| `local_runtime_slow_tasks` | 本地运行慢任务列表，用于判断哪些慢因来自本机 PaddleOCR 等本地推理链路 |
| `mock_latency_unusable_tasks` | mock 延迟不可用任务列表，用于提醒这些延迟不能作为真实供应商性能证据 |
| `task_latency_targets_ms` | 任务级P95延迟目标，用于分别约束 OCR、视觉理解、文本分析等任务，避免用单一全局阈值错误评价不同任务 |
| `task_latency_target_checks` | 任务级延迟检查明细，用于记录每类当前预期任务的观察P95、目标P95、证据口径、通过状态和原因 |
| `task_latency_target_summary` | 任务级延迟检查汇总，用于判断当前批次是否存在任务级延迟硬阻塞或mock证据风险 |
| `evidence_level` | 证据口径，用于区分非mock历史延迟、本地运行历史延迟和mock占位延迟，避免把mock延迟误读成真实供应商性能 |
| `historical_p95_latency_by_task_ms` | 按任务类型整理的历史P95延迟，用于把已有运行经验带入运行前延迟预检查 |
| `controlled_trial_plan` | 受控小样本试跑建议，用于在预算通过但延迟失败时，说明下一轮应缩到哪些文件、哪些后端可以单独试、哪些真实调用需要授权 |

## 5. 成本与延迟追踪逻辑

系统把成本和延迟拆成三个层级：

| 层级 | 记录位置 | 用途 |
|---|---|---|
| 单次模型调用 | `model_calls.jsonl` | 追踪供应商、模型、用量、成本、延迟和状态 |
| 单个文件结果 | `results.jsonl` / `results_readable.md` | 汇总该文件所有调用成本和处理耗时 |
| 整个批次 | `batch_report.json` | 汇总总成本、平均成本、P95 延迟、成功率和质量风险 |

成本计算口径是“用量 × 本地价格表”的工程估算，不是供应商最终账单。真实 API 的 `input_units` 和 `output_units` 优先来自供应商响应 usage；mock 调用使用项目内约定的占位用量和价格；本地 PaddleOCR 当前外部 API 成本记为0，但不包含本机 CPU、内存、电力和运维成本。新批次会在 `batch_metadata.json` 的 `cost_estimation` 中记录价格来源、计算方法、是否包含 mock 估算以及是否完成账单对账。未完成供应商账单对账前，整体估算误差状态应视为未知，不能用本地四舍五入误差替代真实扣费误差。

字段说明：

| 字段 | 含义与作用 |
|---|---|
| `call_id` | 单次模型调用唯一标识，用于排查和追踪调用链路 |
| `input_units` | 输入用量及单位，用于成本估算 |
| `output_units` | 输出用量及单位，用于成本估算 |
| `cost_cny` | 单次模型调用成本，单位人民币 |
| `latency_ms` | 单次模型调用耗时，单位毫秒 |
| `processing_cost_cny` | 文件级总成本，由该文件关联的调用成本累加得到 |
| `processing_time_ms` | 文件级总处理耗时，用于判断用户等待时间 |
| `selected_backends` | 本次命令或配置选择的后端组合，用来复盘 OCR、视觉理解和文本分析分别选择了什么后端 |
| `backend_runtime_summary` | 根据实际模型调用明细汇总的真实 API、本地模型和 mock 组合，用来避免把混合批次误读成全真实批次 |
| `cost_estimation` | 成本估算口径说明，用来记录价格表来源、是否包含 mock 成本、估算误差状态和是否已与真实账单对账 |
| `cost_estimation_method` | 成本估算方法，用于说明本次成本来自系统价格目录而不是供应商实时账单 |
| `price_source` | 价格来源，用于说明模型单价来自本地手工配置、mock假设还是本地运行时假设 |
| `price_updated_at` | 价格更新时间，用于判断当前价格目录是否可能过期 |
| `price_confidence` | 价格可信度，用于区分未验证手工价格、mock价格和本地外部API零成本假设 |
| `price_fetch` | 价格抓取配置，用于说明某个模型是否允许从官方公开页刷新价格，以及使用哪个解析器 |
| `source_url` | 官方价格页地址，用于追溯自动抓取的来源 |
| `parser` | 价格解析器名称，用于说明系统按哪套规则解析官方页面 |
| `preflight_status` | 价格刷新预检查状态，用于判断候选价格是否允许写回本地价格目录 |
| `preflight_checks` | 价格刷新预检查明细，用于记录必要计价单位、正数价格和大幅变化检查是否通过 |
| `allow_large_change` | 大幅价格变化写回授权开关，用于要求人工确认超过50%的价格变化后才能写回配置 |
| `recorded_cost_cny` | 历史模型调用记录中原本保存的成本估算值 |
| `current_estimated_cost_cny` | 按当前价格目录和历史调用用量重新计算出的成本估算值 |
| `reprice_status` | 成本重算状态，用于区分未变化、已变化和无法重算 |
| `cost_basis` | 策略报告使用的成本口径，用于区分历史记录成本和当前价格目录重算成本 |
| `billed_cost_cny` | 供应商后台显示或账单导出的实际扣费，用于和系统估算成本对比 |
| `cost_delta_cny` | 供应商实际扣费减去系统估算成本后的差额，用于判断偏差方向和金额 |
| `cost_delta_rate` | 成本偏差相对估算成本的比例，用于观察误差规模；估算值为0时不硬算 |
| `billing_granularity` | 账单粒度，例如单次调用、小时级、日级或模型级，用于判断对账可信度 |
| `bill_source` | 真实扣费来源，例如供应商控制台人工查看或供应商导出账单，用于说明账单证据从哪里来 |
| `matching_method` | 系统调用记录与供应商账单的匹配方式，例如按供应商、模型和时间窗口匹配 |
| `cost_confidence` | 成本可信度状态，用于区分未验证、单次调用级对账和时间段级对账 |
| `matched_call_ids` | 实际参与账单核对的模型调用编号列表，用于从对账结果反查调用明细 |
| `unmatched_billing_records` | 未匹配到本批次模型调用的账单记录，用于提示账单时间窗口、供应商或模型名称可能填错 |

## 6. 输出文件生成逻辑

每次运行都会生成一个批次目录：

```text
output/{batch_id}/
```

输出文件说明：

| 文件 | 作用 |
|---|---|
| `batch_metadata.json` | 保存批次元数据，例如批次编号、创建时间、预算、输出格式、后端组合和成本估算口径 |
| `results.jsonl` | 保存文件级最终结构化结果；新批次每行一条完整文件记录 |
| `results_readable.md` | 保存人工可读的结果说明 |
| `model_calls.jsonl` | 保存模型调用明细；每行一条完整调用记录，是成本、延迟和调用链追踪的来源 |
| `errors.jsonl` | 保存错误索引；每行一条完整错误记录，用于集中排查失败或部分成功 |
| `batch_report.json` | 保存批次级统计报告 |
| `model_strategy_report.md` / `model_strategy_report.json` | 保存模型组合策略分析 |
| `routing_policy_simulation.md` / `routing_policy_simulation.json` | 保存离线路由策略模拟结果 |
| `routing_preflight_report.md` / `routing_preflight_report.json` | 保存批处理前的输入规模画像、历史延迟画像、任务级延迟目标检查、价格目录画像、延迟阻塞归因、路由完整性、预算、P95延迟、真实覆盖率和mock边界检查 |
| `cost_reconciliation_template.csv` | 保存手工账单录入模板，用于把不同供应商账单转换到统一对账格式 |
| `cost_reconciliation.md` / `cost_reconciliation.json` | 保存成本对账报告，用于比较估算成本与实际扣费，并记录账单粒度和可信度 |
| `cost_reprice_report.md` / `cost_reprice_report.json` | 保存成本重算报告，用于比较历史记录成本和当前价格目录下的重算成本 |
| `preprocess_artifacts/` | 保存视频 V1 预处理产物，例如等距抽出的多张关键帧，用于追踪视频画面证据 |
| `text_topic_eval_template.csv` | 保存文本主分类人工评估模板 |
| `text_topic_eval_report.md` / `text_topic_eval_report.json` | 保存文本主分类评估报告 |
| `image_ocr_eval_summary.md` / `image_ocr_eval_summary.json` | 保存图片OCR分段质量汇总，包括完整段落召回率、字符错误率和OCR延迟 |
| `image_ocr_error_analysis_img_9.md` / `image_ocr_error_analysis_img_9.json` | 保存 `img_9.jpg` 的OCR错误归因和单图闸门判断 |
| `image_ocr_gate_report_keyframes.md` / `image_ocr_gate_report_keyframes.json` | 保存三张关键帧图片的批次级OCR闸门判断 |
| `image_ocr_preprocess_experiment_img_9.md` / `image_ocr_preprocess_experiment_img_9.json` | 保存 `img_9.jpg` 的预处理变体实验结果 |
| `image_ocr_latency_profile_img_9.md` / `image_ocr_latency_profile_img_9.json` | 保存 `img_9.jpg` 的OCR延迟拆分结果，区分引擎创建、解码、模型推理和解析耗时 |
| `ocr_backend_advice.md` / `ocr_backend_advice.json` | 保存OCR后端取舍判断，用于决定是否继续本地OCR路线，或在用户授权后评估服务化OCR |
| `rapidocr_candidate_eval.md` / `rapidocr_candidate_eval.json` | 保存RapidOCR候选评估结果；当前已完成三张关键帧本地实测，闸门结论为未通过 |
| `failure_demo_interpretation.md` | 保存失败 / 部分成功演示解读 |

写入层把机器输出和人工输出分开：三个 `.jsonl` 文件遵循标准 JSONL，便于流式读取和导入；`results_readable.md` 负责分段、换行和解释。读取层保留对历史缩进式连续 JSON 对象的兼容，但不会批量改写已有输出。

字段关系：

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
| `processing_status` | 文件级处理状态，用来判断成功、失败、部分成功或跳过；低质量OCR会使文件进入部分成功 |
| `error_message` | 技术错误信息，用于开发者排查失败原因 |
| `warning_messages` | 面向使用者的风险提示，用于解释结果可信度受到什么影响 |

## 7. 文本主分类评估逻辑

`text_topic_evaluator.py` 不调用 DeepSeek，也不重新生成模型预测。它读取已有 `results.jsonl` 和人工标准答案，生成评估模板或评估报告。

DeepSeek提示词已经实现九类主分类的完整判断顺序：广告营销、新闻资讯、财经商业、科技数码、体育健康、娱乐休闲、生活日常、知识科普和其他。提示词使用通用业务边界，不包含现有评估错例的具体答案。回归批次 `batch_text_eval_20260722_135443` 中，17条有效预测全部正确，原4条错例均已修复；另有1条响应解析失败，没有主分类结果。

DeepSeek 与 Qwen-VL 的响应处理都会区分API外层JSON错误、缺少模型内容、空内容、模型内容非JSON和结果字段不合规。完整包裹JSON的Markdown代码块可以安全移除；任意解释文字不会被猜测成正式结果。只有显式启用一次重试时，可重试错误才会再次请求。第一次失败和第二次成功分别生成不同的 `call_id`，并分别记录 `status`、`cost_cny` 与 `latency_ms`，因此重试不会被隐藏成一次调用。DeepSeek 定向真实验证批次 `batch_text_retry_20260722_192832` 第一次请求即成功，没有实际触发重试；Qwen-VL 视频关键帧真实批次曾出现单帧网络断开，本轮代码已支持该类关键帧级重试，但尚需用户手动复测。

业务用途采用两层约束：提示词要求 `business_use` 只能描述证据直接支持的业务动作；客户端再检查品牌推广、广告投放、带货和转化等高风险表述。如果输入没有明确品牌合作、购买、下单或促销信号，系统会把用途降级为内容归档、检索和人工复核，并在 `quality_flags` 中记录 `business_use_grounded_fallback`。该标记说明防护被触发，便于后续审计；文件仍可保持 `processing_status=success`，因为结构化结果已经成功生成。真实定向批次 `batch_business_use_guard_20260722_222907` 中，模型直接返回了保守用途，因此质量风险标签为空，客户端强制降级分支没有被真实请求触发。

相关字段说明：

| 字段 | 含义与作用 |
|---|---|
| `business_use` | 证据支持的业务用途说明，用于告诉使用者结构化结果可以直接支持什么动作 |
| `quality_flags` | 机器可读的质量风险标签，用于记录用途降级等不影响流程成功、但需要追溯的质量事件 |
| `processing_status` | 文件最终处理状态；用途被安全降级时仍可为成功，因为核心结果已生成 |

如果需要重新处理评估样本，应显式指定评估输入目录：

```powershell
python .\src\main.py --input-dir evaluation\text_topic_small_set --text-analysis-backend mock --batch-id batch_eval_mock
```

这条命令使用 mock 后端，不触发真实 API。重新生成真实预测必须同时显式使用 `--text-analysis-backend deepseek` 和 `--allow-live-api`，并会产生费用。

```text
已有 results.jsonl
  ↓
提取文本文件的 topic 预测
  ↓
生成 text_topic_eval_template.csv
  ↓
人工填写或按 file_name 合并 text_topic_gold.csv
  ↓
计算端到端 Accuracy、有效预测 Accuracy、预测覆盖率、Macro-F1 和分类级 Precision / Recall / F1
  ↓
输出 text_topic_eval_report.json / text_topic_eval_report.md
```

评估器会把“分类判断错误”和“没有有效预测”分开记录。没有预测仍会降低端到端 Accuracy，并作为真实分类的漏报进入分类级指标；但不会把“当前数据未提供”误当作第十个业务分类。有效预测 Accuracy 只观察九类范围内的有效输出，预测覆盖率则反映调用和JSON结构解析的稳定性。

图片OCR评估采用另一套分段流程：

```text
读取 results.jsonl 中指定图片的 ocr_text
  ↓
读取 image_ocr_gold.csv 中人工确认的业务文字块
  ↓
统一全角/半角并移除空白，按非重叠字符范围匹配必选文字块
  ↓
计算文字块精确召回率和分段字符错误率
  ↓
输出 image_ocr_eval_report.json
```

这里不比较整页字符串顺序，因为复杂页面中的多栏文字可能被OCR按不同顺序返回；也不统计点赞数、时长、按钮和被截断的话题标签。同一OCR行可以匹配多个互不重叠的文字块，但同一段字符不能被重复使用。`img_1.png` 的20个必选文字块中19个完整命中，分段字符错误率为1.27%；`img_2.png` 的8个必选文字块全部完整命中，分段字符错误率为0%。两张图仍不足以代表生产分布。

字段说明：

| 字段 | 含义与作用 |
|---|---|
| `predicted_topic` | 模型预测的文本主分类，用来和人工标准答案比较 |
| `gold_topic` | 人工标注的正确主分类，是计算准确率的基准 |
| `reviewer_note` | 人工评审备注，用来解释某条样本为什么属于某个主分类 |
| `evaluated_count` | 已纳入评估的文本样本数，用来判断 Accuracy 的样本基础 |
| `correct_count` | 模型预测与人工标准答案一致的样本数，用来计算 Accuracy |
| `accuracy` | 文本主分类准确率，计算方式为 `correct_count / evaluated_count` |
| `valid_prediction_accuracy` | 有效九分类预测中的准确率，用来单独观察分类判断能力 |
| `prediction_coverage` | 有效九分类预测占已评估样本的比例，用来观察调用和结构解析稳定性 |
| `missing_prediction_count` | 没有产出主分类结果的样本数，用来统计调用或解析失败 |
| `macro_f1` | 各参与评估分类 F1 的简单平均，使不同样本量的分类具有相同权重 |
| `precision` | 预测为某分类的样本中真正属于该分类的比例，用于观察误报 |
| `recall` | 人工标注为某分类的样本中被正确识别的比例，用于观察漏报 |
| `f1` | 单个分类 Precision 与 Recall 的调和平均，用于综合衡量误报和漏报 |
| `support` | 人工标准答案中属于某分类的样本数，用于判断分类证据量 |
| `segment_id` | 图片中文字块的唯一编号，用于逐段定位识别正确或错误的位置 |
| `gold_text` | 人工确认的正确业务文字，是OCR质量评估基准 |
| `exact_segment_recall` | 完整识别的必选业务文字块占比；重复文字必须匹配不同OCR行 |
| `character_error_rate` | 分段编辑距离总和除以人工正确字符总数，不统计被排除的界面噪声 |
| `matched_ocr_text` | 与人工文字块配对的完整OCR行，用于追溯指标来源 |

## 8. 当前架构限制

- 真实模型证据覆盖 DeepSeek 文本分析和 PaddleOCR 图片文字提取；PaddleOCR已完成五张正式图片、共151段人工业务文字评估，但样本量仍不足以外推生产质量。
- 图片 OCR 默认仍为 mock，显式选择后才使用本地 PaddleOCR；图片视觉理解默认仍为 mock，已支持显式选择 Qwen-VL API，并已完成 `img_1.png` 单图真实验证；视频 V1 抽出的多张关键帧已经能接入 PaddleOCR 或 Qwen-VL，且 Qwen-VL 关键帧级重试已完成离线测试；语音识别仍未真实实现。
- 当前 Qwen-VL 真实证据覆盖单张图片和一次视频5关键帧试跑；视频试跑中4帧成功、1帧网络失败，说明链路可运行但可靠性还需要复测。当 `ocr_text` 被低质量闸门判为不可用时，下游文本分析会忽略该OCR证据，但仍会保留原始OCR文字供复核。因此不能把文件级结构化结果解释为完整图片或视频理解质量。
- Paddle 底层推理器对中文路径仍不稳定；本轮直接使用 H 盘中文路径时模型创建失败，改用临时英文盘符映射和 `PADDLE_PDX_CACHE_HOME` 后成功运行。代码尚未自动处理该环境问题。
- 当前本地 CPU 实测中，`img_1.png` 的 OCR 调用耗时15733ms；`img_2.png` 独立冷启动批次耗时51096ms；三张关键帧图片的 OCR 平均延迟为18006ms、P95延迟为28261ms。当前 OCR 延迟仍高于既定图片2秒目标，且本地资源成本也未计量。
- 已生成关键帧 OCR 批次级闸门报告：三张关键帧图片整体完整段落召回率78.05%、字符错误率11.01%、P95延迟28261ms，结论为未通过；其中 `img_9.jpg` 是主要质量阻塞样本，三张图都存在延迟阻塞。
- 已完成 `img_9.jpg` OCR 预处理最小实验：整图放大2倍和左右分区放大2倍均只带来轻微召回提升，字符错误率没有下降，延迟显著高于目标，因此该预处理方向不能直接进入主流程。
- 已完成 `img_9.jpg` OCR 延迟拆分：引擎创建8834ms，首次模型推理60373ms，热启动第二次模型推理56042ms；图片解码15ms/10ms、结果解析0ms，说明当前瓶颈主要在本地CPU模型推理。
- 已完成RapidOCR候选实测：三张关键帧整体完整段落召回率82.93%、字符错误率10.64%、P95延迟4294ms，虽然明显快于当前PaddleOCR CPU批次，但仍未达到当前质量和2秒P95延迟闸门。
- 已更新OCR后端取舍判断：RapidOCR已标记为 `evaluated_not_passed`，不接入主流程；如果继续追求生产可用OCR，下一步只能在用户授权后小样本评估服务化OCR，否则保留PaddleOCR作为当前本地基线。
- 视频 V1 预处理已能用 OpenCV 读取元信息并等距抽取最多 5 张关键帧；这些关键帧已能分别进入 PaddleOCR 或 Qwen-VL 后端；本地 ffmpeg 可用时能抽取 wav 音频文件，但真实语音识别仍未接入，因此不能证明完整视频理解质量。
- 运行时模型路由是固定规则，尚未根据预算、质量要求、延迟目标动态选择模型；路由策略预检查可以基于输入目录生成运行前规模画像，并按本批次实际媒体类型推导会触发的任务，例如无视频批次不会把语音识别纳入当前预算检查，未触发长文本切分时也不会默认纳入跨片段汇总任务。它也可以基于历史 `model_calls.jsonl` 生成任务级P95延迟画像，并优先按 `task_latency_targets_ms` 做任务级延迟闸门检查；当前受控 preflight 报告结论为 `warning` 而非 `fail`，原因是预算、真实覆盖率、OCR任务级延迟和文本分析任务级延迟通过，但视觉理解只有 mock 占位延迟证据，不能解释为真实视觉模型P95达标。该模块不会自动调整模型组合或自动执行试跑命令。
- 多供应商成本对账已经具备通用模板、账单金额校验和报告生成能力，但当前只支持手工录入账单金额，不会自动拉取阿里、DeepSeek、豆包等平台账单；未填入 `billed_cost_cny` 前，`cost_confidence` 仍为 `unverified`。如果 `billed_cost_cny` 是非数字、负数、NaN 或 Infinity，或者同一供应商/模型/响应模型存在重叠时间窗口的重复账单记录，系统会拒绝生成对账报告。
- 决策层报告和路由策略模拟基于已有批次数据生成，不能替代真实多供应商 live test。
- 历史14条受控结果存在标签泄漏，只能证明工程链路；清理后的18条样本修改前基线为77.78% Accuracy和73.70% Macro-F1。
- 九类规则回归批次端到端Accuracy为94.44%、有效预测Accuracy为100.00%、预测覆盖率为94.44%、Macro-F1为96.30%；原4条错例均已修复，但历史批次有1条响应解析失败。
- 结构化响应校验已通过原失败样本的真实定向验证；显式重试分支已通过离线故障测试，但本次真实调用未自然触发重试。
- 高风险商业用途证据约束已完成原样本的真实定向验证，本次没有再生成无证据商业建议；模型主动返回保守用途，所以强制降级分支仍只有离线测试证据。
- 当前18条样本每类只有2条，且参与过规则诊断，不能代表线上内容分布或证明泛化能力。
- 当前没有数据库层和前端页面，输出主要写入本地 JSON、JSONL 和 Markdown 文件。

## 9. 后续可扩展方向

| 优先级 | 方向 | 价值 |
|---|---|---|
| 已完成 | 明确图片 OCR 评估口径 | 只统计账号名称、简介、作品标题和作品说明等业务内容文字 |
| 已完成 | 扩充图片 OCR 评估小集 | 当前有5张正式样本、151段人工业务文字，其中3张来自真实视频关键帧信息图 |
| 已完成 | 建立图片 OCR 评估器 | 分段计算精确召回率和字符错误率，避免多栏阅读顺序污染指标 |
| 已完成 | 受控执行 PaddleOCR 真实图片 | 五张正式图片均已处理；其中关键帧三图验证了结果解析、延迟和调用记录链路 |
| 已完成 | 增加指定文件筛选 | 使用 `--include-files` 只处理目标文件，避免误跑整个输入目录 |
| 已完成 | 分析 OCR 弱样本与延迟瓶颈 | `img_9.jpg` 错误集中在小字号结构图模块、Buffer 和 TLB 指标，且28261ms延迟不达标 |
| 已完成 | 生成关键帧 OCR 批次级闸门报告 | 把单图错误归因升级为批次级是否进入下一功能的判断，结论为未通过 |
| 已完成 | 执行 `img_9.jpg` 预处理最小实验 | 整图放大和左右分区放大无法通过质量与延迟闸门 |
| 已完成 | 拆分 OCR 延迟来源 | 已确认 `img_9.jpg` 慢在本地CPU模型推理，不是图片解码或结果解析 |
| 已完成 | 生成 OCR 后端取舍判断 | 基于PaddleOCR证据和RapidOCR实测结果，明确RapidOCR不接入主流程，服务化OCR需要单独授权 |
| 已完成 | 准备 RapidOCR 候选评估器 | 依赖未安装时安全输出 `dependency_missing`，依赖安装后可复用同一批图片和人工基准 |
| 已完成 | 真实评估 RapidOCR 候选 | 三张关键帧同批样本已跑通，本地0元外部API成本，但质量和延迟闸门未通过 |
| 已完成 | 增加路由策略预检查 | 在批处理前检查当前路由完整性、预算、P95延迟、真实模型覆盖率和mock边界 |
| 已完成 | 补运行前规模画像 | 已基于输入目录统计预计文件数、图片数、视频数、音频秒数和token用量；预算可运行前估算 |
| 已完成 | 补历史 P95 延迟输入 | 已从已有 `model_calls.jsonl` 汇总任务级历史P95；当前受控 preflight 报告改为按任务级目标检查，避免用单一3500ms全局目标误判OCR和文本API |
| 已完成 | 补延迟阻塞归因 | 已把慢因拆成真实外部API、本地PaddleOCR运行和mock占位三类；当前结论是本地OCR和真实DeepSeek文本分析需要按各自任务目标解释，mock任务不能用于供应商性能判断 |
| 已完成 | 补受控小样本试跑建议 | 预算通过但延迟失败时，报告会建议小批量、拆后端、暂不纳入视频，避免直接扩大运行 |
| 已完成 | 补低质量OCR结果闸门 | PaddleOCR调用成功但文字疑似乱码时，文件级结果会进入 `partial_success` 并写入质量风险，不再被简单当作完全成功 |
| 已完成 | 补视频 V1 多关键帧预处理闭环 | 本地读取视频元信息、等距抽取最多 5 张关键帧、记录预处理产物，并把音频缺失明确写成部分成功原因 |
| 已完成 | 补本地音频提取最小闭环 | 本机存在 ffmpeg 时写出 wav 音频文件；缺少依赖、超时、失败或空输出时记录明确状态；真实 ASR 仍未接入 |
| P0 | 保留重试计量回归 | 后续改动继续保证每次尝试独立记录成本、延迟和状态 |
| P1 | 把故障注入接入受保护演示命令 | 让失败 / 部分成功样例更容易复现，同时避免默认流程误触发 |
| P1 | 判断是否授权服务化OCR小样本评估 | RapidOCR已实测未过闸门；如果继续追求生产可用OCR，需要先确认外部API、成本和数据风险 |
| P2 | 后续整理展示材料 | 等核心能力更扎实后再考虑是否补充图示、可视说明和对外说明 |

## 10. 相关文档

| 文档 | 作用 |
|---|---|
| `docs/demo_walkthrough.md` | 解释代表性输出的运行方式、读法、成本延迟和真实 / mock 边界 |
| `docs/tests.md` | 说明已有离线测试、测试命令、覆盖范围、风险缺口和后续测试计划 |
