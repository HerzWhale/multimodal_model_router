# Multimodal Model Router

面向内容平台 AI 团队技术负责人的多模态批处理与模型路由 MVP。

系统目标不是做一个单点内容分类器，而是把文本、图片、视频文件统一接入，拆成可追踪的处理任务，并记录模型调用、成本、延迟、证据来源和错误状态。它要解决的是 AI 团队在批量处理内容素材时常见的工程问题：输入不统一、输出不统一、模型调用过程不可追踪、成本和延迟难以复盘。

当前版本必须诚实说明边界：DeepSeek 文本分析已经完成真实调用验证；图片 OCR 已接入本地 PaddleOCR，并用五张正式图片完成本地 CPU 推理与分段评估，其中 `img_7.jpg`、`img_8.jpg`、`img_9.jpg` 来自真实视频关键帧信息图。图片视觉理解已新增受保护的 Qwen-VL API 接入入口，并已完成 `img_1.png` 单图真实 API 验证；但默认仍为 mock，且该单图批次中的 OCR 和文本分析仍是 mock。视频预处理已从 V0 第一帧升级到 V1 多关键帧，可以读取视频元信息并等距抽取最多 5 张关键帧；这些关键帧可以按配置进入 PaddleOCR 或 Qwen-VL，Qwen-VL 已支持可重试错误最多重试1次并逐次记录调用，但默认仍为 mock。视频音频提取已新增本地 ffmpeg 最小闭环：本机存在 ffmpeg 且提取成功时会生成 wav 音频文件，并进入现有 mock 语音识别分支；真实 ASR 尚未接入，因此当前验证仍不能证明完整多模态理解质量。

## 1. 项目定位

这个项目服务的对象是内容平台或互联网中厂的 AI 团队技术负责人。

典型场景是：团队需要一次处理一批文本、图片和视频素材，希望系统能统一读取文件、判断文件类型、选择对应处理流程、调用模型生成结构化结果，并把每一步调用的模型、成本、延迟和状态记录下来。

核心价值是：

- 统一输入：文本、图片、视频进入同一批处理入口。
- 统一输出：最终结果写入稳定的 JSONL / Markdown 文件。
- 任务拆解：把一个文件拆成 OCR、视觉理解、语音识别、文本分析等子任务。
- 成本核算：记录单次模型调用、单文件、整批任务的成本。
- 延迟记录：记录单次调用延迟、文件级耗时和批次 P95 延迟。
- 证据链追踪：说明最终分析基于哪些证据，缺失哪些证据。
- 模型路由：当前运行时按任务类型固定路由，已支持基于历史批次的离线路由策略模拟。

## 2. 当前已实现能力

| 能力 | 当前状态 | 工程作用 |
|---|---|---|
| 文本、图片、视频文件识别 | 已实现 | 决定文件进入哪条处理流水线 |
| 文本处理流程 | 已实现 | 读取原文并交给文本分析模型生成结构化结果 |
| 图片处理流程 | 已实现工程链路、本地 OCR 接口和受保护视觉理解 API 入口 | OCR 默认 mock，可显式选择 PaddleOCR；视觉理解默认 mock，可显式选择 Qwen-VL API；Qwen-VL 已完成 `img_1.png` 单图真实验证，但尚未形成多图质量评估 |
| 视频处理流程 | 已实现工程链路、视频V1多关键帧预处理、关键帧上游入口和本地音频提取最小闭环 | 可读取视频元信息，默认按 `start_early_then_spaced` 抽取最多3张关键帧并记录预处理产物；关键帧可显式进入 PaddleOCR 或 Qwen-VL；Qwen-VL 支持关键帧级受控重试和失败补偿；本机有 ffmpeg 时可抽取 wav 音频文件；`.flv` 已纳入视频输入识别 |
| DeepSeek 文本分析 | 已真实接入 | 响应校验和业务用途证据约束均完成单样本真实验证；强制降级分支由离线测试覆盖 |
| 模型调用记录 | 已实现 | 记录供应商、模型、任务类型、用量、成本、延迟和状态 |
| 批次报告 | 已实现 | 汇总文件数、成功率、成本、延迟和质量风险 |
| 模型组合策略报告 | 已实现离线分析 | 基于已有批次解释成本、延迟、真实 / mock 边界和模型组合建议 |
| 路由策略模拟 | 已实现离线分析 | 模拟成本优先、延迟优先、质量优先和平衡策略 |
| 路由策略预检查 | 已实现离线检查 | 批处理前读取路由、价格、价格目录新鲜度、输入规模和历史调用记录，检查路由完整性、预算、任务级P95延迟、真实模型覆盖率和mock边界；当存在硬阻塞或风险提示时，会给出受控小样本试跑建议；不自动改路由，不触发模型调用 |
| 多供应商成本对账 | 已实现离线对账 | 基于 `model_calls.jsonl` 生成手工账单模板，并把系统估算成本与供应商实际扣费做统一格式对比；首版不接平台账单 API |
| 文本主分类评估 | 已完成提示词回归评估 | 18条无答案提示样本的端到端 Accuracy 为94.44%、Macro-F1 为96.30%；17条有效预测全部正确，1条因响应解析失败没有分类结果 |
| 图片 OCR 评估 | 已实现五图分段评估和批次级闸门 | `img_1.png`、`img_2.png`、`img_7.jpg`、`img_8.jpg`、`img_9.jpg` 共151段人工业务文字；本轮三张关键帧图片整体完整段落召回率78.05%、字符错误率11.01%，批次级闸门结论为未通过 |
| OCR 后端取舍判断 | 已实现离线判断报告 | 基于已有PaddleOCR质量闸门、延迟拆分和RapidOCR候选实测，判断是否继续本地路线或在授权后评估服务化OCR；RapidOCR已实测未通过，Tesseract和云OCR尚未接入 |
| 失败 / 部分成功演示 | 已实现离线故障注入 | 验证错误链路、缺失证据和状态判断规则 |

字段说明：

| 字段 | 含义与作用 |
|---|---|
| `topic` | 内容主分类，表示一份内容最主要的业务归属 |
| `secondary_topics` | 副分类，表示内容涉及的交叉领域，最多两个 |
| `tags` | 细粒度关键词，用于搜索、筛选和素材管理 |
| `summary` | 内容摘要，用于快速理解文件主要内容 |
| `business_use` | 业务用途说明，用来解释结构化结果可以支持什么业务动作；商业推广、广告、带货或转化建议必须有明确输入证据 |
| `quality_flags` | 机器可读的质量风险标签；当缺少商业证据的高风险用途被降级时会记录 `business_use_grounded_fallback`，当真实OCR返回明显乱码或过度碎片化文本时会记录 `low_quality_ocr_text` |
| `model_name` | 请求模型名称，用于记录系统向供应商请求调用哪个模型 |
| `response_model_name` | 服务端响应模型名称，用于在供应商返回该字段时核对实际响应来自哪个模型或模型别名 |
| `visual_description` | 视觉理解模型生成的图片画面描述，用于补充 OCR 无法覆盖的场景、物体、布局和画面语义 |
| `cost_cny` | 单次模型调用成本，单位人民币，用于成本核算 |
| `cost_estimation_method` | 成本估算方法，用于说明本次成本来自系统价格目录而不是供应商实时账单 |
| `price_source` | 价格来源，用于说明模型单价来自本地手工配置、mock假设还是本地运行时假设 |
| `price_updated_at` | 价格更新时间，用于判断当前价格目录是否可能过期 |
| `price_confidence` | 价格可信度，用于区分未验证手工价格、mock价格和本地外部API零成本假设 |
| `latency_ms` | 单次模型调用耗时，单位毫秒，用于延迟分析 |
| `latency_bottleneck_analysis` | 路由预检查中的延迟阻塞归因结果，用于区分真实外部 API、本地运行和 mock 占位三类慢因 |
| `task_latency_targets_ms` | 按任务类型配置的 P95 延迟目标，用于让 OCR、文本分析、视觉理解等不同任务使用不同延迟闸门；当前值是受控试跑阈值，不是生产 SLA |
| `task_latency_target_checks` | 任务级延迟目标检查明细，用于记录每个任务观察到的 P95、目标 P95、目标来源、证据口径和通过状态 |
| `evidence_level` | 延迟或结果的证据口径，用于区分非 mock 历史、本地运行历史和 mock 占位证据，避免把 mock 延迟解释成真实供应商性能 |
| `preprocessing_artifacts` | 文件预处理产物摘要，用于记录视频元信息、关键帧抽取状态、音频提取状态和预处理风险 |
| `keyframe_paths` | 视频预处理抽出的关键帧路径列表，用于追踪视频画面证据来自哪些本地产物 |
| `keyframe_metadata` | 每张关键帧的帧号、时间位置和路径，用于说明视频画面证据覆盖了哪些时间点 |
| `audio_extraction_status` | 音频提取状态，用于说明当前是否已经生成真实音频文件；可能值包括 `extracted`、`dependency_missing`、`failed`、`timeout`、`empty_output` 和 `not_attempted_no_artifact_dir` |
| `real_api_slow_tasks` | 真实外部 API 慢任务列表，用于判断哪些真实网络调用超过当前 P95 目标 |
| `local_runtime_slow_tasks` | 本地运行慢任务列表，用于判断哪些耗时来自本机 PaddleOCR 等本地推理链路 |
| `mock_latency_unusable_tasks` | mock 延迟不可用任务列表，用于提醒这些 0ms 或极低延迟不能作为供应商性能证据 |
| `is_mock` | 是否为 mock 调用，用于区分真实模型证据和占位流程证据 |

## 3. 系统流程

```text
输入文件夹
  ↓
文件加载与类型识别
  ↓
按文件类型分流
  ├─ 文本：读取原文 → 文本分析
  ├─ 图片：OCR / 视觉理解 → 文本分析
  └─ 视频：预处理 → OCR / 视觉理解 / 语音识别 → 文本分析
  ↓
生成文件级结果
  ↓
记录模型调用明细
  ↓
汇总批次成本、延迟、成功率和质量风险
  ↓
生成策略分析、评估报告和人工可读结果
```

## 4. 快速开始

基础安装：

```powershell
python -m pip install paddlepaddle==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install -r requirements.txt
```

第一条命令安装 CPU 版 PaddlePaddle；第二条安装 PyYAML 和 PaddleOCR。首次真正运行 PaddleOCR 时，官方模型权重可能需要联网下载，下载体积和耗时不属于 API 调用成本。

默认配置使用 mock 文本分析，因此下面的命令只验证本地流程，不访问外部 API：

```powershell
python .\src\main.py
```

如需真实调用 DeepSeek 文本分析，先在本机设置环境变量：

```powershell
$env:DEEPSEEK_API_KEY = "你的 DeepSeek API Key"
```

然后必须同时显式选择 DeepSeek 后端并授权真实 API 调用：

```powershell
python .\src\main.py --text-analysis-backend deepseek --allow-live-api
```

仅选择 DeepSeek、但没有提供 `--allow-live-api` 时，程序会在读取文件和发送网络请求前停止；即使配置文件被改成 DeepSeek，也不能绕过该安全闸门。

如需使用本地 PaddleOCR 处理图片，完成上述安装后显式选择本地后端：

```powershell
python .\src\main.py --ocr-backend paddleocr
```

`--ocr-backend paddleocr` 可作用于图片文件和视频 V1 抽出的最多 5 张关键帧，不需要 API 密钥或 `--allow-live-api`。这仍不代表完整视频 OCR；视频音频提取依赖本机 ffmpeg，真实语音识别仍未实现。

显式使用 PaddleOCR 时，主流程会对非空 `ocr_text` 做一个保守质量闸门：如果文本高度碎片化或疑似乱码，文件不会被当成完全成功，而会写入 `quality_flags=low_quality_ocr_text` 和对应 `warning_messages`，并把文件级 `processing_status` 调整为 `partial_success`。这类情况不代表 OCR 调用失败，所以 `model_calls.jsonl` 中该 OCR 调用仍会保留为 `success`，但下游文本分析不会把这段 OCR 文字当成可靠证据。

如果只想处理指定图片，可以使用 `--include-files`：

```powershell
python .\src\main.py --input-dir .\input\sample_images --include-files img_7.jpg,img_8.jpg,img_9.jpg --ocr-backend paddleocr --text-analysis-backend mock --batch-id batch_paddleocr_keyframes_20260724_retry
```

`--include-files` 按文件名筛选本次输入，作用是避免为了评估少数样本而把整个输入目录都跑一遍。由于 Paddle 底层推理器对中文路径支持不稳定，本轮真实 OCR 使用临时英文盘符映射后成功运行；如果直接用包含中文的项目绝对路径运行，可能出现本地模型创建失败。

为避免显式运行 PaddleOCR 时意外调用配置文件中的 DeepSeek，命令行只选择 PaddleOCR 时，未显式选择的文本分析后端会强制保持 mock；启用 DeepSeek 或 Qwen-VL 时，未显式选择的其他后端也会强制保持 mock。

如需试跑 Qwen-VL 图片或视频关键帧视觉理解，先在本机设置环境变量：

```powershell
$env:DASHSCOPE_API_KEY = "你的阿里云百炼 / DashScope API Key"
```

然后必须同时显式选择 Qwen-VL 后端并授权真实 API 调用：

```powershell
python .\src\main.py --input-dir .\input --include-files img_1.png --vision-backend qwen_vl --allow-live-api --batch-id batch_qwen_vl_image_trial
```

`--vision-backend qwen_vl` 可作用于图片文件和视频 V1 抽出的最多 5 张关键帧。该入口会把 Qwen-VL 返回的画面描述写入 `visual_description`，并在 `model_calls.jsonl` 中记录供应商、请求模型名、服务端响应模型名、输入输出 token、成本、延迟和状态。未设置 `DASHSCOPE_API_KEY` 或未提供 `--allow-live-api` 时，程序会在发送网络请求前停止。当前已有 `img_1.png` 单图真实 API 验证和一个视频5关键帧真实试跑；视频试跑中曾出现单帧网络断开，因此当前代码已补充关键帧级受控重试，但仍需要用户再次手动复测。

如需只跑视频 V1 多关键帧预处理闭环，可以使用当前样例视频：

```powershell
python .\src\main.py --input-dir .\input\sample_videos --include-files 例子.mp4 --text-analysis-backend mock --batch-id batch_video_keyframes_v1_mock_trial
```

这条命令不会触发任何外部 API。当前代码会在 `preprocess_artifacts/file_0001/` 下最多写出 5 张等距关键帧，并在预处理产物中记录 `keyframe_metadata`。如果本机存在 ffmpeg，还会尝试抽取单声道 16kHz wav 音频文件；如果缺少 ffmpeg，会把 `audio_extraction_status` 记录为 `dependency_missing`。历史受控 V1 mock 批次 `output/batch_video_keyframes_v1_mock_trial/` 生成于音频提取最小闭环之前：该批次写出5张关键帧，产生5次mock OCR、5次mock视觉理解、1次预期语音识别失败和1次mock文本分析。历史受控 V0 结果仍保存在 `output/batch_video_v0_preprocess_20260804/`，它只写出了1张关键帧，不能代表当前 V1 多关键帧能力。

如需让视频关键帧进入本地 PaddleOCR，可以使用：

```powershell
python .\src\main.py --input-dir .\input\sample_videos --include-files 例子.mp4 --ocr-backend paddleocr --text-analysis-backend mock --batch-id batch_video_keyframes_paddleocr_trial
```

这条命令不访问外部 API，但会运行本地 PaddleOCR，可能消耗较多 CPU 时间。它只验证视频关键帧 OCR，不代表完整视频 OCR 或完整视频理解。

如需让视频关键帧进入 Qwen-VL，需要显式授权真实 API：

```powershell
python .\src\main.py --input-dir .\input\sample_videos --include-files 例子.mp4 --vision-backend qwen_vl --allow-live-api --text-analysis-backend mock --batch-id batch_video_keyframes_qwen_vl_trial
```

这条命令会访问 Qwen-VL 并产生可能的 API 费用；默认不会自动执行。它只处理最多 5 张等距关键帧，不能证明整段视频理解能力。

如果要对视频关键帧 Qwen-VL 的可重试网络错误启用一次受控重试，可以额外加：

```powershell
--max-api-retries 1
```

启用后，每张关键帧仍独立处理。某一帧第一次失败、第二次成功时，两次尝试都会写入 `model_calls.jsonl`：第一次为 `failed`，第二次为 `success`，最终该帧的 `visual_description` 会进入下游证据；如果重试后仍失败，系统会保留其他成功关键帧的画面描述，并写入 `quality_flags=video_visual_keyframe_failed` 和对应 `warning_messages`。这属于失败补偿，不代表完整视频理解已经实现。

批处理前可以先运行路由策略预检查：

```powershell
python .\src\routing_preflight.py --input-dir .\input --include-files ai_content_sample.txt,img.png,img_1.png --ocr-backend paddleocr --text-analysis-backend deepseek --budget-limit-cny 50 --min-real-coverage-rate 0.4 --expected-audio-seconds-per-video 60 --historical-model-calls ".\output\batch_20260718_150348\model_calls.jsonl,.\output\batch_paddleocr_keyframes_20260724_retry\model_calls.jsonl,.\output\batch_text_eval_20260722_135443\model_calls.jsonl,.\output\batch_controlled_mock_trial_20260802\model_calls.jsonl,.\output\batch_controlled_paddleocr_trial_20260802\model_calls.jsonl,.\output\batch_controlled_deepseek_text_trial_20260802\model_calls.jsonl,.\output\batch_controlled_deepseek_text_retry_diagnostics_20260802\model_calls.jsonl" --output-dir .\output\routing_preflight_current
```

这条命令只读取本地配置、输入目录和已有历史调用记录，不运行 DeepSeek、不运行 PaddleOCR，也不下载模型权重。它会生成 `routing_preflight_report.json` 和 `routing_preflight_report.md`，用于说明当前路由是否完整、输入规模会触发多少任务单位、哪些任务仍是 mock、预算和 P95 延迟是否能在运行前判断。

当前 `output/routing_preflight_current/` 的报告基于 3 个受控输入样本生成：1 个文本、2 张图片、0 个视频。由于本轮没有视频，`speech_to_text` 不进入本批次预期任务；由于当前主流程没有触发长文本切分后的跨片段汇总，`summary_merge` 也不进入本批次预期任务。当前预期任务为 OCR、视觉理解和文本分析，预估总成本为 0.022589 元，50 元预算约束通过。当前版本已把延迟闸门拆成任务级目标：OCR 受控本地 CPU 试跑目标为 60000ms，视觉理解当前占位目标为 3500ms，文本分析受控外部 API 目标为 8000ms；读取已有历史批次后，OCR 和文本分析满足各自任务级目标，视觉理解只有 mock 占位延迟证据，因此任务级延迟汇总为 `warning`，不是 `pass`。当前预检查总状态仍为 `warning`，且 `blocking_reasons` 为空；它表示可以继续做受控小样本试跑，但不能把结果解释为完整真实多模态平台、真实视觉模型 P95 达标或生产 SLA。

报告会同时生成 `controlled_trial_plan`：这是受控小样本试跑建议，用来说明在“预算没问题、仍有mock边界或延迟风险”的情况下下一轮应如何缩小范围。当前建议是先使用 `--include-files ai_content_sample.txt,img.png,img_1.png` 做最多3个文件的小批量试跑，暂不纳入视频；任何真实外部API试跑仍必须另行带 `--allow-live-api` 授权。这个建议只写入报告，不会自动执行命令。

默认不会自动重试。只有在已明确授权真实API调用的基础上，再显式增加以下参数，才允许对可重试错误最多重试1次：

```powershell
--max-api-retries 1
```

每次重试都会生成独立模型调用记录并分别计入成本和延迟。该参数当前适用于 DeepSeek 文本分析和 Qwen-VL 视觉理解；鉴权失败、权限错误和参数错误不会重试。不要在未确认输入范围和费用前启用该参数。

如果 DeepSeek 文本层因为输出过长出现空响应或命中输出上限，可以优先调高 `config/settings.yaml` 中的 `deepseek_max_tokens`，或只在重分析命令中临时指定：

```powershell
python .\src\reanalyze_batch_text.py --source-batch-dir .\output\历史批次 --batch-id batch_text_reanalysis_retry --include-files 文件名.mp4 --allow-live-api --max-api-retries 1 --deepseek-max-tokens 3000
```

`deepseek_max_tokens` 只控制 DeepSeek 文本分析单次输出 token 上限；`qwen_vl_max_tokens` 只控制 Qwen-VL 视觉描述单次输出 token 上限。二者越大，越可能减少截断，但也可能增加成本和延迟；它们不是质量评分。

默认情况下，主流程读取 `input/`，也就是普通业务输入样例。`evaluation/` 是评估样本目录，不会被默认批处理自动读取。

如果要用评估样本跑一次安全的离线流程，可以显式指定输入目录和 mock 后端：

```powershell
python .\src\main.py --input-dir evaluation\text_topic_small_set --text-analysis-backend mock --batch-id batch_eval_mock
```

如果要重新生成真实评估结果，必须同时使用 `--text-analysis-backend deepseek` 和 `--allow-live-api`。这一步需要 API Key、会访问外部服务并产生费用。

运行参数说明：

| 参数 / 配置 | 含义与作用 |
|---|---|
| `config/runtime_policy.yaml` | 运行策略配置文件，用于集中管理文件类型白名单、后端白名单、视频预处理参数、OCR质量闸门、topic体系和DeepSeek提示词 |
| `config/settings.yaml` | 运行环境配置文件，用于管理输入输出目录、默认后端、API地址、模型名、token上限和预算 |
| `ocr_backend` | 图片或视频关键帧 OCR 后端配置；默认使用 mock，显式设为 `paddleocr` 时才运行本地 PaddleOCR |
| `vision_understanding_backend` | 图片或视频关键帧视觉理解后端配置；默认使用 mock，显式设为 `qwen_vl` 时才准备调用 Qwen-VL API |
| `text_analysis_backend` | 文本分析后端配置，用来决定使用 mock 还是 DeepSeek |
| `deepseek_max_tokens` | DeepSeek 文本分析单次输出 token 上限，用于减少长证据合并时被截断导致的空响应风险 |
| `qwen_vl_max_tokens` | Qwen-VL 视觉理解单次输出 token 上限，用于控制画面描述长度、成本和延迟 |
| `--vision-backend` | 命令行视觉理解后端选择，用于在小样本图片或视频关键帧上显式启用 Qwen-VL |
| `--allow-live-api` | 真实 API 调用授权开关，用来防止 DeepSeek 或 Qwen-VL 外部请求和费用被配置文件或默认命令误触发 |
| `--max-api-retries` | 可重试API错误的最大重试次数；默认0，只能显式设为1 |
| `--include-files` | 指定本次只处理哪些文件名，用于受控评估少量图片，避免误处理整个输入目录 |
| `--historical-model-calls` | 指定一个或多个历史 `model_calls.jsonl` 路径，用于运行前自动提取任务级 P95 延迟 |
| `input_dir` | 本次批处理读取的输入目录，用来隔离普通业务输入和评估样本 |
| `batch_id` | 批次唯一编号，用来定位一次运行生成的结果和模型调用记录 |

根目录只保留项目运行、测试和说明所需文件。历史清理记录已放入 `local_notes/`；该目录不参与程序运行，也不作为测试输入。大体积视频样本和历史输出批次仍保留在 `input/` 与 `output/`，因为当前视频链路复核仍依赖这些证据，后续应单独确认正式验证集后再清理。

运行离线测试：

```powershell
python -m unittest discover -s tests
```

说明：默认离线测试会替换 PaddleOCR 引擎，不下载模型权重、不进行真实图片推理，也不会触发 DeepSeek API。

运行受保护离线回归检查：

```powershell
python .\src\offline_regression_check.py
```

检查已有批次是否满足功能完整性验收：

```powershell
python .\src\offline_regression_check.py --skip-unit-tests --check-batch-dir .\output\batch_video_full_real_3videos_3frames
```

如果要判断某个批次能否作为“全真实链路证据”，加上严格模式：

```powershell
python .\src\offline_regression_check.py --skip-unit-tests --check-batch-dir .\output\batch_video_full_real_3videos_3frames --require-no-mock
```

这两条命令都只读取已有输出，不触发 DeepSeek、Qwen-VL、ASR 或 PaddleOCR。它会检查文件级结果、模型调用、视频预处理、证据链、成本和延迟记录是否闭环；`--require-no-mock` 会额外拒绝包含 mock 调用的批次。

该入口会先运行完整离线测试，再在临时目录中跑一次三文件 mock 批处理和一次 routing preflight 冒烟检查。它不调用 DeepSeek，不调用 Qwen-VL，不运行真实 PaddleOCR，不使用云OCR，也不会向正式 `output/` 目录写入新批次。若只想快速检查主流程和预检查链路，可以加 `--skip-unit-tests`。

生成某个批次的成本对账模板：

```powershell
python .\src\cost_reconciliation.py template .\output\batch_qwen_vl_response_model_check .\output\batch_qwen_vl_response_model_check\cost_reconciliation_template.csv
```

这条命令只读取已有 `model_calls.jsonl`，不会访问任何供应商 API。模板只列真实 API 调用组；mock 和本地模型会在报告中作为排除对账项说明。

用户从供应商后台把实际扣费填入 `billed_cost_cny` 后，再生成对账报告：

```powershell
python .\src\cost_reconciliation.py reconcile .\output\batch_qwen_vl_response_model_check .\output\batch_qwen_vl_response_model_check\cost_reconciliation_template.csv .\output\batch_qwen_vl_response_model_check\cost_reconciliation.json .\output\batch_qwen_vl_response_model_check\cost_reconciliation.md
```

如果 `billed_cost_cny` 为空，报告会保持 `cost_confidence=unverified`，并把 `estimation_error_status` 记录为 `unknown_until_bill_reconciliation`。如果 `billed_cost_cny` 是非数字、负数、NaN 或 Infinity，程序会拒绝生成对账报告；如果同一供应商、同一模型、同一响应模型出现重叠时间窗口的重复账单记录，程序也会拒绝处理，避免某条账单被静默覆盖。

当前 Qwen-VL 单图批次已经完成一次真实后台账单对账。由于本次供应商后台显示实际扣费为 0.00 元，因此该样例的差异原因记录为免费额度抵扣。已填写的模板保存在：

```text
output/batch_qwen_vl_response_model_check/cost_reconciliation_billing_free_quota.csv
```

如果需要复现这次对账，可运行：

```powershell
python .\src\cost_reconciliation.py reconcile .\output\batch_qwen_vl_response_model_check .\output\batch_qwen_vl_response_model_check\cost_reconciliation_billing_free_quota.csv .\output\batch_qwen_vl_response_model_check\cost_reconciliation.json .\output\batch_qwen_vl_response_model_check\cost_reconciliation.md
```

这次结果表示：系统估算成本为0.003237元，供应商实际扣费为0.00元，Qwen-VL 对账项的 `cost_confidence=period_level_reconciled`，汇总层的 `summary.confidence_counts` 记录了1条周期级对账。它主要证明“系统能把本地估算和供应商真实账单放到同一套口径中核对”，不应解读为生产环境大部分成本都会被免费额度抵扣，也不应解读为估算误差已经很小。

如需做一次真实成本校准，建议只在一个干净时间窗口内运行 Qwen-VL 图片视觉理解，其他真实 API 全部关闭：

```powershell
python .\src\main.py --input-dir .\input\sample_images --include-files img_1.png,img_7.jpg,img_8.jpg --ocr-backend mock --vision-backend qwen_vl --text-analysis-backend mock --allow-live-api --batch-id batch_qwen_vl_cost_calibration_20260804
```

这条命令会产生 3 次 `qwen-vl-plus` 真实图片视觉理解调用；OCR 和文本分析仍为 mock。运行前后不要同时运行 DeepSeek、PaddleOCR、ASR 或其他真实供应商 API，避免供应商后台账单时间窗口混入其他扣费。该命令需要用户明确授权后才能执行，项目不会自动触发。

运行完成后，先生成该批次的账单模板：

```powershell
python .\src\cost_reconciliation.py template .\output\batch_qwen_vl_cost_calibration_20260804 .\output\batch_qwen_vl_cost_calibration_20260804\cost_reconciliation_template.csv
```

然后从供应商后台把对应时间窗口内的真实扣费填入 `billed_cost_cny`，再生成对账报告：

```powershell
python .\src\cost_reconciliation.py reconcile .\output\batch_qwen_vl_cost_calibration_20260804 .\output\batch_qwen_vl_cost_calibration_20260804\cost_reconciliation_template.csv .\output\batch_qwen_vl_cost_calibration_20260804\cost_reconciliation.json .\output\batch_qwen_vl_cost_calibration_20260804\cost_reconciliation.md
```

如果 `billed_cost_cny` 保持为空，该批次仍是未验证估算；如果供应商后台显示免费额度抵扣导致实际扣费为0，应填入 `0.00`，并在 `bill_source` 和 `note` 中说明来源，避免把免费额度误读为模型理论价格为0。

如需基于已经保存的真实OCR结果重新计算 `img_1.png` 的质量指标：

```powershell
python .\src\image_ocr_evaluator.py evaluate `
  .\output\batch_paddleocr_smoke_20260723_retry\results.jsonl `
  .\evaluation\image_ocr_gold.csv `
  img_1.png `
  .\output\batch_paddleocr_smoke_20260723_retry\image_ocr_eval_report.json
```

该命令只读取现有结果和人工基准，不重新运行PaddleOCR，也不会访问外部API。

本轮三张关键帧信息图的 OCR 汇总结果保存在：

```text
output/batch_paddleocr_keyframes_20260724_retry/image_ocr_eval_summary.md
output/batch_paddleocr_keyframes_20260724_retry/image_ocr_eval_summary.json
output/batch_paddleocr_keyframes_20260724_retry/image_ocr_gate_report_keyframes.md
output/batch_paddleocr_keyframes_20260724_retry/image_ocr_gate_report_keyframes.json
output/batch_paddleocr_keyframes_20260724_retry/image_ocr_preprocess_experiment_img_9.md
output/batch_paddleocr_keyframes_20260724_retry/image_ocr_preprocess_experiment_img_9.json
output/batch_paddleocr_keyframes_20260724_retry/image_ocr_latency_profile_img_9.md
output/batch_paddleocr_keyframes_20260724_retry/image_ocr_latency_profile_img_9.json
output/batch_paddleocr_keyframes_20260724_retry/ocr_backend_advice.md
output/batch_paddleocr_keyframes_20260724_retry/ocr_backend_advice.json
output/batch_paddleocr_keyframes_20260724_retry/rapidocr_candidate_eval.md
output/batch_paddleocr_keyframes_20260724_retry/rapidocr_candidate_eval.json
```

低质量 OCR 结果闸门的受控批次保存在：

```text
output/batch_controlled_paddleocr_gate_20260729/results_readable.md
output/batch_controlled_paddleocr_gate_20260729/results.jsonl
output/batch_controlled_paddleocr_gate_20260729/model_calls.jsonl
output/batch_controlled_paddleocr_gate_20260729/batch_report.json
```

该批次用于验证：OCR 模型调用成功但文字质量明显不足时，文件级结果应进入 `partial_success`，写入 `quality_flags=low_quality_ocr_text` 和风险提示，同时不把低质量 `ocr_text` 交给下游文本分析模型。

## 5. 代表性输出

当前保留的多模态 Demo 批次：

```text
output/batch_20260718_150348/
```

该批次包含 3 个输入文件：1 个文本、1 个图片、1 个视频。本批次能证明统一输入、统一输出、调用记录、成本延迟追踪和 DeepSeek 文本分析链路跑通；不能证明图片 / 视频真实理解质量。

当前九类规则回归批次：

```text
output/batch_text_eval_20260722_135443/
```

该批次在九类定义和判断顺序补齐后，对同一组18条文本进行了18次 DeepSeek 真实分析。17条获得有效九分类结果且全部正确，1条因模型响应无法解析为JSON而失败。端到端 Accuracy 为94.44%，有效预测 Accuracy 为100.00%，预测覆盖率为94.44%，Macro-F1 为96.30%；总成本为0.027852元，平均模型延迟为4356.78ms，P95模型延迟为10955ms。成本仅占50元预算上限约0.0557%，但 P95 延迟仍高于文本任务原定2000ms上限。

修改前的4条错例在本轮都已正确分类，`other` 类 Recall 从0提升到100%。但是原先正确的第14条体育健康样本没有获得预测结果，因此严格的端到端回归闸门尚未完全通过。详细结果见 `output/batch_text_eval_20260722_135443/text_topic_eval_interpretation.md`。本轮复用了参与规则修正的已知样本，只能作为回归证据，不能证明对新内容的泛化能力。

结构化响应加固后的定向验证批次为 `output/batch_text_retry_20260722_192832/`。该批次只处理原失败的第14条体育健康样本，第一次请求即成功返回合法结构，因此没有触发重试；模型主分类与人工答案一致。实际成本为0.0015元，模型延迟为4386ms，错误数为0。该结果证明加固后的正常真实调用链路可用，但不能证明真实重试分支已经被供应商故障触发。

业务用途证据约束的定向验证批次为 `output/batch_business_use_guard_20260722_222907/`。同一条体育健康样本只触发1次DeepSeek请求，主分类仍与人工答案一致；业务用途改为内容归档、检索和人工复核，没有再次生成无证据的品牌推广建议。实际成本为0.00178元，延迟为5711ms，错误数为0。模型本次主动返回保守用途，因此 `quality_flags` 为空；这能证明修改后的真实正常路径符合预期，不能证明程序强制降级分支曾在真实请求中触发。

当前保留的失败 / 部分成功演示批次：

```text
output/batch_failure_demo_20260721_190052/
```

该批次使用离线故障注入模拟 OCR 失败、语音识别失败和文本分析失败，用来验证 `partial_success` 和 `failed` 的状态判断。主流程另有低质量 OCR 闸门：当真实 PaddleOCR 调用成功但文字明显不可用时，会把文件标为 `partial_success`，并记录质量风险，而不是把它混同为模型调用失败。

字段说明：

| 字段 | 含义与作用 |
|---|---|
| `input_dir` | 本次批处理读取的输入目录；默认是 `input/`，也可以显式指定为评估样本目录 |
| `batch_id` | 一次批处理任务的唯一标识，用于把多个文件和模型调用归到同一批任务 |
| `file_id` | 单个输入文件的唯一标识，用于关联结果、模型调用记录和错误记录 |
| `processing_status` | 文件处理状态，用于区分成功、部分成功、失败或跳过 |
| `partial_success` | 部分成功状态，表示最终结果已生成但部分上游证据缺失，或关键证据虽然存在但质量不足以支撑完全可信结果 |
| `quality_flags` | 机器可读质量风险标签，用于批量筛选用途降级、低质量OCR等问题 |
| `warning_messages` | 面向使用者的风险提示，用于解释为什么结果不能被当成完全可靠 |
| `failed` | 失败状态，表示关键步骤失败，无法产出有效最终结果 |
| `accuracy` | 文本主分类准确率，计算方式为正确样本数除以已评估样本数 |
| `valid_prediction_accuracy` | 仅在有效九分类预测中的准确率，用于把分类判断能力和调用可用性分开观察 |
| `prediction_coverage` | 有效九分类预测数占已评估样本数的比例，用于衡量调用和结构解析稳定性 |
| `macro_f1` | 各参与评估分类 F1 的简单平均，用来避免总体正确率掩盖小类别问题 |
| `support` | 某分类的人工标准答案样本数，用来判断该分类评估证据量 |

## 6. 输出文件说明

| 输出文件 | 作用 |
|---|---|
| `batch_metadata.json` | 记录批次编号、创建时间、预算上限、输出格式、选定后端、实际真实 / 本地 / mock 后端组合和成本估算口径 |
| `results.jsonl` | 保存每个文件的最终结构化结果；新生成文件采用缩进式连续 JSON 对象，每个字段独立换行，便于人工检查 |
| `results_readable.md` | 保存人工可读结果，便于检查每个文件的分类、摘要、证据、成本、耗时、请求模型名和服务端响应模型名 |
| `model_calls.jsonl` | 保存每次模型调用的任务类型、供应商、模型名、用量、成本、延迟和状态；新生成文件每个字段独立换行 |
| `errors.jsonl` | 保存失败或部分成功时的问题；新生成文件每个字段独立换行，便于程序解析和排查 |
| `batch_report.json` | 汇总整个批次的文件数、成功率、成本、延迟和质量风险 |
| `model_strategy_report.md` / `model_strategy_report.json` | 基于已有批次生成模型组合策略分析 |
| `routing_policy_simulation.md` / `routing_policy_simulation.json` | 基于已有批次模拟不同路由策略下的取舍 |
| `routing_preflight_report.md` / `routing_preflight_report.json` | 批处理前检查当前输入规模画像、历史延迟画像、任务级P95延迟目标、路由、预算约束、真实模型覆盖率和mock边界 |
| `cost_reconciliation_template.csv` | 成本对账手工录入模板，用于把供应商账单金额填回统一格式 |
| `cost_reconciliation.md` / `cost_reconciliation.json` | 多供应商成本对账报告，用于比较系统估算成本与供应商实际扣费，并记录账单粒度和可信度 |
| `cost_reprice_report.md` / `cost_reprice_report.json` | 成本重算报告，用于在官方价格目录更新后，按当前价格重算历史批次成本 |
| `text_topic_eval_report.md` / `text_topic_eval_report.json` | 保存文本主分类评估结果 |
| `image_ocr_eval_summary.md` / `image_ocr_eval_summary.json` | 保存图片 OCR 分段质量汇总，包括完整段落召回率、字符错误率和 OCR 延迟 |
| `image_ocr_error_analysis_img_9.md` / `image_ocr_error_analysis_img_9.json` | 保存 `img_9.jpg` 的 OCR 错误归因和闸门判断 |
| `image_ocr_gate_report_keyframes.md` / `image_ocr_gate_report_keyframes.json` | 保存三张关键帧图片的批次级 OCR 闸门判断，用于决定是否继续留在 OCR 功能内 |
| `image_ocr_preprocess_experiment_img_9.md` / `image_ocr_preprocess_experiment_img_9.json` | 保存 `img_9.jpg` 的 OCR 预处理实验结果，用于判断整图放大或分区放大是否值得继续 |
| `image_ocr_latency_profile_img_9.md` / `image_ocr_latency_profile_img_9.json` | 保存 `img_9.jpg` 的 OCR 延迟拆分结果，用于区分引擎创建、图片解码、模型推理和结果解析耗时 |
| `ocr_backend_advice.md` / `ocr_backend_advice.json` | 保存 OCR 后端取舍判断，用于决定是否继续本地PaddleOCR，还是下一轮评估本地ONNXRuntime或服务化OCR |
| `rapidocr_candidate_eval.md` / `rapidocr_candidate_eval.json` | 保存 RapidOCR 候选评估结果；当前已完成三张关键帧本地实测，整体完整段落召回率82.93%、字符错误率10.64%、P95延迟4294ms，闸门结论为未通过 |
| `failure_demo_interpretation.md` | 解释失败 / 部分成功演示批次中的错误链路和证据缺失 |

机器读取应使用三个 `.jsonl` 文件；人工逐项检查应优先使用 `results_readable.md`。从本版本开始，新批次的 `results.jsonl`、`model_calls.jsonl` 和 `errors.jsonl` 也采用缩进式连续 JSON 对象，让每个输出字段独立换行。它不是严格“一条记录一行”的标准 JSONL，但项目内部读取器已兼容这种格式。已有历史批次不会被自动改写。

## 7. 成本与延迟统计

成本与延迟来自两层记录：

- `model_calls.jsonl`：单次模型调用记录。
- `batch_report.json`：批次级聚合报告。

成本不是云厂商账单，而是工程估算：系统读取 `config/model_prices.yaml` 中的单价，把每条 `model_calls.jsonl` 里的 `input_units` 和 `output_units` 按计价单位相乘后求和。价格目录会记录 `price_source`、`price_updated_at` 和 `price_confidence`，新生成的模型调用记录会自动带出这些字段，避免用户每次运行都手工提交价格表。当前已新增官方公开价格页刷新入口，可从 Qwen-VL 和 DeepSeek 官方价格页抓取人民币价格并写回本地价格目录。对于 DeepSeek 和 Qwen-VL 这类真实 API，输入 / 输出 token 优先来自供应商响应中的 usage 字段；对于 mock 调用或本地 PaddleOCR，成本只用于流程占位或记录外部 API 成本为0，不包含本机 CPU、内存、电力和运维成本。新生成的 `batch_metadata.json` 会写入 `cost_estimation`，明确是否包含 mock 估算、是否已和真实账单对账。未与供应商账单对账前，整体估算误差不能给出百分比，也不能宣称误差小；只能说明本地计算公式可复现。

只抓取官方价格并生成报告，不写回配置：

```powershell
python .\src\price_catalog_updater.py .\config\model_prices.yaml .\output\price_catalog_refresh.json
```

抓取官方价格并写回本地价格目录：

```powershell
python .\src\price_catalog_updater.py .\config\model_prices.yaml .\output\price_catalog_refresh_20260801.json --apply
```

写回前会先执行价格刷新预检查：候选价格必须包含输入 token 和输出 token 两类计价单位，价格必须是正数且不能是 NaN / Infinity；如果候选价格相对本地旧价格变化超过50%，默认会被拦截，需要人工复核后显式增加 `--allow-large-change` 才允许写回。

本入口只访问官方公开文档页，不登录供应商控制台，不读取真实账单，不调用付费模型 API。若官方网页结构变化、网络异常或预检查失败，报告会保留失败原因，不会静默生成伪价格，也不会覆盖本地价格目录。

官方价格更新后，可以对历史批次按当前价格目录重新估算成本。这个命令不会改写历史 `model_calls.jsonl`，只生成重算报告：

```powershell
python .\src\cost_repricing.py .\output\batch_qwen_vl_response_model_check .\config\model_prices.yaml .\output\batch_qwen_vl_response_model_check\cost_reprice_report.json .\output\batch_qwen_vl_response_model_check\cost_reprice_report.md
```

当前 Qwen-VL 单图批次的重算结果显示：历史记录成本合计为0.013557元，按当前价格目录重算后为0.011923元，变化金额为-0.001634元。变化来自 `qwen-vl-plus` 官方价格刷新；历史记录不被覆盖。

价格自动抓取不建议每次模型调用时运行。当前建议频率和开启时机是：

- 每周刷新一次价格目录，用于降低官方价格变动造成的长期偏差。
- 每次做模型组合策略报告、路由策略预检查或预算评估前刷新一次。
- 每次准备跑真实 API 小样本前刷新一次。
- 如果账单对账发现估算值和实际扣费出现异常偏差，先刷新价格目录，再判断是否是免费额度、优惠、账期或计费规则差异。

正确性不靠“相信网页解析”，而靠四层约束：

- `source_url` 固定指向官方公开价格页，用于追溯价格来源。
- `parser` 对不同供应商页面使用显式解析规则，并有离线测试覆盖。
- 刷新报告保留 `old_pricing_rules` 和 `fetched_pricing_rules`，用于人工审查变更。
- 解析失败、网络失败、缺少必要计价单位、价格不是正数或价格大幅变化时生成错误报告，不静默覆盖本地价格目录。

当前已生成一次不写回配置的预检查报告：`output/price_catalog_refresh_preflight_20260801.json`。该报告中 Qwen-VL 和 DeepSeek 官方公开价格均抓取成功，`preflight_status` 均为 `pass`，价格未发生变化，配置没有被写回。这说明价格刷新入口可以在不改动本地配置的情况下先生成可审查报告。

模型组合策略报告现在可以使用成本重算报告作为成本口径输入：

```powershell
python .\src\model_strategy_advisor.py .\output\batch_qwen_vl_response_model_check .\output\batch_qwen_vl_response_model_check\cost_reprice_report.json
```

传入重算报告后，策略报告会使用 `cost_basis=current_repriced`，即用 `current_estimated_cost_cny` 参与成本分析；没有传入时仍使用历史 `cost_cny`。

当前多模态 Demo 批次的重要统计：

| 指标 | 数值 | 说明 |
|---|---:|---|
| 文件数 | 3 | 文本、图片、视频各 1 个 |
| 模型调用数 | 8 | 包含 OCR、视觉理解、语音识别和文本分析 |
| 总成本 | 0.042107 元 | 按本地价格配置估算 |
| DeepSeek 文本分析成本 | 0.002107 元 | 真实文本分析调用成本估算 |
| 平均文件处理耗时 | 2154.33 ms | 文件级平均耗时 |
| 模型调用 P95 延迟 | 3425 ms | 单次模型调用 95 分位延迟 |

注意：mock 上游任务的成本和延迟只能用于流程验证，不能代表真实供应商报价或性能。

## 8. 技术栈

- Python：主开发语言。
- PyYAML：读取配置文件。
- PaddlePaddle / PaddleOCR：作为图片本地 OCR 运行时；PaddlePaddle 3.3.0 已安装，并已完成五张正式图片的本地 CPU 推理与分段评估。
- RapidOCR / ONNXRuntime：作为本地 OCR 候选后端；已完成三张关键帧同批样本实测，但未通过当前质量和延迟闸门，暂不接入主流程。
- ffmpeg：作为视频音频提取的本地可选依赖；存在时用于抽取单声道 16kHz wav 音频文件，不代表真实 ASR 已接入。
- DeepSeek API：当前真实文本分析后端。
- Qwen-VL API：图片和视频关键帧视觉理解候选后端；已完成受保护接入入口、离线测试和 `img_1.png` 单图真实 API 验证。
- JSON / JSONL / Markdown：用于结构化输出、调用明细和人工可读结果。
- unittest：本地离线测试。

## 9. 相关文档

| 文档 | 作用 |
|---|---|
| `docs/architecture.md` | 说明系统模块、处理流程、模型路由、成本延迟追踪和当前架构限制 |
| `docs/demo_walkthrough.md` | 说明代表性输出如何复现和解读 |
| `docs/tests.md` | 说明测试范围、运行方式、未覆盖风险和后续测试计划 |
| `docs/deferred_technical_issues.md` | 记录暂时不便扩展的技术问题、暂缓原因、重新开启条件和小字号OCR归因 |

## 10. 当前限制

- 真实模型证据当前覆盖 DeepSeek 文本分析和 PaddleOCR 图片文字提取；PaddleOCR 已完成五张正式图片、共151段人工业务文字评估，但样本量仍不能外推为生产质量。
- 图片 OCR 默认仍为 mock，只有显式选择时才使用本地 PaddleOCR；图片视觉理解默认仍为 mock，已支持显式选择 Qwen-VL API，并已有 `img_1.png` 单图真实批次证据；视频预处理V1已能读取元信息，默认按 `start_early_then_spaced` 抽取最多3张关键帧，关键帧已能按配置进入 PaddleOCR 或 Qwen-VL，本地音频提取依赖 ffmpeg；`.flv` 已纳入视频输入识别。
- 当前 Qwen-VL 证据只覆盖单张图片的 `visual_description`，不能证明多图稳定质量。该批次中的 OCR 和文本分析仍是 mock，因此不能作为完整图片理解质量结论。
- Paddle 底层推理器对中文路径仍不稳定；直接使用 H 盘中文路径时模型创建失败，改用临时英文盘符映射和 `PADDLE_PDX_CACHE_HOME` 后成功运行。代码尚未自动处理该环境问题。
- `img_1.png` 的真实 OCR 耗时15733ms；`img_2.png` 独立冷启动批次耗时51096ms；三张关键帧图片的 OCR 平均延迟为18006ms、P95延迟为28261ms。当前 OCR 延迟仍高于既定图片2秒目标，本地算力、电力和运维成本尚未计量。
- 批次级 OCR 闸门报告显示，三张关键帧图片整体完整段落召回率78.05%、字符错误率11.01%、P95延迟28261ms，均未达到当前MVP观察阈值；其中 `img_9.jpg` 是主要质量阻塞样本，三张图都存在延迟阻塞。
- `img_9.jpg` 的预处理最小实验已完成：整图放大2倍把完整段落召回率从47.62%提升到52.38%，左右分区放大2倍提升到50.00%，但字符错误率仍为20.27%，延迟也分别达到64146ms和32421ms，因此预处理方向有轻微价值但不能通过闸门。
- `img_9.jpg` 的延迟拆分已完成：引擎创建耗时8834ms，首次图片推理60373ms，热启动第二次图片推理56042ms，图片解码和结果解析都不是主要瓶颈；即使不计引擎创建，本地CPU单图推理仍远高于2秒目标。
- RapidOCR 候选评估已经完成：三张关键帧整体完整段落召回率82.93%、字符错误率10.64%、P95延迟4294ms，虽然明显快于当前 PaddleOCR CPU 批次，但仍未达到90%召回、5%字符错误率和2秒P95延迟目标，因此不接入主流程。
- OCR 后端取舍判断已经更新：当前信号仍为 `evaluate_alternative_backends`，但 RapidOCR 已被标记为 `evaluated_not_passed`；如继续追求生产可用 OCR，只能在单独授权后小样本评估服务化 OCR，否则保留 PaddleOCR 作为当前本地基线。
- 运行时模型路由仍按任务类型固定选择模型；离线路由策略模拟和路由策略预检查已经实现，但它们都不会自动改写运行时调度。预检查中的受控试跑建议只给出缩小范围和安全命令，不等于动态路由；其中 mock 任务的延迟只会进入非阻塞风险解释，不能被当成真实模型性能通过。
- 多供应商成本对账层已经实现；当前 Qwen-VL 单图批次已填入供应商后台 0.00 元真实扣费，因此 Qwen-VL 对账项的成本可信度已进入 `period_level_reconciled`。本次对账结论不是“估算准确”，而是“系统已经能记录理论估算成本、实际扣费和差异原因”；免费额度只是这一次样例的差异原因，不代表生产环境常态。
- 视频 V1 当前证明本地预处理链路：能解析元信息、默认写出最多3张前段优先关键帧并把预处理产物写入结果；本机有 ffmpeg 时可抽取 wav 音频文件。代码已支持把关键帧送入 PaddleOCR 或 Qwen-VL，并已跑过小批量真实视频链路；但样本量仍不足以证明生产级完整视频OCR、完整视频视觉理解或真实语音识别质量。
- 九类规则回归仍只有18条已知样本、每类2条，不能外推为线上稳定质量或泛化能力。
- 回归批次当时有1条调用没有产出可解析JSON；结构化响应加固后已对该样本完成一次真实定向验证并成功，但重试分支只完成离线故障测试。
- 本轮 P95 模型延迟为10955ms，没有达到文本任务2秒以内的既定目标。
- `other` 分类在本轮两条样本均正确，但这只证明已知回归样本得到改善，不能证明新样本上的稳定召回。
- 当前质量评估只覆盖主分类；副分类、关键词、摘要和业务用途尚未建立独立人工评估。业务用途已完成单样本真实回归并增加高风险商业表述防护，但它不是完整的语义事实核验器。
- 当前没有数据库层和前端页面，主要通过本地文件输出验证工程链路。

## 11. 下一步技术方向

按“先做扎实一个功能，再开启下一个功能”的原则，图片 OCR 已形成本地基线、五图评估、关键帧闸门、弱样本归因、预处理实验、延迟拆分、RapidOCR 候选评估和低质量 OCR 结果闸门；结论是本地 OCR 路线仍未通过质量和延迟闸门。当前单独收束的是路由策略预检查：它在批处理前读取路由配置、价格表、输入规模和历史模型调用记录，判断预算、P95延迟、真实模型覆盖率和mock边界，并给出受控小样本试跑建议。它不会自动改路由，不会自动执行试跑，也不会调用任何模型。

建议顺序：

| 优先级 | 事项 | 目的 |
|---|---|---|
| 已完成 | 明确图片 OCR 评估口径 | 只统计账号名称、简介、作品标题和作品说明等业务内容文字 |
| 已完成 | 扩充图片 OCR 评估小集 | 当前有5张正式样本、151段人工业务文字，其中3张来自真实视频关键帧信息图 |
| 已完成 | 建立图片 OCR 评估器 | 计算文字块精确召回率和分段字符错误率，并保留逐段明细 |
| 已完成 | 受控执行 PaddleOCR 真实图片 | 五张正式图片均已处理；其中关键帧三图验证了结果解析、延迟和调用记录链路 |
| 已完成 | 增加指定文件筛选 | 使用 `--include-files` 只处理目标图片，避免误跑整个输入目录 |
| 已完成 | 分析 OCR 弱样本与延迟瓶颈 | `img_9.jpg` 错误集中在小字号结构图模块、Buffer 和 TLB 指标，且28261ms延迟不达标 |
| 已完成 | 生成关键帧 OCR 批次级闸门报告 | 结论为未通过：质量阻塞集中在 `img_9.jpg`，延迟阻塞覆盖三张关键帧图片 |
| 已完成 | 判断是否做 OCR 预处理实验 | 整图放大和左右分区放大均只带来轻微召回提升，字符错误率未改善，延迟更高，暂不能通过闸门 |
| 已完成 | 拆分 OCR 延迟来源 | `img_9.jpg` 的主要瓶颈是模型推理，图片解码和结果解析不是主要耗时来源 |
| 已完成 | 做 OCR 方案取舍判断 | RapidOCR已实测未过闸门；如继续追求生产可用OCR，服务化OCR只作为后续授权候选 |
| P0 | 收束低质量 OCR 结果闸门 | 确认 `low_quality_ocr_text` 的判断、测试、输出证据和文档一致，并保留受控批次证据 |
| P0 | 保留重试回归测试 | 后续修改DeepSeek客户端时，继续保证失败与成功尝试分别计量 |
| P1 | 把故障注入接入受保护演示命令 | 让失败 / 部分成功演示更容易复现，但默认不误触发 |
| 已完成 | 安装并评估一个OCR替代候选 | RapidOCR已用同一批关键帧图片和同一份人工基准对照当前PaddleOCR，结论为未通过 |
| 已完成 | 增加路由策略预检查 | 批处理前检查当前路线是否满足路由完整性、预算、P95延迟、真实覆盖率和mock边界要求；报告证据保存在 `output/routing_preflight_current/` |
| 已完成 | 增加本地音频提取最小闭环 | 视频预处理在本机存在 ffmpeg 时抽取 wav 音频文件；缺少 ffmpeg 或提取失败时记录明确状态；真实 ASR 仍未接入 |
| 已完成 | 补运行前规模画像 | 已基于 `input/` 生成预计文件数、图片数、视频数、音频秒数和token用量；预算可运行前估算 |
| 已完成 | 补历史 P95 延迟输入 | 已从已有 `model_calls.jsonl` 自动生成任务级历史延迟画像；当前受控 preflight 报告改为使用任务级P95目标，OCR 56401ms低于60000ms目标，文本分析7112ms低于8000ms目标，预检查状态为warning |
| 已完成 | 补受控小样本试跑建议 | 当前预算约束通过但仍有mock边界或延迟风险时，报告会建议最多3个文件、暂不纳入视频，并要求真实外部API试跑必须单独授权 |
| P1 | 判断是否授权服务化OCR小样本评估 | 如果继续突破OCR质量和延迟闸门，需要先确认API Key、费用、网络和数据风险 |
| P2 | 后续再整理展示材料 | 等核心功能更扎实后，再考虑是否补充图示、可视说明和对外说明 |
## 成本统计口径补充

新生成的 `batch_report.json` 中，`cost_stats.total_cost_cny` 只统计真实 API 的本地价格表估算成本，用于判断理论预算消耗，不等同供应商后台真实扣费；`recorded_total_cost_cny` 记录所有模型调用里的成本字段合计，包括 mock 占位成本；`live_api_cost_cny` 记录真实 API 估算成本；`local_model_cost_cny` 记录本地模型成本；`mock_cost_cny` 记录 mock 占位成本，不代表真实扣费。

这样做是为了避免视频批次中 mock ASR 或 mock 文本分析的占位成本被误读为供应商真实成本。供应商实际扣费仍需要通过成本对账报告单独验证；如果供应商使用免费额度抵扣，后台真实扣费可能为 0，此时不能用估算成本直接反推真实扣款。

## 输入格式识别范围补充

当前文件扫描入口会按后缀把输入文件识别为文本、图片或视频。文本类支持 `.txt`、`.text`、`.md`、`.csv`、`.tsv`、`.json`、`.jsonl`、`.yaml`、`.yml`、`.xml`、`.html`、`.htm`、`.srt`、`.vtt`、`.log`；图片类支持 `.jpg`、`.jpeg`、`.png`、`.bmp`、`.webp`、`.jfif`、`.tif`、`.tiff`；视频类支持 `.mp4`、`.mov`、`.avi`、`.mkv`、`.webm`、`.flv`、`.m4v`、`.wmv`、`.mpg`、`.mpeg`、`.3gp`、`.3g2`、`.mts`、`.m2ts`、`.ts`。

这里的“支持”只表示文件会进入对应流水线；文本仍按 UTF-8 读取，图片仍依赖当前 OCR / 视觉理解后端能否实际解码，视频仍依赖 OpenCV 和 ffmpeg 能否解析。PDF、Word、HEIC 和 GIF 动图暂不纳入，避免只靠后缀放行但主流程无法稳定处理。
