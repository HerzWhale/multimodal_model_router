# Demo Walkthrough

这份文档用于解释当前代表性输出如何运行、怎么看，以及哪些结论可以相信、哪些不能外推。

## 1. 多模态 Demo 批次

当前保留的多模态批次：

```text
H:\实习\multimodal_model_router\output\batch_20260718_150348
```

这个批次包含文本、图片和视频三类输入，适合检查统一输入、统一输出、调用记录、成本延迟追踪和 DeepSeek 文本分析链路。

| 输入文件 | 类型 | 当前作用 |
|---|---|---|
| `ai_content_sample.txt` | 文本 | 展示文本读取、DeepSeek 文本分析、分类、摘要和业务用途生成 |
| `img.png` | 图片 | 展示图片文件进入 OCR、视觉理解和文本分析流水线；OCR 和视觉理解为 mock |
| `例子.mp4` | 视频 | 旧批次展示视频文件进入统一流水线；历史视频 V0 批次只能抽取第一帧；当前代码已升级为视频 V1，可读取元信息并等距抽取最多 5 张关键帧，这些关键帧可交给 PaddleOCR 或 Qwen-VL；本机存在 ffmpeg 时可抽取 wav 音频文件，但真实 ASR 尚未接入 |

字段说明：

| 字段 | 含义与作用 |
|---|---|
| `file_name` | 原始文件名，用来识别每条结果对应哪份内容 |
| `media_type` | 文件媒体类型，用来决定进入文本、图片还是视频流程 |
| `source_path` | 原始文件路径，用来追溯输入文件来源 |
| `preprocessing_artifacts` | 预处理产物摘要，用来说明视频元信息、关键帧抽取、音频提取状态和风险边界 |

## 1.1 视频 V1 多关键帧预处理说明

当前视频 V1 受控 mock 批次：

```text
H:\实习\multimodal_model_router\output\batch_video_keyframes_v1_mock_trial
```

运行命令：

```powershell
python .\src\main.py --input-dir .\input\sample_videos --include-files 例子.mp4 --text-analysis-backend mock --batch-id batch_video_keyframes_v1_mock_trial
```

该批次只验证视频 V1 多关键帧工程闭环，不调用 DeepSeek，不调用 Qwen-VL，不运行 PaddleOCR，不调用云端OCR或ASR。

当前 V1 结果应这样读：

- `preprocessing_artifacts` 显示视频元信息已经读取成功：时长约301567ms，9047帧，30fps，分辨率720×1280；
- `keyframe_paths` 记录已写出的5张等距关键帧路径，说明视频画面证据覆盖了开头、中段和结尾附近的多个时间点；
- `keyframe_metadata` 记录每张关键帧的源帧号和估算时间位置，例如0ms、75400ms、150767ms、226133ms和301533ms；
- `model_calls.jsonl` 中有5次mock OCR、5次mock视觉理解、1次预期语音识别失败和1次mock文本分析；该历史批次生成于本地音频提取最小闭环之前；
- 当前代码如果本机存在 ffmpeg，会尝试生成 wav 音频文件并进入 mock 语音识别分支；如果缺少 ffmpeg，会把 `audio_extraction_status` 记录为 `dependency_missing`，并继续把 `audio_transcript` 作为缺失证据；
- 该批次不能解释为真实视频OCR质量、真实视频视觉理解质量或真实ASR质量。

历史视频 V0 受控批次：

```text
H:\实习\multimodal_model_router\output\batch_video_v0_preprocess_20260804
```

运行命令：

```powershell
python .\src\main.py --input-dir .\input\sample_videos --include-files 例子.mp4 --text-analysis-backend mock --batch-id batch_video_v0_preprocess_20260804
```

该历史批次只验证本地视频预处理闭环，不调用 DeepSeek，不调用 Qwen-VL，不运行 PaddleOCR，不调用云端OCR或ASR。需要注意：它是 V0 单帧证据，不能代表当前 V1 多关键帧能力。

当前结果应这样读：

- `preprocessing_artifacts` 显示视频元信息已经读取成功：时长约301567ms，9047帧，30fps，分辨率720×1280；
- `keyframe_paths` 在历史 V0 批次中只记录第一帧路径，说明当时的视频画面证据只有一个本地产物可以追踪；
- `audio_extraction_status=not_implemented` 是历史 V0 批次的旧状态，表示当时未实现音频提取；当前代码会进一步区分 `extracted`、`dependency_missing`、`failed`、`timeout`、`empty_output` 和 `not_attempted_no_artifact_dir`；
- 文件级 `processing_status=partial_success`，原因不是视频文件完全失败，而是“关键帧预处理成功，但音频证据缺失”；
- `model_calls.jsonl` 中 OCR 和视觉理解仍是 mock 调用，不能解释为真实视频OCR或真实视频视觉理解质量；
- `speech_to_text` 记录为 failed，且 `input_units` 为0，表示未调用ASR，不应把视频时长虚算成已发生的ASR成本。

从当前代码版本起，如果要验证视频 V1 多关键帧进入上游模型，可以分别运行以下受控命令。

本地 PaddleOCR 关键帧 OCR：

```powershell
python .\src\main.py --input-dir .\input\sample_videos --include-files 例子.mp4 --ocr-backend paddleocr --text-analysis-backend mock --batch-id batch_video_keyframes_paddleocr_trial
```

Qwen-VL 关键帧视觉理解：

```powershell
python .\src\main.py --input-dir .\input\sample_videos --include-files 例子.mp4 --vision-backend qwen_vl --allow-live-api --text-analysis-backend mock --batch-id batch_video_keyframes_qwen_vl_trial
```

第二条命令会访问 Qwen-VL 并可能产生费用，必须由用户明确授权后再运行。两条命令都会处理视频 V1 抽出的最多 5 张等距关键帧，但仍不代表完整视频理解；当前只有本地音频提取最小闭环，真实音频转写仍未实现。

如果要验证 Qwen-VL 关键帧级重试，可以在确认费用和输入范围后增加：

```powershell
--max-api-retries 1
```

该参数只允许对可重试错误最多重试1次。某张关键帧第一次失败、第二次成功时，`model_calls.jsonl` 会同时保留失败尝试和成功尝试；最终 `visual_description` 会使用成功尝试的画面描述。如果某张关键帧重试后仍失败，系统会保留其他成功关键帧的画面描述，并把该文件标记为 `partial_success`，同时写入 `quality_flags=video_visual_keyframe_failed`。这属于失败补偿，不是完整视频理解能力。

Qwen-VL 请求图片默认会按 `qwen_vl_max_image_side=960` 生成压缩后的请求副本，只影响发给 Qwen-VL 的 Base64 图片，不会改写本地原图，也不会影响 PaddleOCR 使用原始图片。该配置用于降低视觉理解延迟风险；是否达标仍以真实小批次的 `latency_ms` 和批次 P95 为准。

字段说明：

| 字段 | 含义与作用 |
|---|---|
| `keyframe_paths` | 关键帧本地路径列表，用来追踪视频画面证据来自哪些预处理产物 |
| `keyframe_metadata` | 每张关键帧的帧号、时间位置和路径，用来说明视频画面证据覆盖了哪些时间点 |
| `keyframe_extraction_status` | 关键帧抽取状态，用来区分已抽取、未尝试、读取失败或写入失败 |
| `audio_extraction_status` | 音频提取状态，用来说明是否已经生成真实音频文件；`extracted` 表示已写出音频，`dependency_missing` 表示本机缺少 ffmpeg，其他失败状态用于排查提取问题 |
| `duration_source` | 视频时长来源，用来说明 `duration_ms` 是由 OpenCV 帧数和FPS估算，还是当前不可获得 |
| `missing_evidence` | 缺失证据列表，用来解释为什么结果不是完全成功 |

## 2. 运行方式

进入项目目录：

```powershell
cd H:\实习\multimodal_model_router
```

安装依赖：

```powershell
pip install -r requirements.txt
```

默认运行使用 mock 文本分析，不访问外部 API：

```powershell
python .\src\main.py
```

如需真实调用 DeepSeek 文本分析，需要先在本机设置环境变量：

```powershell
$env:DEEPSEEK_API_KEY = "你的 DeepSeek API Key"
```

然后显式选择 DeepSeek 并授权真实 API 调用：

```powershell
python .\src\main.py --text-analysis-backend deepseek --allow-live-api
```

缺少 `--allow-live-api` 时，程序会在读取文件和发送网络请求前拒绝 DeepSeek 运行，避免误调用和意外费用。

默认运行只读取 `input/`。如果要处理 `evaluation/` 中的文本评估样本，需要显式指定输入目录：

```powershell
python .\src\main.py --input-dir evaluation\text_topic_small_set --text-analysis-backend mock --batch-id batch_eval_mock
```

这条命令只用于离线验证评估入口，不触发 DeepSeek API。真实评估必须同时使用 `--text-analysis-backend deepseek` 和 `--allow-live-api`，并会产生费用。

真实运行默认不自动重试。`--max-api-retries 1` 表示只对可重试错误额外请求一次；每次尝试都会独立进入模型调用明细。该参数会增加最多一次请求成本，必须在确认输入范围后显式使用。

字段说明：

| 字段 | 含义与作用 |
|---|---|
| `input_dir` | 本次批处理读取的输入目录，用来区分默认业务输入和受控评估样本 |
| `batch_id` | 批次唯一标识，用来定位本次运行产生的输出目录 |
| `text_analysis_backend` | 文本分析后端配置，用来决定使用 mock 还是 DeepSeek |
| `vision_understanding_backend` | 图片或视频关键帧视觉理解后端配置，用来决定使用 mock 还是 Qwen-VL |
| `--max-api-retries` | 可重试错误的最大重试次数；默认0，显式设为1才允许一次重试；当前适用于 DeepSeek 和 Qwen-VL |
| `--allow-live-api` | 真实 API 调用授权开关，用来防止误触发外部调用和费用 |
| `text_analysis_evidence_char_limit` | 文本分析输入证据字符上限，用于减少长视频证据直接塞入文本分析模型造成的延迟风险；原始证据仍保留在结果文件中 |
| `qwen_vl_max_image_side` | Qwen-VL 请求图片副本最长边上限，用来减少图片请求体积和视觉理解延迟风险；原图不被修改 |

## 3. `results_readable.md` 怎么读

`results_readable.md` 是人工可读的文件级结果。重点看每个文件的状态、分类、摘要、证据、模型使用、成本和耗时。

关键字段说明：

| 字段 | 含义与作用 |
|---|---|
| `file_id` | 单个文件的唯一标识，用来关联最终结果和模型调用记录 |
| `processing_status` | 文件处理状态，用来判断该文件是否成功生成结果 |
| `topic` | 主分类，用来说明内容最主要的业务归属 |
| `secondary_topics` | 副分类，用来说明内容是否存在交叉领域 |
| `tags` | 关键词，用于后续搜索、筛选和素材管理 |
| `summary` | 内容摘要，用于快速理解文件内容 |
| `business_use` | 业务用途说明，用来解释结构化结果可以支持什么业务动作 |
| `evidence_used` | 实际使用的证据，用来判断结果基于原文、OCR、音频转写还是视觉描述 |
| `missing_evidence` | 缺失证据，用来说明结果可信度可能受到什么影响 |
| `models_used` | 文件级模型使用摘要，用来快速看到该文件经过哪些模型 |
| `response_model_name` | 服务端响应模型名称，用来在人工可读结果中核对实际响应来自哪个模型或模型别名 |
| `processing_cost_cny` | 文件级处理成本，单位人民币 |
| `processing_time_ms` | 文件级处理耗时，单位毫秒 |

本批次中文本文件的文本分析由 DeepSeek 真实生成。图片和视频结果需要如实按工程链路解读：OCR、视觉理解、语音识别仍是 mock，因此不能把这部分说成真实识别效果。

机器处理时应读取 `results.jsonl`：它保存文件级结构化记录。从当前版本起，新批次中的 JSON 对象采用缩进式连续写法，每个输出字段独立换行，人工查看时不会挤在一行。新版 `results_readable.md` 会在“使用模型”里显示 `response_model_name`；历史批次不会自动改写。项目读取器兼容这种缩进式连续 JSON 对象，但它不是严格“一条记录一行”的标准 JSONL。

## 3.1 `batch_metadata.json` 怎么读

`batch_metadata.json` 用来解释这次批处理的运行口径，而不是保存单个文件结果。新批次会记录三个关键块：

| 字段 | 含义与作用 |
|---|---|
| `selected_backends` | 本次命令或配置选择的后端组合，用来说明 OCR、视觉理解和文本分析分别选择了什么 |
| `backend_runtime_summary` | 根据实际 `model_calls.jsonl` 汇总出的真实 API、本地模型和 mock 组合，用来判断本批次是否混合了真实与 mock 后端 |
| `cost_estimation` | 成本估算说明，用来记录价格表来源、计算方法、是否包含 mock 估算、误差状态和是否已与真实账单对账 |

注意：`cost_estimation` 只能说明估算方法是否可复现，不能自动证明估算值接近真实扣费。只有把本地估算值与供应商后台账单进行对账后，才能讨论整体误差范围；未对账前应记录为误差未知。

如果 `backend_runtime_summary.contains_live_api=true` 且 `backend_runtime_summary.contains_mock=true`，说明该批次是“真实 API + mock”的混合批次。此时只能把真实 API 对应分支当作真实证据，不能把整个文件处理结果解释成全真实链路。

## 4. `batch_report.json` 怎么读

`batch_report.json` 是批次级统计报告，用来回答“这批任务整体表现如何”。

当前多模态批次核心结果：

| 指标 | 数值 | 含义 |
|---|---:|---|
| 总文件数 | 3 | 本批次处理了 3 个输入文件 |
| 成功文件数 | 3 | 3 个文件都生成了文件级结果 |
| 成功率 | 100% | 成功文件数除以总文件数 |
| 总成本 | 0.042107 元 | 本批次全部模型调用成本合计 |
| 平均每文件成本 | 0.014036 元 | 总成本除以文件数 |
| 平均文件处理耗时 | 2154.33 ms | 文件级处理耗时平均值 |
| 模型调用 P95 延迟 | 3425 ms | 单次模型调用 95 分位延迟 |

字段说明：

| 字段 | 含义与作用 |
|---|---|
| `file_stats` | 文件处理统计，用来查看总数、成功数、失败数和成功率 |
| `cost_stats` | 成本统计，用来查看总成本、平均成本、供应商成本和预算使用率 |
| `latency_stats` | 延迟统计，用来查看平均耗时、P95 延迟和最慢文件 |
| `error_quality_stats` | 错误和质量统计，用来查看失败原因、质量风险和部分成功原因 |

## 5. `model_calls.jsonl` 怎么读

`model_calls.jsonl` 是模型调用明细。每一个 JSON 对象表示一次模型调用；新批次采用缩进式连续 JSON 对象，每个调用字段独立换行，项目内部读取器可解析，但普通“逐行 JSONL”工具可能无法直接读取。

当前多模态批次共有 8 次调用：

| 文件 | 调用链路 |
|---|---|
| `img.png` | mock OCR → mock 视觉理解 → DeepSeek 文本分析 |
| `ai_content_sample.txt` | DeepSeek 文本分析 |
| `例子.mp4` | mock OCR → mock 视觉理解 → mock 语音识别 → DeepSeek 文本分析 |

关键字段说明：

| 字段 | 含义与作用 |
|---|---|
| `call_id` | 单次模型调用的唯一标识，用来区分同一文件触发的多次调用 |
| `batch_id` | 批次唯一标识，用来把多次文件处理归到同一批任务下 |
| `file_id` | 文件唯一标识，用来把模型调用和最终文件结果关联起来 |
| `task_type` | 模型调用任务类型，例如 OCR、视觉理解、语音识别或文本分析 |
| `provider` | 模型供应商，用来做供应商维度成本和延迟统计 |
| `model_name` | 具体模型名称，用来追踪结果来自哪个模型 |
| `response_model_name` | 服务端响应模型名称，用来核对请求模型和供应商实际返回模型是否一致 |
| `input_units` | 输入用量和单位，例如输入 token、图片数量、帧数或音频秒数 |
| `output_units` | 输出用量和单位，例如输出 token 或文本字符数 |
| `cost_cny` | 单次调用成本，单位人民币 |
| `latency_ms` | 单次调用延迟，单位毫秒 |
| `status` | 单次调用状态，用来判断该调用成功还是失败 |
| `error_message` | 错误信息；成功时为空，失败时用于排查原因 |

## 6. 策略报告怎么读

策略报告来自已有批次，不重新跑批处理，也不触发 DeepSeek API。

```text
output/batch_20260718_150348/model_strategy_report.md
output/batch_20260718_150348/model_strategy_report.json
output/batch_20260718_150348/routing_policy_simulation.md
output/batch_20260718_150348/routing_policy_simulation.json
```

## 6.1 成本对账报告怎么读

成本对账报告来自已有 `model_calls.jsonl` 和手工账单 CSV，不调用任何供应商 API。当前 Qwen-VL 单图批次已经生成：

```text
output/batch_qwen_vl_response_model_check/cost_reconciliation_template.csv
output/batch_qwen_vl_response_model_check/cost_reconciliation_billing_free_quota.csv
output/batch_qwen_vl_response_model_check/cost_reconciliation.json
output/batch_qwen_vl_response_model_check/cost_reconciliation.md
```

当前 Qwen-VL 单图批次已经基于供应商后台显示的 0.00 元实际扣费完成一次对账：系统估算成本为 0.003237 元，`billed_cost_cny` 填入 0.00 元，Qwen-VL 对账项的 `cost_confidence` 进入 `period_level_reconciled`，汇总层通过 `summary.confidence_counts` 记录周期级对账数量。这次结论不是“估算准确”，而是“系统已经能把理论估算成本、供应商实际扣费和差异原因记录到同一份报告中”。免费额度只是本次样例的差异原因，不代表生产环境常态。原始空模板仍保留为 `cost_reconciliation_template.csv`，已填写本次后台账单证据的模板为 `cost_reconciliation_billing_free_quota.csv`。

对账模板有两类保护规则：第一，`billed_cost_cny` 为空时表示未对账；非数字、负数、NaN 或 Infinity 会被拒绝。第二，同一供应商、同一请求模型、同一响应模型和重叠时间窗口不能出现多条账单记录，避免被后写入记录静默覆盖。

如果要做一次真实成本校准，可以新建一个干净时间窗口，只运行 Qwen-VL 图片视觉理解：

```powershell
python .\src\main.py --input-dir .\input\sample_images --include-files img_1.png,img_7.jpg,img_8.jpg --ocr-backend mock --vision-backend qwen_vl --text-analysis-backend mock --allow-live-api --batch-id batch_qwen_vl_cost_calibration_20260804
```

运行前后不要同时运行 DeepSeek、PaddleOCR、ASR 或其他真实 API。运行完成后，先生成成本对账模板：

```powershell
python .\src\cost_reconciliation.py template .\output\batch_qwen_vl_cost_calibration_20260804 .\output\batch_qwen_vl_cost_calibration_20260804\cost_reconciliation_template.csv
```

再把供应商后台对应时间窗口内的真实扣费填入 `billed_cost_cny`，并生成对账报告：

```powershell
python .\src\cost_reconciliation.py reconcile .\output\batch_qwen_vl_cost_calibration_20260804 .\output\batch_qwen_vl_cost_calibration_20260804\cost_reconciliation_template.csv .\output\batch_qwen_vl_cost_calibration_20260804\cost_reconciliation.json .\output\batch_qwen_vl_cost_calibration_20260804\cost_reconciliation.md
```

没有真实账单金额前，不能把估算成本写成已验证成本。如果供应商后台显示免费额度抵扣，应填入真实扣费 `0.00`，并在账单来源和备注中说明免费额度来源。

字段说明：

| 字段 | 含义与作用 |
|---|---|
| `estimated_cost_cny` | 系统根据用量和本地价格表计算出的估算成本 |
| `billed_cost_cny` | 供应商后台显示或账单导出的实际扣费 |
| `cost_delta_cny` | 实际扣费减去估算成本后的金额差 |
| `cost_delta_rate` | 成本偏差比例，用于观察误差规模 |
| `billing_granularity` | 账单粒度，用来区分单次调用、小时级、日级或模型级对账 |
| `bill_source` | 真实扣费来源，例如供应商控制台人工查看或供应商导出文件 |
| `matching_method` | 系统调用记录和供应商账单的匹配方式，例如供应商、模型和时间窗口匹配 |
| `bill_reconciled` | 是否已经填入供应商账单金额并完成对账 |
| `cost_confidence` | 成本可信度状态，用于区分未验证、单次调用级对账和时间段级对账 |
| `matched_call_ids` | 实际参与账单核对的模型调用编号列表 |
| `unmatched_billing_records` | 没有匹配到本批次模型调用的账单记录，用于排查账单时间窗口或模型名称错误 |

推荐阅读顺序：

| 报告部分 | 能回答的问题 |
|---|---|
| 批次概览 | 这批任务整体规模和表现如何 |
| 成本分析 | 哪些任务或模型贡献了主要成本 |
| 延迟分析 | 当前瓶颈大概率在哪里 |
| 质量与可信度边界 | 哪些结论可以相信，哪些不能夸大 |
| 模型组合建议 | 面对预算、延迟、质量目标时应该怎么取舍 |

字段说明：

| 字段 | 含义与作用 |
|---|---|
| `model_call_count` | 模型调用次数，用来衡量本批次实际触发了多少次模型任务 |
| `cost_share` | 成本占比，用来判断某个任务、供应商或模型是否是主要成本来源 |
| `is_mock` | 是否为 mock 调用，用来区分真实模型调用和占位调用 |
| `real_model_calls` | 真实模型调用汇总，用来说明当前批次哪些部分可以作为真实模型证据 |
| `mock_model_calls` | mock 调用汇总，用来说明当前批次哪些部分只能证明流程跑通 |
| `missing_data_notes` | 数据缺失说明，用来提示某些报告字段无法从现有批次中取得 |

### 6.1 路由策略预检查怎么读

路由策略预检查用于批处理开始前，不读取业务结果，也不调用任何模型。它回答的是“当前配置能不能进入受控试跑”，不是“模型质量好不好”。

当前预检查输出位于：

```text
output/routing_preflight_current/routing_preflight_report.md
output/routing_preflight_current/routing_preflight_report.json
```

复现命令：

```powershell
python .\src\main.py --preflight-only --input-dir .\input\sample_videos --include-files "例子.mp4,例子1.mp4,例子2.mp4,例子3.mp4" --ocr-backend paddleocr --vision-backend qwen_vl --speech-backend dashscope_asr --text-analysis-backend deepseek --historical-model-calls .\output\batch_video_evidence_gate_check_retry\model_calls.jsonl --batch-id preflight_video_full_real_current
```

本次预检查基于 4 个视频文件、显式选择的 PaddleOCR、Qwen-VL、DashScope ASR、DeepSeek 后端和已有历史调用批次生成。它只读取本地文件清单、配置和历史 `model_calls.jsonl`，不触发任何真实模型调用，也不产生新的费用。

当前结果应这样解读：

- `preflight_status` 为 `warning`，表示运行前没有硬阻塞，但仍有预算、价格目录或成本估算边界需要解释；
- `workload_profile` 显示本次纳入4个视频样本；
- `latency_profile` 从已有历史调用记录中提取任务级延迟：OCR、视觉理解、语音识别和文本分析都有历史证据；
- `task_latency_targets_ms` 使用任务级P95目标：OCR 为60000ms，视觉理解为20000ms，语音识别为10000ms，文本分析为12000ms；这些目标是当前受控试跑闸门，不是线上生产SLA；
- `task_latency_target_checks` 显示当前四类任务均满足受控试跑目标；这只说明可以继续小批量验证，不代表线上延迟达标；
- `latency_bottleneck_analysis` 把慢因拆成真实外部 API、本地运行和 mock 占位；当前没有硬性延迟阻塞，但历史样本仍不能外推为稳定 SLA；
- `current_route` 显示 OCR、视觉理解、语音识别和文本分析均为非mock路线；
- `real_coverage_rate` 为 100%，表示当前4类预期任务都走非mock路线；
- `expected_units_by_task` 基于运行前假设生成：OCR 12张关键帧、视觉理解12帧、文本分析输入3200 token和输出1200 token；因为未提供视频音频秒数，语音识别成本仍无法估算；
- `budget_limit_cny` 当前未显式提供，因此预算检查为 `unknown`，不能硬判断预算是否达标；
- `p95_latency_limit_ms` 在当前配置中保留为全局兜底值；当 `task_latency_targets_ms` 已配置时，约束检查优先使用任务级目标，避免用同一个3500ms同时要求OCR、本地推理和文本API；
- `blocking_reasons` 为空，表示当前没有硬阻塞项；
- `warning_messages` 会提示价格目录过期或可信度不足、预算缺少显式上限等边界；
- `controlled_trial_plan` 会根据当前 `warning` 状态给出受控试跑边界；它只生成建议，不会自动执行真实模型调用。

因此，本报告的结论不是“可以直接扩大运行”，而是：当前真实模型覆盖率达标，任务级延迟闸门在受控试跑口径下通过，但预算、价格目录和语音识别成本估算仍有边界。下一轮仍应小批量、受控范围跑，不能直接扩大到完整输入目录。

如果把同一批历史延迟切换到 `production_sla` 口径，命令为：

```powershell
python .\src\main.py --preflight-only --preflight-policy production_sla --input-dir .\input\sample_videos --include-files "例子.mp4,例子1.mp4,例子2.mp4,例子3.mp4" --ocr-backend paddleocr --vision-backend qwen_vl --speech-backend dashscope_asr --text-analysis-backend deepseek --historical-model-calls .\output\batch_video_evidence_gate_check_retry\model_calls.jsonl --batch-id preflight_video_full_real_production_sla_current
```

当前生产候选 SLA 报告为 `fail`：`visual_understanding` 和 `text_analysis` 超过各自任务级 P95 目标。这说明当前路线能做受控工程验证，但还不能写成生产 SLA 达标。

字段说明：

| 字段 | 含义与作用 |
|---|---|
| `preflight_status` | 运行前预检查总状态，用来判断当前配置是可继续、存在风险还是不建议直接运行 |
| `workload_profile` | 运行前规模画像，用来统计输入文件数量、媒体类型分布和预估任务单位 |
| `latency_profile` | 历史延迟画像，用来从已有调用记录中汇总任务级平均延迟、P95延迟和最大延迟 |
| `latency_bottleneck_analysis` | 延迟阻塞归因，用来把慢因拆成真实外部API、本地运行和mock占位三类 |
| `real_api_slow_tasks` | 真实外部API慢任务列表，用来判断哪些真实网络调用超过当前P95目标 |
| `local_runtime_slow_tasks` | 本地运行慢任务列表，用来判断哪些慢因来自本机PaddleOCR等本地推理链路 |
| `mock_latency_unusable_tasks` | mock延迟不可用任务列表，用来提醒这些延迟不能作为真实供应商性能证据 |
| `task_latency_targets_ms` | 任务级P95延迟目标，用来给 OCR、视觉理解、文本分析等不同耗时结构的任务设置不同闸门 |
| `task_latency_target_checks` | 任务级延迟检查明细，用来说明每个当前预期任务的历史P95、目标P95、证据口径和状态 |
| `task_latency_target_summary` | 任务级延迟检查汇总，用来判断当前批次是否存在任务级延迟硬阻塞或mock证据风险 |
| `expected_units_by_task` | 各任务的预估计量单位，用来把单位价格转换成整批预算估算 |
| `historical_p95_latency_by_task_ms` | 各任务的历史P95延迟，用来判断运行前延迟约束是否可能失败 |
| `current_route` | 当前每个任务类型对应的供应商和模型，用来核对批处理真正会走哪条模型路线 |
| `budget_limit_cny` | 预算上限，用来判断预估用量下的模型组合是否可能超预算 |
| `p95_latency_limit_ms` | P95延迟限制，用来判断最慢的高分位任务延迟是否超过业务目标 |
| `min_real_coverage_rate` | 最低真实模型覆盖率，用来约束mock任务占比不能过高 |
| `blocking_reasons` | 硬阻塞原因列表，用来说明为什么当前配置不应直接扩大运行 |
| `warning_messages` | 风险提示列表，用来说明哪些地方可以继续试跑但不能过度解读 |
| `controlled_trial_plan` | 受控小样本试跑建议，用来在预算通过但延迟失败时说明下一轮应该缩到哪些文件、如何拆开 mock / PaddleOCR / DeepSeek 试跑、哪些调用需要授权 |
| `suggested_include_files` | 建议传给 `--include-files` 的文件名列表，用来避免误跑完整输入目录 |
| `trial_commands` | 报告生成的试跑命令参考，只供人工选择执行；报告本身不会执行命令或触发模型调用 |

## 7. 文本主分类评估怎么读

文本主分类评估用于回答：DeepSeek 输出的 `topic` 主分类是否命中人工标准答案。

当前九类规则回归批次：

```text
output/batch_text_eval_20260722_135443
```

该批次在完整九类规则补入提示词后，重新处理18条已移除分类答案提示的人工改写文本，没有处理图片、视频或默认 `input/` 目录。9个主分类各有2条人工标准答案，其中4条为约850—950个中文字符的长难例。

人工标准答案保存在 `evaluation/text_topic_gold.csv`。该文件使用带BOM的UTF-8编码，Windows Excel直接打开时可以自动识别中文；评估器使用 `utf-8-sig` 读取，因此BOM不会混入第一列表头。

本次受控评估结果：

| 指标 | 数值 | 解读 |
|---|---:|---|
| 文本样本数 | 18 | 纳入评估的无答案提示样本 |
| 有效预测数 | 17 | 成功产出九类范围内主分类的样本数 |
| 缺少预测数 | 1 | 1条响应无法解析为JSON，没有主分类结果 |
| 正确数 | 17 | 按全部18条统计，模型主分类与人工主分类一致的数量 |
| 端到端 Accuracy | 94.44% | 无有效预测按未命中计算 |
| 有效预测 Accuracy | 100.00% | 17条有效预测全部分类正确 |
| 预测覆盖率 | 94.44% | 有效九分类预测占全部已标注样本的比例 |
| Macro-F1 | 96.30% | 9个业务分类的F1等权平均 |
| 总成本 | 0.027852 元 | 本批次18次 DeepSeek 文本分析的实际记录成本 |
| 平均模型延迟 | 4356.78 ms | 18次真实模型调用的平均耗时 |
| P95 模型延迟 | 10955 ms | 单次模型调用 P95 延迟 |

约束解读：总成本0.027852元，仅占50元预算上限约0.0557%；但 P95 模型延迟10955ms高于文本任务原定2000ms上限，因此成本约束通过、延迟与稳定性约束未通过。

字段说明：

| 字段 | 含义与作用 |
|---|---|
| `predicted_topic` | 从模型结果中提取出的预测主分类，用来和人工答案对比 |
| `gold_topic` | 人工标注的正确主分类，是计算准确率的基准 |
| `reviewer_note` | 人工评审备注，用来解释某条样本为什么属于某个主分类 |
| `evaluated_count` | 已纳入评估的文本样本数，用来判断准确率的样本基础 |
| `correct_count` | 预测主分类与人工主分类一致的样本数 |
| `accuracy` | 文本主分类准确率，计算方式为 `correct_count / evaluated_count` |
| `valid_prediction_accuracy` | 有效九分类预测中的准确率，用来把分类判断和调用失败分开观察 |
| `prediction_coverage` | 有效九分类预测占已评估样本的比例，用来观察调用和解析稳定性 |
| `missing_prediction_count` | 没有产出有效主分类的样本数，用来定位调用或解析失败 |
| `macro_f1` | 各参与评估分类 F1 的简单平均，用于避免总体正确率掩盖小类别问题 |
| `precision` | 预测为某个分类的样本中实际属于该分类的比例，用于观察误报 |
| `recall` | 实际属于某个分类的样本中被正确识别的比例，用于观察漏报 |
| `f1` | 单个分类 Precision 与 Recall 的调和平均，用于综合衡量误报和漏报 |
| `support` | 某分类的人工标准答案样本数，用于判断分类证据量 |

解读边界：

- 这个批次能证明九类规则修正后，原4条已知错例都得到正确分类，17条有效预测没有语义误判。
- `other` 类两条样本均正确，Recall为100%；但样本参与过规则诊断，不能把该数值解释为独立泛化表现。
- 第14条体育健康样本没有预测结果；端到端Accuracy、有效预测Accuracy和预测覆盖率必须同时看，不能用100.00%掩盖调用失败。
- P95模型延迟没有达到文本任务2秒以内的目标；17次成功返回也不能被解释为延迟和稳定性达标。
- 当前评估不覆盖摘要质量、关键词质量、副分类质量和业务用途质量，也不覆盖图片或视频结果。
- 18条样本仍然很小且由人工改写，不能把本轮指标外推为线上稳定准确率。

结构化响应加固后，原失败的第14条样本在 `output/batch_text_retry_20260722_192832/` 中完成了单样本真实验证：第一次调用成功，主分类为体育健康，与人工答案一致；成本0.0015元，延迟4386ms，未触发第二次请求。原18条批次的94.44%端到端Accuracy继续保留，不能用定向复测覆盖首次失败事实。

该历史定向结果中的 `business_use` 提到了运动营养品品牌推广，但输入没有品牌合作、购买入口、促销或带货证据，因此它是后续质量问题的真实来源。当前代码已增加高风险商业用途证据防护：遇到这种情况会把用途降级为内容归档、检索和人工复核，并在 `quality_flags` 中记录 `business_use_grounded_fallback`。历史批次不会被改写；程序强制降级分支由离线测试覆盖，下面的新批次验证真实模型正常路径。

完成证据约束后，使用同一输入生成了定向批次 `output/batch_business_use_guard_20260722_222907/`。本批次只包含1个文本文件、1次DeepSeek调用和0个错误；主分类仍为体育健康，业务用途为内容归档、检索和人工复核，成本0.00178元，延迟5711ms。`quality_flags` 为空，表示模型这次直接遵守了提示词，没有触发客户端强制降级。两个批次应并列保留，用来说明“问题如何被真实发现、代码如何加固、真实正常路径如何复测”，不能删除旧结果后只展示修正结果。

字段说明：

| 字段 | 含义与作用 |
|---|---|
| `business_use` | 模型给出的业务用途说明，应只包含输入证据直接支持的动作 |
| `quality_flags` | 机器可读的质量风险标签；`business_use_grounded_fallback` 表示无证据商业用途已被保守降级 |

## 8. 图片 OCR 关键帧评估批次

```text
output/batch_paddleocr_keyframes_20260724_retry/
```

该批次只处理 `img_7.jpg`、`img_8.jpg`、`img_9.jpg` 三张图片。三张图来自真实视频关键帧信息图，使用本地 PaddleOCR 提取文字；文本分析和视觉理解仍保持 mock。

评估汇总见：

```text
output/batch_paddleocr_keyframes_20260724_retry/image_ocr_eval_summary.md
output/batch_paddleocr_keyframes_20260724_retry/image_ocr_eval_summary.json
output/batch_paddleocr_keyframes_20260724_retry/image_ocr_error_analysis_img_9.md
output/batch_paddleocr_keyframes_20260724_retry/image_ocr_error_analysis_img_9.json
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

本批次的关键结果：

| 指标 | 数值 | 说明 |
|---|---:|---|
| 评估文件数 | 3 | 只包含 `img_7.jpg`、`img_8.jpg`、`img_9.jpg` |
| 人工业务文字块 | 123 | 来自 `evaluation/image_ocr_gold.csv` |
| 完整识别文字块 | 96 | OCR结果中完整命中的业务文字块 |
| 完整段落召回率 | 78.05% | 完整识别文字块数 / 人工业务文字块数 |
| 字符错误率 | 11.01% | 分段编辑距离 / 人工正确字符数 |
| OCR外部API成本 | 0.000000元 | PaddleOCR本地运行，不调用外部API |
| OCR平均延迟 | 18006ms | 三张图本地OCR调用平均耗时 |
| OCR P95延迟 | 28261ms | 受最慢图片 `img_9.jpg` 影响 |

该批次说明图片 OCR 链路已经有真实质量证据，但批次级闸门仍未通过：三张关键帧整体完整段落召回率78.05%、字符错误率11.01%、P95延迟28261ms，均未达到当前MVP观察阈值。其中 `img_9.jpg` 的完整段落召回率只有47.62%，错误主要集中在 `pipeline_module`、`buffer_size` 和 `tlb_size` 等小字号结构图文字；三张图虽然 `img_7.jpg` 和 `img_8.jpg` 的质量指标较好，但延迟也超过2秒目标。

`image_ocr_gate_report_keyframes.md` 的读法：

- 先看 `gate_decision`：这是批次级闸门判断，用来决定是否能离开图片OCR功能进入下一功能；
- 再看 `blocking_files`：这是阻塞闸门通过的文件列表，用来区分质量阻塞和延迟阻塞；
- 再看逐图检查：`img_9.jpg` 是质量和延迟双阻塞，`img_7.jpg` 与 `img_8.jpg` 主要是延迟阻塞；
- 最后看建议：预处理实验和延迟拆分已经完成，下一轮应基于质量、延迟和本地算力边界做 OCR 方案取舍判断，而不是直接扩展新功能。

`image_ocr_preprocess_experiment_img_9.md` 的读法：

- 原图基线：`img_9.jpg` 完整段落召回率47.62%、字符错误率20.27%、OCR延迟28261ms；
- 整图放大2倍：完整段落召回率提升到52.38%，但字符错误率仍为20.27%，OCR延迟增加到64146ms；
- 左右分区放大2倍：完整段落召回率提升到50.00%，字符错误率仍为20.27%，OCR延迟为32421ms；
- 结论：预处理方向有轻微召回提升，但没有达到90%召回率、5%字符错误率和2秒延迟目标，不能作为已通过能力写入主流程。

`image_ocr_latency_profile_img_9.md` 的读法：

- 先看 `engine_create_ms`：本地OCR引擎创建和模型加载耗时8834ms，用于判断冷启动或初始化开销；
- 再看 `predict_ms`：首次图片推理60373ms，热启动第二次图片推理56042ms，说明主要瓶颈在本地CPU模型推理；
- 再看 `decode_ms` 和 `parse_ms`：图片解码15ms/10ms，结果解析0ms，说明慢点不在文件读取或后处理；
- 结论：即使不计引擎创建，单图热启动OCR仍远高于2秒目标；下一轮不应扩大到ASR、视觉理解或视频真实处理，应先判断是否接受本地CPU OCR边界，或后续评估更轻量/服务化OCR方案。

`ocr_backend_advice.md` 的读法：

- 先看 `switch_signal`：当前为 `evaluate_alternative_backends`，表示已有PaddleOCR证据不足以直接放行，需要评估替代OCR；
- 再看 `recommended_next_backend_id`：当前为 `cloud_ocr_service`，表示 RapidOCR 已实测未过闸门后，只有在用户授权外部API时才继续做服务化OCR小样本评估；
- 再看 `evaluation_order`：它是测试顺序，不是已接入能力，也不是最终模型选型结论；
- 最后看 `candidate_evaluations`：这里记录 RapidOCR 候选已经真实本地运行，状态为 `not_passed`，因此不应接入主流程。

`rapidocr_candidate_eval.md` 的读法：

- 先看依赖状态：当前为 `available`，表示本机 RapidOCR 相关依赖已经可用，本轮确实运行了本地候选OCR；
- 再看闸门判断：当前为 `not_passed`，表示三张关键帧样本没有达到当前召回率、字符错误率和P95延迟目标；
- 再看批次指标：完整段落召回率82.93%、字符错误率10.64%、P95延迟4294ms，外部API成本0元；
- 再看逐图结果：`img_7.jpg` 为100.00%召回、0.00%字符错误率，`img_8.jpg` 为95.65%召回、3.32%字符错误率，`img_9.jpg` 只有54.76%召回、20.08%字符错误率，说明弱样本仍未解决；
- 下一步不是接入 RapidOCR，而是把它作为已实测未通过候选记录下来。

字段说明：

| 字段 | 含义与作用 |
|---|---|
| `backend_id` | OCR候选后端的唯一标识，用来区分当前后端和待评估后端 |
| `dependency` | 本地依赖状态，用来判断候选OCR本轮是否真实运行 |
| `switch_signal` | 是否需要从当前PaddleOCR转向替代方案评估的判断信号 |
| `evaluation_order` | 下一步建议评估的OCR候选顺序，只表示测试优先级，不表示已接入 |
| `engine_create_ms` | 本地OCR引擎创建耗时，用于观察模型加载和初始化开销 |
| `decode_ms` | 图片读取和解码耗时，用于判断是否慢在文件读取或图像解码 |
| `predict_ms` | OCR模型推理耗时，用于判断核心瓶颈是否在模型识别 |
| `parse_ms` | PaddleOCR结果解析耗时，用于判断后处理是否形成明显开销 |
| `attempt_total_ms` | 单次图片解码、模型推理和结果解析的合计耗时，不包含引擎创建 |

## 9. 失败 / 部分成功演示怎么读

离线故障演示批次：

```text
output/batch_failure_demo_20260721_190052
```

这个批次不触发真实 API，使用显式故障注入模拟三类情况：

| 文件 | 注入故障 | 状态 | 解读 |
|---|---|---|---|
| `img.png` | OCR 失败 | `partial_success` | 图片文字证据缺失，但视觉理解仍成功，文本分析继续产出结果 |
| `ai_content_sample.txt` | 文本分析失败 | `failed` | 关键文本分析失败，无法产出主分类、关键词和摘要 |
| `例子.mp4` | 语音识别失败 | `partial_success` | 音频转写缺失，但 OCR 和视觉理解仍成功，文本分析继续产出结果 |

新增的低质量 OCR 闸门处理的是另一类情况：OCR 模型调用本身成功，但返回文字疑似乱码或过度碎片化。此时 `model_calls.jsonl` 中 OCR 调用仍是 `success`，因为模型没有报错；但 `results.jsonl` 中该文件会进入 `partial_success`，并写入 `quality_flags=low_quality_ocr_text` 和 `warning_messages`。这能避免把“有一段OCR文字”直接误当成“有可靠文字证据”。

对应的受控批次是 `output/batch_controlled_paddleocr_gate_20260729/`。该批次只用于验证低质量 OCR 结果闸门是否生效，不代表图片理解能力已经完成，也不代表视频 OCR 已经接入真实模型。

字段说明：

| 字段 | 含义与作用 |
|---|---|
| `partial_success` | 部分成功状态，表示最终结果已经生成，但证据不完整 |
| `failed` | 失败状态，表示关键步骤失败，无法产出有效最终结果 |
| `missing_evidence` | 缺失证据，用于解释部分成功或失败原因 |
| `quality_flags` | 机器可读质量风险标签，用于标记低质量OCR、用途降级等可统计问题 |
| `warning_messages` | 风险提示，用于说明缺失证据会怎样影响结果可信度 |
| `error_message` | 技术错误信息，用于定位失败环节 |

## 10. 真实与 mock 边界

当前真实模型结果主要来自 DeepSeek 文本分析。

真实部分包括：

- 基于文本证据生成主分类、副分类、标签、摘要和业务用途。
- 返回输入 token 和输出 token 用量，并用于估算真实调用成本。
- 记录真实调用延迟。
- 文本主分类评估可以读取真实 DeepSeek 文本分析结果，并与人工标准答案比较。
- 图片文件在显式选择 `paddleocr` 后，可以使用本地 PaddleOCR 生成真实 `ocr_text` 并计算分段质量指标。
- RapidOCR 已作为本地OCR候选在三张关键帧上完成对照评估，但未接入主流水线。
- 图片视觉理解已实现 Qwen-VL 受保护 API 入口，并已生成 `output/batch_qwen_vl_response_model_check/` 单图真实批次结果；视频 V1 也已生成一次 Qwen-VL 5关键帧真实试跑，结果是4帧成功、1帧网络失败。当前代码已补充关键帧级受控重试，但尚需复测，不证明多图或视频稳定质量。

当前 mock 或占位部分包括：

| 任务 | 当前状态 | 说明 |
|---|---|---|
| 图片 OCR | 默认 mock / 可选本地 PaddleOCR | 默认返回模拟文字；显式选择 `paddleocr` 后已有五张正式图片真实评估；RapidOCR只做候选评估，未接入主流程 |
| 视频 OCR | 默认 mock / 可选本地 PaddleOCR | 当前代码可把视频 V1 抽出的最多 5 张关键帧交给 PaddleOCR；历史视频 V0 批次仍是 mock 单帧证据，尚未形成视频关键帧真实质量评估 |
| 图片视觉理解 | 默认 mock / 可选 Qwen-VL API | 默认返回模拟视觉描述；显式选择 `qwen_vl` 且授权后才会调用 Qwen-VL，当前已有 `img_1.png` 单图真实验证，但尚未形成多图质量评估 |
| 视频视觉理解 | 默认 mock / 可选 Qwen-VL API | 当前代码可把视频 V1 抽出的最多 5 张关键帧交给 Qwen-VL；需要 `--allow-live-api`；支持关键帧级受控重试；已有一次真实试跑但仍需复测可靠性 |
| 语音识别 | mock | 本地音频文件存在时返回模拟音频转写，不是真实音频识别 |
| 视频预处理 | 已实现 V1 | 可读取视频元信息并等距抽取最多 5 张关键帧；本机有 ffmpeg 时可抽取 wav 音频，缺少依赖或提取失败时会记录明确状态 |
| 多供应商动态路由 | 未实现 | 目前按任务类型固定路由，不按成本、延迟或质量动态选择 |
## 成本字段读取口径补充

`batch_report.json` 的 `cost_stats` 中，`total_cost_cny` 表示真实 API 的本地价格表估算成本，用来判断理论预算消耗，不等同供应商后台真实扣费；`recorded_total_cost_cny` 表示所有模型调用记录中的成本合计，可能包含 mock 占位成本；`live_api_cost_cny` 表示真实 API 估算成本；`local_model_cost_cny` 表示本地模型成本；`mock_cost_cny` 表示 mock 占位成本，不能当成供应商真实扣费。

如果一个视频批次同时使用 Qwen-VL、PaddleOCR、mock ASR 和 mock 文本分析，应优先看 `live_api_cost_cny` 和 `cost_by_provider` 判断真实 API 估算成本；`mock_cost_cny` 只用于说明当前流程里还有占位环节。若供应商用免费额度抵扣，后台真实扣费可能为 0，需要成本对账报告单独记录。
## `results_readable.md` 证据块说明

新生成的 `results_readable.md` 会在文件级摘要后展开关键证据原文：`raw_text` 表示文本原文；`ocr_text` 表示 OCR 识别文字；`visual_description` 表示视觉理解模型生成的画面描述；`audio_transcript` 表示音频转写文字。这样人工评判视频结果时，可以直接对照逐帧 OCR 和逐帧视觉描述，不必先打开 `results.jsonl`。

注意：如果某个字段来自 mock，例如当前视频批次中的 `audio_transcript`，它只能说明流程占位，不代表真实音频识别能力。
## 视频分类人工评判基准

四视频批次的视觉证据可以在 `output/batch_video_qwen_vl_4videos_review/results_readable.md` 中人工检查。当前已把用户确认的三个分类边界样本记录到 `evaluation/video_topic_gold.csv`：`例子.mp4` 的正确主分类是 `other`；`例子2.mp4` 的正确主分类是 `technology` 且不应加入 `entertainment` 副分类；`例子3.mp4` 的正确主分类是 `finance_business` 且不应加入 `technology` 副分类。

注意：该批次的 `topic` 和 `secondary_topics` 来自 mock 文本分析，不能作为真实分类质量结论。后续如果要评估分类质量，应基于已有 OCR 和 Qwen-VL 证据单独授权真实文本分析回归。

## 只重跑 DeepSeek 文本分析层

如果已经有一批视频结果包含 OCR 和 Qwen-VL 视觉理解证据，但最终分类来自 mock 文本分析，可以不重新跑 OCR、Qwen-VL 或视频预处理，只复用已有证据重跑 DeepSeek 文本分析层。

推荐命令：

```powershell
.\.venv\Scripts\python.exe .\src\reanalyze_batch_text.py --source-batch-dir .\output\batch_video_qwen_vl_4videos_review --batch-id batch_video_deepseek_text_reanalysis_review --allow-live-api --max-api-retries 1
```

如果只想补跑某个失败文件，可以加 `--include-files`：

```powershell
.\.venv\Scripts\python.exe .\src\reanalyze_batch_text.py --source-batch-dir .\output\batch_video_qwen_vl_4videos_review --batch-id batch_video_deepseek_text_reanalysis_file0001_retry --include-files 例子.mp4 --allow-live-api --max-api-retries 1
```

这条命令会读取源批次的 `results.jsonl`。其中 `results.jsonl` 是文件级结果文件，用于保存每个文件的分类、摘要、证据和状态；`ocr_text` 是历史 OCR 文字证据；`visual_description` 是历史视觉理解证据；`audio_transcript` 是音频转写证据，本入口会主动丢弃 mock 音频转写，不把它当成真实证据。新批次会写入 `source_batch_id`，用于说明本次重分析复用了哪个历史批次。

新输出批次会生成：

| 文件 | 含义与作用 |
|---|---|
| `results_readable.md` | 人工可读结果，用于检查 DeepSeek 重分析后的 `topic`、`secondary_topics`、`tags`、`summary` 和证据边界 |
| `results.jsonl` | 机器可读文件级结果，用于后续评估或导入 |
| `model_calls.jsonl` | 本轮 DeepSeek 文本分析调用记录；不包含历史 OCR 或 Qwen-VL 调用 |
| `batch_report.json` | 本轮 DeepSeek 文本分析的成本、延迟和成功率汇总 |
| `errors.jsonl` | 本轮失败记录；无失败时为空 |

字段说明：`batch_id` 是新重分析批次编号；`source_batch_id` 是被复用的历史批次编号；`topic` 是主分类；`secondary_topics` 是副分类；`evidence_used` 是实际用于文本分析的证据列表；`missing_evidence` 是缺失证据列表；`processing_status` 是文件级处理状态。由于真实 ASR 尚未接入，视频重分析结果即使 DeepSeek 成功，也可能保持 `partial_success`，表示结果基于 OCR 和视觉理解证据，但缺少真实音频证据。

如果启用 `--max-api-retries 1` 后发生第一次失败、第二次成功，`model_calls.jsonl` 会保留两条调用记录，`call_ids` 也会同时关联两次尝试，便于追踪重试成本和延迟。
