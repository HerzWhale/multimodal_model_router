# Multimodal Model Router

面向内容平台 AI 团队技术负责人的多模态批处理与模型路由 MVP。

系统目标不是做一个单点内容分类器，而是把文本、图片、视频文件统一接入，拆成可追踪的处理任务，并记录模型调用、成本、延迟、证据来源和错误状态。它要解决的是 AI 团队在批量处理内容素材时常见的工程问题：输入不统一、输出不统一、模型调用过程不可追踪、成本和延迟难以复盘。

当前版本必须诚实说明边界：DeepSeek 文本分析已经完成真实调用验证；图片 OCR 已接入本地 PaddleOCR，并用五张正式图片完成本地 CPU 推理与分段评估，其中 `img_7.jpg`、`img_8.jpg`、`img_9.jpg` 来自真实视频关键帧信息图。图片视觉理解、语音识别以及视频上游证据提取仍是 mock 或占位逻辑，因此这次验证只能证明真实图片文字提取链路和 OCR 质量评估链路可运行，不能证明完整多模态理解质量。

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
| 图片处理流程 | 已实现工程链路和本地 OCR 接口 | OCR 默认 mock，可显式选择 PaddleOCR；已完成五张正式图片的本地推理与质量评估，视觉理解仍为 mock |
| 视频处理流程 | 已实现工程链路 | 视频预处理、OCR、视觉理解、语音识别为 mock 或占位 |
| DeepSeek 文本分析 | 已真实接入 | 响应校验和业务用途证据约束均完成单样本真实验证；强制降级分支由离线测试覆盖 |
| 模型调用记录 | 已实现 | 记录供应商、模型、任务类型、用量、成本、延迟和状态 |
| 批次报告 | 已实现 | 汇总文件数、成功率、成本、延迟和质量风险 |
| 模型组合策略报告 | 已实现离线分析 | 基于已有批次解释成本、延迟、真实 / mock 边界和模型组合建议 |
| 路由策略模拟 | 已实现离线分析 | 模拟成本优先、延迟优先、质量优先和平衡策略 |
| 路由策略预检查 | 已实现离线检查 | 批处理前读取路由、价格、输入规模和历史调用记录，检查路由完整性、预算、P95延迟、真实模型覆盖率和mock边界；当预算通过但延迟失败时，会给出受控小样本试跑建议；不自动改路由，不触发模型调用 |
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
| `model_name` | 具体模型名称，用于追踪结果由哪个模型产生 |
| `cost_cny` | 单次模型调用成本，单位人民币，用于成本核算 |
| `latency_ms` | 单次模型调用耗时，单位毫秒，用于延迟分析 |
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

`--ocr-backend paddleocr` 当前只作用于图片文件，不需要 API 密钥或 `--allow-live-api`。视频关键帧预处理仍是占位，因此视频 OCR 会继续保留为 mock，不能写成本地 PaddleOCR 真实推理。

显式使用 PaddleOCR 时，主流程会对非空 `ocr_text` 做一个保守质量闸门：如果文本高度碎片化或疑似乱码，文件不会被当成完全成功，而会写入 `quality_flags=low_quality_ocr_text` 和对应 `warning_messages`，并把文件级 `processing_status` 调整为 `partial_success`。这类情况不代表 OCR 调用失败，所以 `model_calls.jsonl` 中该 OCR 调用仍会保留为 `success`，但下游文本分析不会把这段 OCR 文字当成可靠证据。

如果只想处理指定图片，可以使用 `--include-files`：

```powershell
python .\src\main.py --input-dir .\input\sample_images --include-files img_7.jpg,img_8.jpg,img_9.jpg --ocr-backend paddleocr --text-analysis-backend mock --batch-id batch_paddleocr_keyframes_20260724_retry
```

`--include-files` 按文件名筛选本次输入，作用是避免为了评估少数样本而把整个输入目录都跑一遍。由于 Paddle 底层推理器对中文路径支持不稳定，本轮真实 OCR 使用临时英文盘符映射后成功运行；如果直接用包含中文的项目绝对路径运行，可能出现本地模型创建失败。

为避免显式运行 PaddleOCR 时意外调用配置文件中的 DeepSeek，命令行只选择 PaddleOCR 时，未显式选择的文本分析后端会强制保持 mock；启用 DeepSeek 时，未显式选择的 OCR 后端也会强制保持 mock。

批处理前可以先运行路由策略预检查：

```powershell
python .\src\routing_preflight.py --input-dir .\input --ocr-backend paddleocr --text-analysis-backend deepseek --budget-limit-cny 50 --expected-audio-seconds-per-video 60 --historical-model-calls ".\output\batch_20260718_150348\model_calls.jsonl,.\output\batch_paddleocr_keyframes_20260724_retry\model_calls.jsonl,.\output\batch_text_eval_20260722_135443\model_calls.jsonl" --output-dir .\output\routing_preflight_current
```

这条命令只读取本地配置、输入目录和已有历史调用记录，不运行 DeepSeek、不运行 PaddleOCR，也不下载模型权重。它会生成 `routing_preflight_report.json` 和 `routing_preflight_report.md`，用于说明当前路由是否完整、输入规模会触发多少任务单位、哪些任务仍是 mock、预算和 P95 延迟是否能在运行前判断。

当前 `output/routing_preflight_current/` 的报告基于 12 个输入样本生成：1 个文本、10 张图片和 1 个视频。按每个视频 3 帧、60 秒音频、每个文件 300 个输出 token 的运行前假设，预估总成本为 0.171279 元，50 元预算约束通过；读取三个已有历史批次后，最大任务级 P95 延迟为 28261ms，超过 balanced 策略 3500ms 目标，因此当前预检查状态为 `fail`，不建议直接扩大运行。

报告会同时生成 `controlled_trial_plan`：这是受控小样本试跑建议，用来说明在“预算没问题、延迟失败”的情况下下一轮应如何缩小范围。当前建议是先使用 `--include-files ai_content_sample.txt,img.png,img_1.png` 做最多3个文件的小批量试跑，暂不纳入视频；先跑纯 mock 流程，再单独跑本地 PaddleOCR，DeepSeek 文本试跑必须另行带 `--allow-live-api` 授权。这个建议只写入报告，不会自动执行命令。

默认不会自动重试。只有在已明确授权真实API调用的基础上，再显式增加以下参数，才允许对可重试错误最多重试1次：

```powershell
--max-api-retries 1
```

每次重试都会生成独立模型调用记录并分别计入成本和延迟。鉴权失败、权限错误和参数错误不会重试。不要在未确认输入范围和费用前启用该参数。

默认情况下，主流程读取 `input/`，也就是普通业务输入样例。`evaluation/` 是评估样本目录，不会被默认批处理自动读取。

如果要用评估样本跑一次安全的离线流程，可以显式指定输入目录和 mock 后端：

```powershell
python .\src\main.py --input-dir evaluation\text_topic_small_set --text-analysis-backend mock --batch-id batch_eval_mock
```

如果要重新生成真实评估结果，必须同时使用 `--text-analysis-backend deepseek` 和 `--allow-live-api`。这一步需要 API Key、会访问外部服务并产生费用。

运行参数说明：

| 参数 / 配置 | 含义与作用 |
|---|---|
| `ocr_backend` | 图片 OCR 后端配置；默认使用 mock，显式设为 `paddleocr` 时才运行本地 PaddleOCR |
| `text_analysis_backend` | 文本分析后端配置，用来决定使用 mock 还是 DeepSeek |
| `--allow-live-api` | DeepSeek API 调用授权开关，用来防止外部请求和费用被配置文件或默认命令误触发 |
| `--max-api-retries` | 可重试API错误的最大重试次数；默认0，只能显式设为1 |
| `--include-files` | 指定本次只处理哪些文件名，用于受控评估少量图片，避免误处理整个输入目录 |
| `--historical-model-calls` | 指定一个或多个历史 `model_calls.jsonl` 路径，用于运行前自动提取任务级 P95 延迟 |
| `input_dir` | 本次批处理读取的输入目录，用来隔离普通业务输入和评估样本 |
| `batch_id` | 批次唯一编号，用来定位一次运行生成的结果和模型调用记录 |

运行离线测试：

```powershell
python -m unittest discover -s tests
```

说明：默认离线测试会替换 PaddleOCR 引擎，不下载模型权重、不进行真实图片推理，也不会触发 DeepSeek API。

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
| `batch_metadata.json` | 记录批次编号、创建时间、预算上限和输出格式 |
| `results.jsonl` | 保存每个文件的最终结构化结果；新生成文件遵循标准 JSONL，每行是一条完整文件记录 |
| `results_readable.md` | 保存人工可读结果，便于检查每个文件的分类、摘要、证据、成本和耗时 |
| `model_calls.jsonl` | 保存每次模型调用的任务类型、供应商、模型名、用量、成本、延迟和状态；每行是一条完整调用记录 |
| `errors.jsonl` | 保存失败或部分成功时的问题；每行是一条完整错误记录，便于程序解析和排查 |
| `batch_report.json` | 汇总整个批次的文件数、成功率、成本、延迟和质量风险 |
| `model_strategy_report.md` / `model_strategy_report.json` | 基于已有批次生成模型组合策略分析 |
| `routing_policy_simulation.md` / `routing_policy_simulation.json` | 基于已有批次模拟不同路由策略下的取舍 |
| `routing_preflight_report.md` / `routing_preflight_report.json` | 批处理前检查当前输入规模画像、历史延迟画像、路由、预算约束、P95延迟约束、真实模型覆盖率和mock边界 |
| `text_topic_eval_report.md` / `text_topic_eval_report.json` | 保存文本主分类评估结果 |
| `image_ocr_eval_summary.md` / `image_ocr_eval_summary.json` | 保存图片 OCR 分段质量汇总，包括完整段落召回率、字符错误率和 OCR 延迟 |
| `image_ocr_error_analysis_img_9.md` / `image_ocr_error_analysis_img_9.json` | 保存 `img_9.jpg` 的 OCR 错误归因和闸门判断 |
| `image_ocr_gate_report_keyframes.md` / `image_ocr_gate_report_keyframes.json` | 保存三张关键帧图片的批次级 OCR 闸门判断，用于决定是否继续留在 OCR 功能内 |
| `image_ocr_preprocess_experiment_img_9.md` / `image_ocr_preprocess_experiment_img_9.json` | 保存 `img_9.jpg` 的 OCR 预处理实验结果，用于判断整图放大或分区放大是否值得继续 |
| `image_ocr_latency_profile_img_9.md` / `image_ocr_latency_profile_img_9.json` | 保存 `img_9.jpg` 的 OCR 延迟拆分结果，用于区分引擎创建、图片解码、模型推理和结果解析耗时 |
| `ocr_backend_advice.md` / `ocr_backend_advice.json` | 保存 OCR 后端取舍判断，用于决定是否继续本地PaddleOCR，还是下一轮评估本地ONNXRuntime或服务化OCR |
| `rapidocr_candidate_eval.md` / `rapidocr_candidate_eval.json` | 保存 RapidOCR 候选评估结果；当前已完成三张关键帧本地实测，整体完整段落召回率82.93%、字符错误率10.64%、P95延迟4294ms，闸门结论为未通过 |
| `failure_demo_interpretation.md` | 解释失败 / 部分成功演示批次中的错误链路和证据缺失 |

机器读取应使用三个 `.jsonl` 文件；人工逐项检查应使用 `results_readable.md`。从本版本开始，新批次严格采用“一条记录占一个物理行”的标准 JSONL。已有历史批次不会被改写，项目内部读取器继续兼容历史缩进式连续 JSON 对象。

## 7. 成本与延迟统计

成本与延迟来自两层记录：

- `model_calls.jsonl`：单次模型调用记录。
- `batch_report.json`：批次级聚合报告。

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
- DeepSeek API：当前真实文本分析后端。
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
- 图片 OCR 默认仍为 mock，只有显式选择时才使用本地 PaddleOCR；视觉理解、语音识别和视频 OCR 仍是 mock 或占位。
- 图片最终结构化结果仍混合真实 `ocr_text` 与 mock `visual_description`，因此不能作为完整图片理解质量结论。
- Paddle 底层推理器对中文路径仍不稳定；直接使用 H 盘中文路径时模型创建失败，改用临时英文盘符映射和 `PADDLE_PDX_CACHE_HOME` 后成功运行。代码尚未自动处理该环境问题。
- `img_1.png` 的真实 OCR 耗时15733ms；`img_2.png` 独立冷启动批次耗时51096ms；三张关键帧图片的 OCR 平均延迟为18006ms、P95延迟为28261ms。当前 OCR 延迟仍高于既定图片2秒目标，本地算力、电力和运维成本尚未计量。
- 批次级 OCR 闸门报告显示，三张关键帧图片整体完整段落召回率78.05%、字符错误率11.01%、P95延迟28261ms，均未达到当前MVP观察阈值；其中 `img_9.jpg` 是主要质量阻塞样本，三张图都存在延迟阻塞。
- `img_9.jpg` 的预处理最小实验已完成：整图放大2倍把完整段落召回率从47.62%提升到52.38%，左右分区放大2倍提升到50.00%，但字符错误率仍为20.27%，延迟也分别达到64146ms和32421ms，因此预处理方向有轻微价值但不能通过闸门。
- `img_9.jpg` 的延迟拆分已完成：引擎创建耗时8834ms，首次图片推理60373ms，热启动第二次图片推理56042ms，图片解码和结果解析都不是主要瓶颈；即使不计引擎创建，本地CPU单图推理仍远高于2秒目标。
- RapidOCR 候选评估已经完成：三张关键帧整体完整段落召回率82.93%、字符错误率10.64%、P95延迟4294ms，虽然明显快于当前 PaddleOCR CPU 批次，但仍未达到90%召回、5%字符错误率和2秒P95延迟目标，因此不接入主流程。
- OCR 后端取舍判断已经更新：当前信号仍为 `evaluate_alternative_backends`，但 RapidOCR 已被标记为 `evaluated_not_passed`；如继续追求生产可用 OCR，只能在单独授权后小样本评估服务化 OCR，否则保留 PaddleOCR 作为当前本地基线。
- 运行时模型路由仍按任务类型固定选择模型；离线路由策略模拟和路由策略预检查已经实现，但它们都不会自动改写运行时调度。预检查中的受控试跑建议只给出缩小范围和安全命令，不等于动态路由。
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
| 已完成 | 补运行前规模画像 | 已基于 `input/` 生成预计文件数、图片数、视频数、音频秒数和token用量；预算可运行前估算 |
| 已完成 | 补历史 P95 延迟输入 | 已从已有 `model_calls.jsonl` 自动生成任务级历史延迟画像；当前因 OCR 历史 P95 28261ms 超过3500ms目标，预检查状态为fail |
| 已完成 | 补受控小样本试跑建议 | 当前预算约束通过但延迟约束失败时，报告会建议最多3个文件、暂不纳入视频、拆开 mock / PaddleOCR / DeepSeek 三条试跑线 |
| P1 | 判断是否授权服务化OCR小样本评估 | 如果继续突破OCR质量和延迟闸门，需要先确认API Key、费用、网络和数据风险 |
| P2 | 后续再整理展示材料 | 等核心功能更扎实后，再考虑是否补充图示、可视说明和对外说明 |
