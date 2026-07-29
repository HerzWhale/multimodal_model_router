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
| `例子.mp4` | 视频 | 展示视频文件进入预处理、OCR、视觉理解、语音识别和文本分析流水线；视频上游为 mock 或占位 |

字段说明：

| 字段 | 含义与作用 |
|---|---|
| `file_name` | 原始文件名，用来识别每条结果对应哪份内容 |
| `media_type` | 文件媒体类型，用来决定进入文本、图片还是视频流程 |
| `source_path` | 原始文件路径，用来追溯输入文件来源 |

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
| `--max-api-retries` | 可重试错误的最大重试次数；默认0，显式设为1才允许一次重试 |
| `--allow-live-api` | 真实 API 调用授权开关，用来防止误触发外部调用和费用 |

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
| `processing_cost_cny` | 文件级处理成本，单位人民币 |
| `processing_time_ms` | 文件级处理耗时，单位毫秒 |

本批次中文本文件的文本分析由 DeepSeek 真实生成。图片和视频结果需要如实按工程链路解读：OCR、视觉理解、语音识别仍是 mock，因此不能把这部分说成真实识别效果。

机器处理时应读取 `results.jsonl`：它保存文件级结构化记录。从当前版本起，新批次中每一条完整记录只占一个物理行；人工查看仍优先使用本节介绍的 `results_readable.md`。本页引用的历史 Demo 批次生成较早，保留了当时的缩进式连续 JSON 对象，项目读取器仍可兼容，但不会为了展示而改写历史证据。

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

`model_calls.jsonl` 是模型调用明细。每一个 JSON 对象表示一次模型调用；新批次中每次调用占一个物理行，可由普通 JSONL 工具逐行解析。

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
python .\src\routing_preflight.py --input-dir .\input --ocr-backend paddleocr --text-analysis-backend deepseek --budget-limit-cny 50 --expected-audio-seconds-per-video 60 --historical-model-calls ".\output\batch_20260718_150348\model_calls.jsonl,.\output\batch_paddleocr_keyframes_20260724_retry\model_calls.jsonl,.\output\batch_text_eval_20260722_135443\model_calls.jsonl" --output-dir .\output\routing_preflight_current
```

本次预检查基于默认输入目录、显式选择的 PaddleOCR 图片OCR后端、DeepSeek 文本分析后端和三个已有历史调用批次生成。它只读取本地文件清单、配置和历史 `model_calls.jsonl`，不触发 PaddleOCR 推理，不触发 DeepSeek API，也不产生新的费用。

当前结果应这样解读：

- `preflight_status` 为 `fail`，表示当前不建议直接扩大运行；
- `workload_profile` 显示本次纳入12个输入样本：1个文本、10张图片、1个视频；
- `latency_profile` 从已有历史调用记录中提取任务级延迟：OCR P95 为28261ms，文本分析 P95 为7112ms；
- `current_route` 显示 OCR、文本分析和汇总任务是非mock路线，视觉理解和语音识别仍是mock；
- `real_coverage_rate` 为 60%，表示5类预期任务中有3类走非mock路线；
- `expected_units_by_task` 基于运行前假设生成：OCR 13张图、视觉理解13帧、语音识别60秒、文本分析输入3989 token和输出3600 token；
- `budget_limit_cny` 本次显式设为50元，按本地价格表估算总成本为0.171279元，因此预算检查通过；
- `p95_latency_limit_ms` 在 balanced 策略中为3500ms；当前最大任务级历史P95为28261ms，因此延迟检查失败；
- `blocking_reasons` 指出阻塞项是 `p95_latency_limit_ms` 未满足；
- `warning_messages` 会提示视觉理解和语音识别仍是mock，不能解释为完整真实多模态能力；
- `controlled_trial_plan` 给出下一轮受控小样本建议：最多3个文件，暂不纳入视频，建议使用 `--include-files ai_content_sample.txt,img.png,img_1.png`；先跑纯 mock，再单独跑本地 PaddleOCR，DeepSeek 文本试跑必须另行授权。

因此，本报告的结论不是“完全不能继续”，而是：当前50元预算没有问题，但延迟约束失败；下一轮应继续小批量、受控范围跑，不能直接扩大到完整输入目录。

字段说明：

| 字段 | 含义与作用 |
|---|---|
| `preflight_status` | 运行前预检查总状态，用来判断当前配置是可继续、存在风险还是不建议直接运行 |
| `workload_profile` | 运行前规模画像，用来统计输入文件数量、媒体类型分布和预估任务单位 |
| `latency_profile` | 历史延迟画像，用来从已有调用记录中汇总任务级平均延迟、P95延迟和最大延迟 |
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

当前 mock 或占位部分包括：

| 任务 | 当前状态 | 说明 |
|---|---|---|
| 图片 OCR | 默认 mock / 可选本地 PaddleOCR | 默认返回模拟文字；显式选择 `paddleocr` 后已有五张正式图片真实评估；RapidOCR只做候选评估，未接入主流程 |
| 视频 OCR | mock | 视频关键帧预处理仍是占位，因此视频 OCR 不能写成本地 PaddleOCR 真实推理 |
| 视觉理解 | mock | 返回模拟视觉描述，不是真实图像理解 |
| 语音识别 | mock | 返回模拟音频转写，不是真实音频识别 |
| 视频预处理 | 占位 | 用于跑通关键帧和音频链路，不代表完整视频理解 |
| 多供应商动态路由 | 未实现 | 目前按任务类型固定路由，不按成本、延迟或质量动态选择 |
