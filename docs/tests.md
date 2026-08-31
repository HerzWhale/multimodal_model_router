# Tests

这份文档说明当前项目已经有哪些测试、如何运行、覆盖了哪些模块，以及哪些风险还没有被测试覆盖。它的目标不是把项目包装成“测试完备”，而是让后续维护者知道当前验证边界。

## 1. 测试目标

当前测试目标是验证本地 MVP 的核心离线流程：

- 文件能被识别成文本、图片或视频。
- 预处理模块能为不同文件类型生成后续流水线所需输入。
- 视频 V1 预处理能读取视频元信息、等距抽取最多 5 张关键帧、记录关键帧元数据和预处理产物，并在本机存在 ffmpeg 时抽取 wav 音频文件；缺少依赖或提取失败时会记录明确状态。
- 模型路由能按任务类型选择供应商和模型。
- mock 模型能返回符合系统要求的结构。
- DeepSeek 客户端在缺少 API Key 时能明确失败，不会静默产出假结果。
- Qwen-VL 图片或视频关键帧视觉理解入口能在缺少 API Key 或授权开关时停止，离线测试只验证请求构造、响应解析、调用记录和失败降级。
- DashScope ASR 入口能在缺少授权、本地音频上传失败或响应结构异常时停止；离线测试只验证本地上传调用、请求构造、响应解析和流水线记录，不调用真实 ASR。
- 成本、延迟、结果写入和批次报告能按预期生成。
- 成本对账能基于已有 `model_calls.jsonl` 生成多供应商手工账单模板，在未填账单金额时明确保持未验证状态，并拒绝非法金额和重复账单记录；已覆盖免费额度 0 元扣费时的周期级对账报告展示。
- 决策层能基于已有批次数据生成成本、延迟、真实/mock 边界和模型组合建议。
- 路由策略模拟能基于已有批次检查预算、延迟、真实覆盖率等约束。
- 路由策略预检查能在批处理前检查输入规模画像、历史延迟画像、任务级P95延迟目标、延迟阻塞归因、路由完整性、预算约束、真实模型覆盖率和mock边界。
- 受保护离线回归检查入口能在临时目录中验证 mock 批处理、视频 V1 预处理边界和 routing preflight 核心路径，不触发真实模型。
- 文本主分类评估能生成评估模板、合并人工标准答案，并计算 Accuracy、Macro-F1 和分类级指标。
- 主入口能在临时目录中跑通一次 mock 批处理。
- 默认运行保持离线；PaddleOCR 单元测试使用替代引擎，不下载权重，DeepSeek、Qwen-VL 和 DashScope ASR 真实调用必须经过显式授权。
- 延迟文本分析模式会保留证据、写出 `pending` 状态且不调用文本 API；文本重分析成功后会清除延迟标记。
- 延迟模式在没有任何可用证据时会直接失败，不会制造无法完成的待处理记录。
- Phase 2 门禁能检查同步后端对照、两阶段调用隔离、源批次关联和重复成功调用。

## 2. 当前已有测试数量

最近一次完整离线运行结果为：

```text
Ran 349 tests

OK
```

测试命令：

```powershell
python -m unittest discover -s tests
```

说明：这条命令只运行本地离线单元测试，不会触发 DeepSeek、Qwen-VL 或 DashScope ASR API。

受保护离线回归检查入口：

```powershell
python .\src\offline_regression_check.py
```

该命令会运行完整离线测试，并在临时目录中验证三文件 mock 批处理、视频 V1 预处理边界和 routing preflight 报告生成。它不会调用 DeepSeek，不会调用 Qwen-VL，不会调用 DashScope ASR，不会运行真实 PaddleOCR，不会调用云OCR，也不会向正式 `output/` 目录写入新批次。

## 3. 覆盖的模块

| 测试文件 | 覆盖模块 | 已验证内容 |
|---|---|---|
| `test_file_loader.py` | `file_loader.py` | 文件类型识别、文件清单生成、文件级编号生成 |
| `test_preprocessor.py` | `preprocessor.py` | 文本读取、图片路径处理、视频 V1 元信息读取、多关键帧等距抽取、OpenCV 中文路径写入兜底、ffmpeg 音频提取成功路径和缺少依赖状态输出 |
| `test_model_router.py` | `model_router.py` | 历史规则兼容、路由计划构造、媒体 pipeline 解析、真实 API 标记、fail 拒绝、配置指纹漂移拒绝、精简决策快照校验和 warning 伪装为 pass 的拒绝 |
| `test_model_clients.py` | `model_clients.py` | mock客户端、PaddleOCR结果解析、Qwen-VL请求构造和响应解析、DashScope本地音频上传、DashScope ASR请求构造和转写结果解析、DeepSeek结构化响应和有限重试 |
| `test_cost_latency_tracker.py` | `cost_latency_tracker.py` | 模型价格读取、价格目录元数据读取、按输入输出用量计算成本、模型调用记录生成 |
| `test_price_catalog_updater.py` | `price_catalog_updater.py` | 官方价格页解析、候选价格报告生成、价格刷新预检查、`--apply` 写回本地价格目录、刷新报告写出；测试使用假网页，不访问网络 |
| `test_cost_repricing.py` | `cost_repricing.py` | 历史批次成本重算、缺价格处理、批次文件读取、JSON/Markdown报告写出、字段说明渲染 |
| `test_pipeline_runner.py` | `pipeline_runner.py` | 三条文件级流水线、路由计划在任务选择层覆盖旧参数、图片本地 OCR、Qwen-VL关键帧、调用记录、上游失败、低质量OCR闸门、视频多关键帧和 DashScope ASR 分支 |
| `test_result_writer.py` | `result_writer.py` | JSON、标准单行 JSONL、空记录文件、Markdown、人类可读结果、服务端响应模型名展示、视频预处理产物展示、模型调用明细和错误文件写入 |
| `test_report_generator.py` | `report_generator.py` | 批次成本、延迟、成功率和错误统计；验证标准单行 JSONL 与缩进式连续 JSON 对象均可读取 |
| `test_model_strategy_advisor.py` | `model_strategy_advisor.py` | 成本汇总、历史成本与当前重算成本口径切换、延迟瓶颈识别、真实/mock 边界识别、字段缺失稳健处理、JSON 和 Markdown 报告生成 |
| `test_routing_policy.py` | `model_catalog.py` / `routing_policy.py` | 模型目录聚合、策略约束判断、真实/mock 覆盖率、预算扩展模拟和配置读取 |
| `test_routing_preflight.py` | `routing_preflight.py` | 运行前输入规模画像、历史延迟画像、预算和P95约束、受控试跑建议、mock边界、配置路由覆盖，以及报告附带路由计划时的独立 JSON 写出 |
| `test_offline_regression_check.py` | `offline_regression_check.py` | 受保护离线回归入口、临时目录mock批处理、临时目录routing preflight冒烟检查、安全边界字段和CLI JSON输出 |
| `test_cost_reconciliation.py` | `cost_reconciliation.py` | 多供应商成本分组、手工账单模板生成、单次调用级对账、时间段级对账、未填账单保持未验证、非法账单金额拒绝、重叠重复账单拒绝、mock和本地模型排除、账单来源与备注进入Markdown报告、JSON/Markdown报告写出 |
| `test_strategy_simulator.py` | `strategy_simulator.py` | 基于已有批次生成路由策略模拟 JSON/Markdown 报告，验证缺字段时不会硬算 |
| `test_text_topic_evaluator.py` | `text_topic_evaluator.py` / `evaluation/` | 端到端 Accuracy、有效预测 Accuracy、预测覆盖率、Macro-F1、分类级指标、缺失/非法预测分离、缺标签、报告写入、18条样本一致性、标签泄漏、长样本长度和人工答案CSV的Excel编码兼容性 |
| `test_image_ocr_evaluator.py` | `image_ocr_evaluator.py` / `evaluation/image_ocr_gold.csv` | 文字规范化、重复文字块不可重复占用、同一OCR行内多个非重叠文字块、最佳连续片段、字符编辑距离、可选文字排除、基准字段校验、JSON报告生成、单图错误归因、批次级闸门判断和报告写出 |
| `test_image_ocr_preprocessing_experiment.py` | `image_ocr_preprocessing_experiment.py` | 预处理图片生成、变体评估、基线对比、实验结论、OCR延迟拆分和JSON/Markdown报告写出；测试使用假OCR引擎，不运行真实PaddleOCR |
| `test_ocr_backend_advisor.py` | `ocr_backend_advisor.py` | OCR后端候选排序、PaddleOCR不过闸门时的替代方案建议、RapidOCR已评估未通过时不重复推荐、隐私约束、缺失字段稳健处理、JSON/Markdown报告写出 |
| `test_rapidocr_candidate_evaluator.py` | `rapidocr_candidate_evaluator.py` | RapidOCR候选依赖检查、依赖缺失安全跳过、假OCR客户端评估、常见返回结构解析、JSON/Markdown报告写出 |
| `test_main.py` | `main.py` | mock批处理、双层settings、路由计划生成与显式消费、外部候选证据接入、warning计划执行、未知候选/fail/配置漂移/参数冲突拒绝、真实API二次授权、批次路由快照，以及原有后端安全闸门 |

Phase 4.2 专项执行 `python -m unittest tests.test_phase2_gate tests.test_model_router tests.test_main`，61项通过；完整离线测试执行 `python -m unittest discover -s tests`，365项通过。测试没有访问模型 API。

Phase 4.3 离线实现专项覆盖 Qwen3.5-OCR 密钥缺失、原图请求、固定模型、合法/非法响应、token 与响应模型记录、图片流水线、视频多关键帧、路由计划、API二次授权和长ASCII碎片洪水拦截。完整离线测试执行 `python -m unittest discover -s tests`，375项通过；本轮测试没有访问真实 API。

## 4. 关键字段与测试作用

| 字段 | 含义与作用 | 当前测试情况 |
|---|---|---|
| `file_id` | 单个输入文件的唯一标识，用来关联结果、模型调用和错误记录 | 已测试文件清单和结果写入中会生成与保留 |
| `batch_id` | 一次批处理任务的唯一标识，用来把多个文件和调用记录归到同一批次 | 已测试主入口和报告生成会保留批次编号 |
| `call_id` | 单次模型调用的唯一标识，用来追踪某个文件触发的具体调用 | 已测试结果写入和流水线会关联调用编号 |
| `task_type` | 模型调用任务类型，用来区分 OCR、视觉理解、语音识别和文本分析 | 已测试路由和流水线按任务类型生成调用链 |
| `provider` | 模型供应商，用来按供应商汇总成本和延迟 | 已测试模型调用记录和报告中的供应商统计 |
| `model_name` | 请求模型名称，用来记录系统向供应商请求调用哪个模型 | 已测试文件级模型摘要会保留模型名称 |
| `response_model_name` | 服务端响应模型名称，用来核对供应商实际返回的模型名或模型别名 | 已测试 Qwen-VL 响应中的模型名会进入调用记录和文件级模型摘要 |
| `selected_backends` | 本次命令或配置选择的后端组合，用来说明 OCR、视觉理解和文本分析分别选择了什么 | 已测试主流程会写入批次元数据 |
| `backend_runtime_summary` | 根据实际模型调用明细汇总出的真实 API、本地模型和 mock 组合 | 已测试能区分 Qwen-VL 真实 API、PaddleOCR 本地模型和 mock 调用 |
| `cost_estimation` | 成本估算口径说明，用来记录价格表来源、计算方法、是否包含 mock 估算、误差状态和是否已对账 | 已测试 mock 批次会标记包含 mock 成本估算；真实账单误差未对账前不做百分比断言 |
| `cost_estimation_method` | 成本估算方法，用于说明本次成本来自系统价格目录而不是供应商实时账单 | 已测试模型调用记录会写入 `price_catalog` |
| `price_source` | 价格来源，用于说明模型单价来自本地手工配置、mock假设还是本地运行时假设 | 已测试配置存在时读取原值，缺失时使用安全默认值 |
| `price_updated_at` | 价格更新时间，用于判断当前价格目录是否可能过期 | 已测试配置存在时读取原值，缺失时为空 |
| `price_confidence` | 价格可信度，用于区分未验证手工价格、mock价格和本地外部API零成本假设 | 已测试配置存在时读取原值，缺失时标记为 unknown |
| `price_fetch` | 价格抓取配置，用于说明该模型是否启用官方公开页刷新 | 已测试启用抓取的模型会进入刷新报告 |
| `source_url` | 官方价格页地址，用于追溯自动抓取来源 | 已测试刷新器按URL读取对应页面 |
| `parser` | 价格解析器名称，用于说明系统按哪套规则解析官方页面 | 已测试 Qwen-VL 与 DeepSeek 的解析器 |
| `preflight_status` | 价格刷新预检查状态，用于判断候选价格是否允许写回本地价格目录 | 已测试正常通过、大幅变化阻断和人工允许后的 warning |
| `preflight_checks` | 价格刷新预检查明细，用于记录必要计价单位、正数价格和大幅变化检查是否通过 | 已测试缺少必要计价单位、非数字、零、负数、NaN 和 Infinity |
| `allow_large_change` | 大幅价格变化写回授权开关，用于要求人工确认超过50%的价格变化后才能写回配置 | 已测试默认阻断大幅变动，显式允许后才可写回 |
| `recorded_cost_cny` | 历史模型调用记录中原本保存的成本估算值 | 已测试会从历史 `cost_cny` 映射而来 |
| `current_estimated_cost_cny` | 按当前价格目录和历史调用用量重新计算出的成本估算值 | 已测试 Qwen-VL 官方价格更新后会得到新的重算成本 |
| `reprice_status` | 成本重算状态，用于区分未变化、已变化和无法重算 | 已测试 changed、unchanged 和 not_repriced |
| `cost_basis` | 策略报告使用的成本口径，用于区分历史记录成本和当前价格目录重算成本 | 已测试默认使用 `historical_recorded`，传入重算报告后使用 `current_repriced` |
| `estimated_cost_cny` | 系统根据模型调用用量和本地价格表计算出的估算成本 | 已测试会按供应商和模型聚合 |
| `billed_cost_cny` | 供应商后台显示或账单导出的实际扣费 | 已测试为空时保持未验证，填入后计算偏差 |
| `cost_delta_cny` | 实际扣费减去系统估算值后的金额差 | 已测试单次调用级对账能生成差额 |
| `cost_delta_rate` | 成本差额相对估算成本的比例 | 已测试估算值存在时计算比例，缺真实账单时保持为空 |
| `billing_granularity` | 账单粒度，用来区分单次调用、小时级、日级或模型级对账 | 已测试单次调用级和时间段级可信度不同 |
| `cost_confidence` | 成本可信度状态，用于区分未验证、单次调用级对账和时间段级对账 | 已测试 `unverified`、`call_level_reconciled` 和 `period_level_reconciled` |
| `bill_source` | 真实扣费来源，用来说明账单金额来自供应商控制台人工查看还是账单导出文件 | 已测试账单记录会保留来源字段 |
| `matching_method` | 系统调用记录与供应商账单的匹配方式，用来说明按什么规则关联账单和调用 | 已测试报告会保留匹配方式字段 |
| `matched_call_ids` | 实际参与对账的模型调用编号列表，用于从对账结果反查调用明细 | 已测试单次调用级对账能关联到具体调用 |
| `unmatched_billing_records` | 没有匹配到本批次模型调用的账单记录，用于暴露账单时间窗口、供应商或模型名称填写错误 | 已测试未匹配记录不会被静默丢弃 |
| `cost_cny` | 单次模型调用成本，单位人民币，用于成本核算 | 已测试按配置价格计算成本 |
| `latency_ms` | 单次模型调用延迟，单位毫秒，用于性能分析 | 已测试调用记录和报告会保留延迟 |
| `processing_status` | 文件级处理状态，用来判断成功、部分成功、失败或跳过 | 已测试报告可统计成功、部分成功和失败状态 |
| `preprocessing_artifacts` | 文件预处理产物摘要，用来记录视频元信息、关键帧抽取状态、音频提取状态和预处理风险 | 已测试视频 V1 结果会写入该字段，并在人类可读结果中展示 |
| `keyframe_paths` | 视频 V1 抽出的关键帧路径列表，用于追踪视频画面证据来自哪些本地产物 | 已测试 OpenCV 可用时会写出多张等距关键帧路径；中文路径直写失败时会用编码写入兜底 |
| `keyframe_metadata` | 每张关键帧的帧号、时间位置和路径，用于说明视频画面证据覆盖哪些时间点 | 已测试多关键帧元数据会写入预处理结果和人工可读输出 |
| `audio_extraction_status` | 音频提取状态，用于说明当前是否已经生成真实音频文件；`extracted` 表示已生成音频，`dependency_missing` 表示本机缺少 ffmpeg，失败状态用于排查提取问题 | 已测试 ffmpeg 成功提取、缺少依赖、音频存在时进入 mock 语音识别、音频缺失时不虚算音频秒数 |
| `business_use` | 业务用途说明，用来解释结果可以支持什么业务动作 | 已测试无商业证据的推广、广告或转化建议会降级，明确商业证据和普通用途不会被误改 |
| `quality_flags` | 机器可读的质量风险标签，用来批量筛选和追溯质量事件 | 已测试用途降级会记录 `business_use_grounded_fallback`；低质量OCR会记录 `low_quality_ocr_text` 并使文件进入部分成功 |
| `model_call_count` | 模型调用次数，用来衡量本批次实际触发了多少次模型任务 | 已测试策略报告会基于模型调用明细计算调用次数 |
| `is_mock` | 是否为 mock 调用，用来区分真实模型调用和占位调用 | 已测试策略报告能识别真实 DeepSeek 调用与 mock 上游调用 |
| `cost_share` | 成本占比，用来判断某个任务或模型是否是主要成本来源 | 已测试策略报告会计算任务和模型维度的成本占比 |
| `policy_name` | 路由策略名称，用来区分成本优先、延迟优先、质量优先和平衡策略 | 已测试不同策略会生成不同约束判断和建议 |
| `constraint_status` | 策略约束满足状态，用来判断当前批次是否符合某类业务目标 | 已测试通过、失败和部分未知状态 |
| `real_coverage_rate` | 真实模型调用占全部模型调用的比例，用来衡量真实 API 证据覆盖程度 | 已测试真实调用和 mock 调用能被区分并计算比例 |
| `preflight_status` | 运行前预检查总状态，用来判断当前路由配置是否可以继续受控试跑 | 已测试 pass/warning/fail 相关路径中的 warning 和 fail |
| `price_catalog_profile` | 价格目录画像，用来检查本次路由涉及模型的价格来源、更新时间和可信度 | 已测试新鲜官方价格通过、过期价格变为 warning、未验证价格可信度会进入 warning |
| `max_price_age_days` | 价格过期阈值，用来判断价格目录超过多少天后需要刷新 | 已测试超过 7 天的价格会被标记为过期 |
| `price_freshness_status` | 价格新鲜度状态，用来区分 fresh、stale、missing_updated_at、invalid_updated_at 和 future_updated_at | 已测试 fresh 和 stale 路径 |
| `price_confidence_status` | 价格可信度状态，用来区分 trusted、explainable、unknown 和 unverified | 已测试官方公开价格为 trusted，未验证手工价格为 unverified |
| `workload_profile` | 运行前规模画像，用于统计输入文件数、媒体类型分布和预计任务单位 | 已测试能从输入目录生成画像，并按指定文件过滤范围 |
| `latency_profile` | 历史延迟画像，用于从已有模型调用记录中汇总任务级平均延迟、P95延迟和最大延迟 | 已测试能从 `model_calls.jsonl` 生成画像，并拆分真实API、本地运行和mock延迟口径 |
| `latency_bottleneck_analysis` | 延迟阻塞归因，用于把慢因拆成真实外部API、本地运行和mock占位三类 | 已测试本地PaddleOCR慢、真实DeepSeek API慢和mock延迟不可用能被分开输出 |
| `real_api_slow_tasks` | 真实外部API慢任务列表，用于判断哪些真实网络调用超过当前P95目标 | 已测试超过阈值的真实文本分析API会进入该列表 |
| `local_runtime_slow_tasks` | 本地运行慢任务列表，用于判断哪些耗时来自本机推理链路 | 已测试超过阈值的本地PaddleOCR会进入该列表 |
| `mock_latency_unusable_tasks` | mock延迟不可用任务列表，用于提醒这些延迟不能作为真实供应商性能证据 | 已测试mock视觉理解会进入该列表 |
| `overall_status` | 受保护离线回归检查的总状态，用来判断全部核心检查是否通过 | 已测试全部步骤通过时返回 `pass` |
| `boundary` | 回归检查的安全边界说明，用来确认是否调用真实API、真实PaddleOCR或正式output | 已测试默认不调用真实模型，且只写临时目录 |
| `steps` | 回归检查的逐项结果列表，用来定位mock批处理或routing preflight哪一步失败 | 已测试包含 `mock_batch_smoke` 和 `routing_preflight_smoke` |
| `expected_units_by_task` | 各任务的预估计量单位，用来把单位价格转换成整批预算估算 | 已测试完整用量可计算总成本，视频缺少音频秒数时不会硬算语音识别成本；也已测试无视频批次不会把 `speech_to_text` 纳入成本 unknown，普通短文本/图片批次不会默认纳入 `summary_merge` |
| `historical_p95_latency_by_task_ms` | 按任务类型整理的历史P95延迟，用于把已有运行经验带入运行前延迟预检查 | 已测试可触发延迟约束失败 |
| `budget_limit_cny` | 预算上限，用来判断预估用量下的模型组合是否可能超预算 | 已测试提供预估用量时能计算成本并触发预算失败 |
| `p95_latency_limit_ms` | P95延迟限制，用来判断高分位延迟是否超过业务目标 | 已测试提供历史P95延迟时能触发延迟失败 |
| `task_latency_targets_ms` | 按任务类型配置的P95延迟目标，用来避免OCR、文本分析、视觉理解共用同一个不合理阈值 | 已测试任务级目标可以覆盖全局阈值，且非法配置会被拒绝 |
| `task_latency_target_checks` | 任务级延迟目标检查明细，用来逐项记录观察P95、目标P95、目标来源、证据口径和通过状态 | 已测试通过、失败、mock证据warning和从配置文件读取路径 |
| `min_real_coverage_rate` | 最低真实模型覆盖率，用来约束mock任务占比不能过高 | 已测试全mock路由会触发真实覆盖率失败 |
| `predicted_topic` | 模型预测的文本主分类，用来和人工答案对比 | 已测试能从结果文件中提取并写入评估模板 |
| `gold_topic` | 人工标注的正确主分类，是计算准确率的基准 | 已测试能从人工标准答案表读取和合并 |
| `accuracy` | 文本主分类准确率，计算方式为 `correct_count / evaluated_count` | 已测试正确样本数、缺标签样本和准确率计算 |
| `valid_prediction_accuracy` | 有效分类预测中的准确率，用于隔离分类判断能力 | 已测试缺少预测时不会把调用失败混成分类错误类型 |
| `prediction_coverage` | 有效分类预测占已评估样本的比例，用于观察调用和解析稳定性 | 已测试缺少预测和非法预测会降低覆盖率 |
| `macro_f1` | 各参与评估分类 F1 的简单平均，用来避免总体正确率掩盖小类别问题 | 已测试类别不均衡时能与 Accuracy 产生不同结果 |
| `precision` | 预测为某分类的样本中真正属于该分类的比例，用于观察误报 | 已测试分类级 Precision 计算 |
| `recall` | 人工标注为某分类的样本中被正确识别的比例，用于观察漏报 | 已测试分类级 Recall 计算 |
| `f1` | 单个分类 Precision 与 Recall 的调和平均，用于综合评价误报和漏报 | 已测试分类级 F1 与 Macro-F1 聚合 |
| `support` | 人工标准答案中属于某分类的样本数，用于判断分类证据量 | 已测试分类级样本数统计 |
| `input_dir` | 本次批处理读取的输入目录，用来区分默认业务输入和评估样本输入 | 已测试命令行显式指定评估目录时只处理评估样本 |
| `--include-files` | 本次只处理的文件名列表，用于受控评估少量图片，避免误跑整个输入目录 | 已测试只处理指定文件，且指定不存在文件时停止 |
| `pipelines` | 分析主体配置，用来分别选择文本、图片、视频和音频链路的任务后端 | 已测试嵌套配置生效，文本与图片可在同一批次使用不同后端 |
| `backends` | 模型主体配置，用来集中读取供应商、模型名、接口和生成限制 | 已测试 DeepSeek 与 Qwen-VL 嵌套参数会传入对应客户端，并兼容旧扁平配置 |
| `selected_pipelines` | 批次实际媒体链路快照，用来复核各媒体类型采用的后端组合 | 已测试由运行时按输入媒体类型生成并写入批次元数据 |
| `visual_description` | 图片或视频关键帧视觉理解输出的画面描述，用于补充 OCR 无法覆盖的视觉证据 | 已测试 Qwen-VL 成功时写入结果和证据链，失败时进入部分成功 |
| `segment_id` | 图片内文字块的唯一编号，用于逐段关联人工正确文本和OCR结果 | 已测试同图重复编号会被拒绝 |
| `exact_segment_recall` | 完整识别的必选业务文字块占比 | 已测试重复文字必须匹配不同OCR行 |
| `character_error_rate` | 分段编辑距离总和除以人工正确字符总数 | 已测试缺字、相邻噪声和可选文字排除 |
| `error_bucket` | OCR错误归因类型，用来区分标签丢失、数值丢失、片段截断或字符替换 | 已测试错误类型聚合和报告写入 |
| `error_by_segment_type` | 按人工文字块类型汇总错误，用来判断问题集中在哪类内容 | 已测试能按文字块类型聚合错误段和编辑距离 |
| `gate_decision` | 基于当前MVP阈值生成的OCR功能闸门判断，用来决定是否继续留在当前功能内 | 已测试低召回、高错误率和高延迟会阻止进入下个功能，并能生成批次级报告 |
| `variant_name` | OCR预处理方案名称，用来区分整图放大、分区放大等实验输入 | 已测试不同变体会分别生成图片并独立评估 |
| `decision` | OCR预处理实验结论，用来判断该预处理方向是否值得继续 | 已测试变体优于基线、通过闸门和未通过闸门的判断逻辑 |
| `engine_create_ms` | 本地OCR引擎创建耗时，用来观察模型加载和初始化开销 | 已测试延迟拆分报告会记录引擎创建耗时 |
| `decode_ms` | 图片读取和解码耗时，用来判断是否慢在文件读取或图像解码 | 已测试每次OCR尝试都会记录解码耗时 |
| `predict_ms` | OCR模型推理耗时，用来判断核心瓶颈是否在模型识别 | 已测试延迟拆分会识别主要耗时阶段 |
| `parse_ms` | PaddleOCR结果解析耗时，用来判断后处理是否形成明显开销 | 已测试每次OCR尝试都会记录解析耗时 |
| `attempt_total_ms` | 单次图片解码、模型推理和结果解析的合计耗时，不包含引擎创建 | 已测试报告会生成单次合计耗时和Markdown表格 |
| `backend_id` | OCR候选后端的唯一标识，用来区分当前后端和待评估后端 | 已测试候选目录能区分当前PaddleOCR、RapidOCR、Tesseract和云OCR |
| `dependency` | 本地依赖状态，用来判断候选OCR本轮能否真实运行 | 已测试依赖缺失时会输出 `dependency_missing`，不会编造质量和延迟指标；当前真实RapidOCR报告显示依赖可用 |
| `switch_signal` | 是否需要从当前PaddleOCR转向替代方案评估的判断信号 | 已测试PaddleOCR质量和延迟不过闸门时会生成替代评估信号 |
| `evaluation_order` | 下一步建议评估的OCR候选顺序，只表示测试优先级，不表示已接入 | 已测试隐私约束会把云OCR放到本地候选之后，也已测试RapidOCR未通过后不会被重复推荐 |
| `candidate_evaluations` | 候选OCR后端的已评估结果摘要，用来避免重复推荐已经实测未通过的后端 | 已测试RapidOCR候选报告可被OCR后端建议器读取，并更新候选状态 |
| `text_analysis_backend` | 文本分析后端配置，用来决定使用 mock 还是 DeepSeek | 已测试默认值为 mock，且 DeepSeek 未授权时会被拒绝 |
| `--allow-live-api` | 真实 API 调用授权开关，用来防止误触发外部请求和费用 | 已测试必须与显式 DeepSeek 或 Qwen-VL 后端同时使用 |
| `--max-api-retries` | 可重试错误的最大重试次数；默认0，显式设为1才允许一次重试 | 已测试不能超过1，不能用于mock后端，并能分别传递给 DeepSeek 或 Qwen-VL |

## 5. 未覆盖的风险

当前测试仍有以下缺口：

- 已实现真实 API 调用安全闸门、响应错误分类和显式单次重试。原失败样本已完成一次真实定向验证并在首次请求成功；真实重试分支没有被自然触发，仍由离线故障测试覆盖。
- 路由策略预检查已经能识别配置风险，并可基于输入目录生成运行前规模画像和预算估算，也可从已有 `model_calls.jsonl` 生成任务级历史P95延迟画像和延迟阻塞归因；但这些历史延迟来自既有小样本批次，不能等同于下一批真实生产延迟。
- PaddlePaddle 3.3.0 和 PaddleOCR 已安装在项目虚拟环境，五张正式图片已完成本地 CPU 推理与分段评估；但仍未系统测量CPU、内存和批量吞吐。
- `img_1.png` 的真实 OCR 调用耗时15733ms；`img_2.png` 独立冷启动批次耗时51096ms；三张关键帧图片的 OCR 平均延迟为18006ms、P95延迟为28261ms。当前仍未满足既定图片2秒目标。
- Windows中文路径和默认oneDNN分别触发过模型加载与运行时兼容问题；代码已关闭MKLDNN并支持中文输入图片路径，但 Paddle 底层推理器对中文模型缓存路径仍不稳定。本轮通过临时英文盘符映射和 `PADDLE_PDX_CACHE_HOME` 成功运行，代码尚未自动处理该环境问题。
- 本轮三张关键帧图片整体完整段落召回率为78.05%、字符错误率为11.01%；其中 `img_9.jpg` 完整段落召回率只有47.62%。错误归因显示问题主要集中在小字号结构图中的 `pipeline_module`、`buffer_size` 和 `tlb_size`。批次级闸门报告进一步确认：质量阻塞集中在 `img_9.jpg`，延迟阻塞覆盖三张关键帧图片。
- `img_9.jpg` 已完成一次真实PaddleOCR预处理最小实验。整图放大2倍和左右分区放大2倍均成功运行，但只带来轻微召回提升，字符错误率未下降，延迟更高；因此不能把预处理写成已解决方案。
- `img_9.jpg` 已完成一次真实PaddleOCR延迟拆分。引擎创建8834ms，首次模型推理60373ms，热启动第二次模型推理56042ms；图片解码和结果解析不是主要瓶颈。该结果说明本地CPU单图OCR延迟边界仍未达标，但不代表GPU或服务化OCR一定同样慢。
- 已新增OCR后端取舍判断测试。该测试只验证基于已有证据和候选评估报告生成候选排序和边界说明，不会安装RapidOCR、Tesseract，也不会调用云OCR。
- RapidOCR候选评估器已经实现并完成一次本地真实对照：三张关键帧整体完整段落召回率82.93%、字符错误率10.64%、P95延迟4294ms，外部API成本0元；默认离线测试仍使用假OCR客户端，不会真实运行RapidOCR。
- 图片和视频关键帧视觉理解已实现 Qwen-VL 受保护 API 入口，并完成离线请求构造、响应解析、失败降级和关键帧级重试记录测试；当前已有 `img_1.png` 单图真实批次和一次视频5关键帧真实试跑，可证明真实调用链路、成本和延迟记录可用，但仍不能证明多图或视频关键帧稳定视觉理解质量。
- 已有离线单元测试证明视频 V1 多张关键帧可以分别进入 PaddleOCR 或 Qwen-VL 调用分支，并把多帧文字和画面描述汇总给文本分析；本地音频提取已覆盖 ffmpeg 成功和缺依赖路径，但真实语音识别仍是占位。
- 没有大批量 500 文件压力测试，因此还不能证明大规模处理性能。
- 已有离线失败 / 部分成功演示批次；但还没有真实供应商故障样例，因此不能把该批次解释为真实故障率。
- 决策层测试目前验证的是离线报告逻辑，不能证明真实多供应商模型组合效果。
- 文本主分类评估测试验证的是评估器逻辑，包括标签合并、Accuracy、Macro-F1 和分类级指标计算；它不等于证明模型线上质量。
- 当前评估集有18条清理后样本，测试会检查文件与人工标签一致、模型可见输入不包含分类答案提示，以及4条新增长难例每条不少于约800个中文字符。
- 九类规则补齐后已完成一次18条受控 DeepSeek 回归；端到端 Accuracy 为94.44%、有效预测 Accuracy 为100.00%、预测覆盖率为94.44%、Macro-F1 为96.30%。该结果是实验产物，不由离线单元测试伪造或保证。
- 高风险商业用途证据约束已重新调用真实DeepSeek验证原样本，本次没有再生成无证据商业建议；模型主动返回保守用途，所以程序强制降级分支仍由离线测试覆盖。
- 当前每类只有2条人工样本，且回归复用了参与规则诊断的已知样本；`other` 类本轮 Recall 为100%，但不能据此推断新样本或线上流量的稳定表现。
- 回归批次有1条响应无法解析为JSON；离线测试已经保证评估器不会把缺失预测误当成新的业务分类，但尚未解决供应商响应解析的端到端稳定性。
- 没有自动化 CI 配置，因此测试尚未成为提交前自动门禁。
- 没有跨平台路径测试，尤其是中文路径和 Windows/Linux 路径差异。

## 6. live test 与离线 test 的区别

| 类型 | 含义 | 是否默认运行 | 原因 |
|---|---|---|---|
| 离线 test | 只使用本地 mock、替代 OCR 引擎、临时目录和固定输入 | 是 | 稳定、低成本、不下载权重、不依赖 API Key |
| 本地模型验证 | 使用真实 PaddleOCR 权重处理真实图片 | 否 | 首次可能下载权重，并消耗本机 CPU、内存和磁盘 |
| live test | 调用真实 DeepSeek、Qwen-VL 或其他供应商 API | 否 | 会消耗费用，可能受网络、额度、供应商响应变化影响 |

## 7. 为什么默认不运行真实 API 测试

默认不运行真实 DeepSeek 和 Qwen-VL API 测试有三个原因：

- 成本控制：项目预算有限，自动测试不应在每次运行时消耗 API 费用。
- 稳定性：外部 API 可能因为网络、额度、供应商状态而波动，不适合作为默认单元测试。
- 密钥安全：默认测试不应该要求本地存在真实 API Key，也不能把 API Key 写入代码或提交记录。

更合适的方式是后续增加受保护的 live test：只有在用户明确设置环境变量并手动开启时，才运行真实 API 验证。

PaddleOCR 不使用 API Key。它必须显式选择 `paddleocr` 后端；程序会在生成批次输出前检查 PaddlePaddle 和 PaddleOCR 是否已安装。

## 8. 后续测试计划

### 文本双后端离线测试

当前离线测试覆盖 Qwen 文本后端在无 API Key 时请求前停止、固定请求模型、关闭思考模式、合法 JSON 与 token 用量解析、非法 JSON 拒绝、供应商返回模型记录，以及文本重分析只复用历史证据。第一轮真实对照另行验证了3次调用链路和P95延迟，但Qwen出现一次副分类边界错误，因此不能将离线测试或真实小样本解释为质量门槛通过。

| 优先级 | 测试计划 | 类型 | 目的 |
|---|---|---|---|
| P0 | 保留DeepSeek响应与重试回归 | 离线单元测试 | 后续修改客户端时防止错误分类、重试边界和调用计量回退 |
| P1 | 把故障注入接入受保护演示命令 | 离线集成测试 | 让失败和部分成功样例更容易复现，同时避免默认流程误触发 |
| P0 | 增加 `.gitignore` 和 Demo 文件保留检查 | 手动检查 / 版本管理检查 | 避免纳入缓存、密钥或误删保留证据批次 |
| P0 | 增加策略报告回归样例 | 离线测试 | 保证 `model_strategy_report.md` 和 `model_strategy_report.json` 的关键结论不随意漂移 |
| 已完成 | 增加路由策略预检查测试 | 离线单元测试 | 保证运行前能检查路由完整性、预算、P95延迟、真实覆盖率和mock边界 |
| 已完成 | 增加延迟阻塞归因测试 | 离线单元测试 | 保证路由预检查能区分本地PaddleOCR慢、真实API慢和mock延迟不可用于供应商判断 |
| 已完成 | 增加任务级延迟目标测试 | 离线单元测试 | 保证OCR、视觉理解和文本分析可以使用不同P95目标，避免单一全局阈值误判 |
| P1 | 增加 DeepSeek live test 开关 | 受保护 live test | 手动验证真实 API 响应结构、成本和延迟记录 |
| P1 | 增加更大样本批处理测试 | 离线集成测试 | 验证批量处理、报告生成和性能边界 |
| P2 | 增加自动化 CI | CI 测试 | 让离线测试成为版本提交前门禁 |
| 已完成 | 分析图片 OCR 弱样本与延迟瓶颈 | 离线报告复核 | 已基于 `img_9.jpg` 生成错误归因和闸门判断 |
| 已完成 | 生成关键帧 OCR 批次级闸门报告 | 离线报告复核 | 已基于现有评估汇总和错误归因，判断当前批次是否可以进入下一功能 |
| 已完成 | 评估是否做 OCR 预处理实验 | 受控本地模型测试 / 离线报告复核 | 已完成整图放大和左右分区放大实验；方向有轻微收益但未过闸门 |
| 已完成 | 拆分 OCR 延迟来源 | 受控本地模型测试 / 离线报告复核 | 已区分引擎创建、图片解码、模型推理和结果解析；确认当前主要慢在本地CPU模型推理 |
| 已完成 | 做 OCR 方案取舍判断 | 离线报告复核 | 已基于PaddleOCR证据和RapidOCR实测结果，明确RapidOCR不接入主流程，服务化OCR需要单独授权 |
| 已完成 | 准备 RapidOCR 候选评估器 | 离线单元测试 / 依赖缺失报告 | 依赖缺失时明确跳过，依赖安装后可复用同一批图片和人工基准 |
| 已完成 | 安装并真实评估 RapidOCR 候选 | 受控本地模型测试 / 离线报告复核 | 三张关键帧同批样本已跑通，本地0元外部API成本，但质量和延迟闸门未通过 |
| 已完成 | 补低质量OCR结果闸门 | 离线单元测试 / 受控批次复核 | 已验证明显碎片化OCR文本会写入 `quality_flags` 和 `warning_messages`，且不会进入下游文本分析证据；受控批次保存在 `output/batch_controlled_paddleocr_gate_20260729/` |
| P0 | 判断是否授权服务化OCR小样本评估 | 受保护 live test / 离线报告复核 | RapidOCR已实测未过闸门；如继续追求生产可用OCR，需要先确认API Key、费用、网络和数据风险 |
| 已完成 | 增加 Qwen-VL 图片/关键帧视觉理解受保护入口 | 离线单元测试 | 默认不触发 API；已验证请求构造、响应解析、调用记录和失败降级 |
| 已完成 | 增加 Qwen-VL 关键帧级重试与失败补偿测试 | 离线单元测试 | 已验证远端断开连接可重试一次、鉴权错误不重试、视频关键帧重试成功会记录失败和成功两次尝试、重试仍失败会保留部分成功与风险标记 |
| 已完成 | 增加本地音频提取最小闭环测试 | 离线单元测试 | 已验证 ffmpeg 成功提取音频、缺少依赖时记录状态、音频存在时进入 mock 语音识别、音频缺失时仍保持部分成功 |
| P0 | 扩展 Qwen-VL 图片视觉理解小样本批次 | 受保护 live test | 在用户授权后从单图扩展到2到3张图片，继续验证真实 `visual_description`、成本和延迟 |
