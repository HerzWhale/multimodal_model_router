# Phase 3.1 阶段报告

## 阶段目标

建立统一最终结果质量评价与证据可信度门禁。该门禁只读取已有批次输出和人工答案，不调用 DeepSeek、Qwen-VL、DashScope ASR 或 PaddleOCR。

## 用户执行决策

| 问题 | 用户选择 | 工程含义 |
|---|---|---|
| 质量门禁实现范围 | A | 只在现有 `text_topic_evaluator.py` 内补最小闭环，不新建评估框架 |
| 代表批次选择 | A | 只评价人工答案能覆盖的现存代表批次，不扫描全部历史 output |

## 完成任务

1. 在现有评估器中增加视频结果质量门禁。
2. 保留主分类与副分类评估逻辑，并新增证据风险汇总。
3. 修正评估报告中“九类允许值”的旧口径为当前 11 类口径。
4. 生成 DeepSeek 与 Qwen 三视频对照批次的质量门禁报告。
5. 增加离线测试覆盖 pass、warning、fail 三种视频质量门禁状态。
6. 同步 README、架构说明、测试说明、进度和发现记录。

## 代码证据

- `src/text_topic_evaluator.py`：新增 `result_quality_gate_status`、`requires_human_review`、`result_quality_risk_reasons` 和 `quality_gate_summary`。
- `tests/test_text_topic_evaluator.py`：新增视频质量门禁汇总测试和 11 类口径回归测试。

## 输出证据

| 批次 | 报告 | 结论 |
|---|---|---|
| `output/batch_text_compare_deepseek_round1_20260825/` | `video_quality_gate_report.md` / `video_quality_gate_report.json` | 3 条视频主分类和副分类均通过，质量门禁 3 pass |
| `output/batch_text_compare_qwen_round1_20260825/` | `video_quality_gate_report.md` / `video_quality_gate_report.json` | 3 条视频主分类通过，1 条副分类失败，质量门禁 2 pass / 1 fail |

## 测试证据

- 专项命令：`python -m unittest tests.test_text_topic_evaluator`
- 专项结果：23 项通过，0 失败。
- 完整命令：`python -m unittest discover -s tests`
- 完整结果：346 项通过，0 失败。

## 性能结果

本阶段只做离线读取、CSV/JSON/Markdown 生成和指标计算，不触发真实模型调用；性能不作为本阶段门禁。

## 成本结果

本阶段没有真实 API 调用，没有新增模型费用。报告中的历史成本仍来自对应批次的本地价格表估算，不代表供应商后台实际扣费。

## 遗留问题与风险

1. 当前只评价现存三视频对照批次，不能代表全部视频内容分布。
2. 旧的 9 类文本回归批次仍是历史证据；当前运行时分类体系已经扩展到 11 类。
3. 成本真实性对账不是本阶段目标，仍需后续独立处理。
4. 当前 output 中只保留少量代表批次，部分历史文档仍可能提到已清理的旧批次。

## 最终状态

`COMPLETED`

原因：质量门禁代码已实现，专项和完整离线测试通过，两个现存代表视频批次已生成 JSON/Markdown 报告，文档已同步当前 11 类口径和 Phase 2.2 完成状态。本状态只代表 Phase 3.1 的离线质量评价门禁闭环完成，不代表生产级视频质量已经达标。

## 字段说明

| 字段 | 含义与作用 |
|---|---|
| `result_quality_gate_status` | 单条视频质量门禁状态，用于区分 pass、warning 和 fail |
| `requires_human_review` | 是否需要人工复核，用于把分类错误、证据弱或质量风险样本筛出来 |
| `result_quality_risk_reasons` | 质量风险原因，用于解释为什么某条结果需要复核 |
| `quality_gate_summary` | 批次级质量门禁汇总，用于统计通过、警告、失败和需复核数量 |
| `gold_topic` | 人工主分类答案，用于和模型主分类预测对比 |
| `predicted_topic` | 模型主分类预测，用于计算主分类 Accuracy 和 Macro-F1 |
| `gold_secondary_topics` | 人工副分类答案，用于检查模型是否错误添加或漏掉交叉领域 |
| `predicted_secondary_topics` | 模型副分类预测，用于计算副分类完全匹配率 |
| `video_evidence_stability` | 视频证据稳定性，用于判断关键帧和前段证据是否足以支撑分类 |
| `quality_flags` | 机器可读质量风险标签，用于定位 OCR、视觉理解、语音识别或证据稳定性问题 |

# Phase 3.2 阶段执行报告

## 阶段目标

恢复未来批次的标准 JSONL 机器契约，同时保留历史缩进式连续 JSON 对象的读取兼容；人工阅读继续使用 Markdown，不迁移旧输出。

## 用户执行决策

| 问题 | 选择 | 工程影响 |
|---|---|---|
| 修改范围 | A | 只修统一写入函数和直接相关测试，保留历史格式测试样本 |
| 人工阅读入口 | A | 三个 JSONL 文件使用单行记录，字段级阅读由 `results_readable.md` 承担 |

## 完成任务与代码证据

- `src/result_writer.py`：统一写入函数改为每个物理行写一条完整 JSON 记录。
- `tests/test_result_writer.py`：逐行调用 JSON 解析器，验证两条输出对应两个物理行。
- `tests/test_report_generator.py`：继续验证标准单行 JSONL 和历史多行对象均可读取。
- README、架构、Demo 和测试文档统一机器输出与人工输出职责。

## 测试证据

- 专项测试：`python -m unittest tests.test_result_writer tests.test_report_generator`，14 项通过。
- 完整离线测试：`python -m unittest discover -s tests`，349 项通过。
- 代码检查：`git diff --check` 未发现本轮代码空白错误；仅显示 Windows 换行提示。

## 性能与成本

本阶段修改序列化契约，不运行性能基准。没有调用真实 API，新增模型费用为 0 元。

## 风险与边界

- 历史输出不重写，仍可能是多行对象；兼容读取器继续支持该格式。
- 标准 JSONL 不再满足“每个字段单独换行”的人工阅读偏好，该需求由 Markdown 输出满足。
- 本阶段不改变模型调用、分类、成本或路由逻辑。

## 最终状态

`COMPLETED`

Phase 3.2 的代码、测试、验证、文档与风险门禁均已满足。Execution Window 已结束，必须返回 Planning 决定下一阶段。

# Phase 4.1 阶段报告：可执行路由计划闭环

## 阶段目标

以双层 `settings.yaml` 为新运行路径唯一事实来源，让预检查生成可审查路由计划，并由主流程在用户显式传入后执行；真实 API 授权继续独立生效。

## 用户执行决策

| 问题 | 选择 | 工程影响 |
|---|---|---|
| 路由计划传递方式 | B | 路由计划进入 `pipeline_runner.py` 任务选择层，不只在入口转换 |
| warning 计划执行 | A | fail 阻止；warning 经显式传入后允许受控执行 |

## 代码证据

- `src/model_router.py`：路由计划构造、配置指纹、状态与漂移校验、媒体后端解析、调用记录路由解析。
- `src/routing_preflight.py`：双层 settings 路由可直接传入预检查，并独立写出 `route_plan.json`。
- `src/main.py`：显式 `--route-plan`、参数冲突拒绝、真实 API 二次授权和批次计划快照。
- `src/pipeline_runner.py`：在任务选择层消费路由计划，模型客户端保持原接口。

## 测试与验证证据

- 路由专项：114项通过。
- 完整离线测试：359项通过。
- 预检查证据：`output/preflight_phase4_1_route_plan/`，状态 warning。
- 执行证据：`output/batch_phase4_1_route_plan_offline/`，1个图片、3次模型调用、0错误。
- 实际调用组合：PaddleOCR本地模型、mock视觉、mock文本；调用记录与计划一致。

## 性能与成本

- PaddleOCR本地调用约35秒，延续既有本地OCR延迟风险，不属于本阶段修复目标。
- 没有外部 API 调用，没有新增真实供应商费用；mock成本仍只是流程估算。

## 风险与边界

- 当前是显式可执行路由计划，不是自动动态路由。
- 跨媒体同一任务选择多个后端时，现有任务级预检查只展示代表路线并产生 warning；精确媒体路线保留在计划中。
- 不支持在线流量分配、自动故障切换或按单文件实时质量改路由。

## 最终状态

`COMPLETED`

Phase 4.1 核心功能、测试、离线验证、文档和风险门禁均已满足。Execution Window 在 Round 1 结束，必须返回 Planning。

# Phase 4.2 Execution Round 1 中期报告：证据驱动的可执行路由选择

## 阶段目标

把已有 DeepSeek / Qwen 文本质量、延迟和成本证据接入 Phase 4.1 的显式路由计划；没有全部达标候选时，允许生成披露缺口的 warning 推荐，但不把它写成合格方案。

## 实际完成

- 严格候选与 warning 推荐已经分离；当前严格候选为空，warning 推荐为 DeepSeek，未满足文本分析 P95 延迟门槛。
- 外部决策报告与精简计划快照已经打通；只有文本分析后端被候选证据覆盖，其他任务保持固定配置并标记未比较。
- fail、未知后端、报告状态不一致、快照与 pipeline 不一致、warning 伪装为 pass 等情况会在执行前被拒绝。

## 测试与证据

- 专项测试：61项通过。
- 完整离线测试：365项通过。
- 决策证据：`output/phase4_2_text_route_decision.json`。
- 计划证据：`output/preflight_phase4_2_evidence_route_plan/route_plan.json`，状态 warning。
- 本 Round 没有调用真实 API，没有产生新增模型费用。

## 当前状态

`WARNING`

代码、离线测试与离线证据已完成；一张图片的真实计划执行仍待用户明确授权，因此 Phase 4.2 尚未完成，Execution Window 继续到 Round 2。

## 受控真实执行结果

用户随后执行 `output/batch_phase4_2_evidence_route_plan_live/`。1张图片处理成功，PaddleOCR、Qwen-VL、DeepSeek共3次调用全部成功，错误记录为0；实际供应商、请求模型和任务类型与路由计划一致。

文本分析本次延迟2885ms，视觉理解3647ms，本地OCR 17827ms，整文件24509ms。本地价格目录估算总成本0.003435元；该金额没有与供应商账单对账。单次文本延迟不能替代历史P95，因此推荐状态继续为warning。

## Execution Window 最终状态

`COMPLETED`，Phase状态保持`WARNING`。

本阶段核心链路已经真实验证，但没有产生全部门槛通过的文本候选。窗口必须返回Planning，由下一次规划决定进入其他任务候选覆盖还是继续处理文本性能，Execution不能自行进入下一Phase。

# 2026-08-31 Planning Checkpoint：Phase 4.2结束与Phase 4.3启动

Phase 4.2以`WARNING`结束。warning文本推荐已经完成离线决策、路由计划和一次真实执行，核心链路成立；没有候选同时通过质量和历史P95门槛是保留的阶段结论，不再阻塞项目覆盖下一类任务。

用户在Planning检查点选择AAC：Qwen3.5-OCR成为受API授权保护的新默认OCR，PaddleOCR保留为本地基线；图片与视频关键帧共用该后端；必须以三张图片、一段视频关键帧场景及完整质量、延迟、成本、错误和模型记录完成验证。

Planning决策为`NEXT_PHASE`，进入Phase 4.3。当前只是规划完成，Qwen3.5-OCR尚未进入业务代码，也没有进行真实API调用。

# Phase 4.3 Execution Round 1：离线实现报告

Qwen3.5-OCR 已进入现有模型客户端、图片流水线、视频关键帧流水线和双层配置。图片与视频共用同一调用入口；默认发送原图，PaddleOCR 保留为本地基线。缺少真实 API 授权或密钥时会在网络请求前停止。

完整离线测试374项通过，覆盖固定请求模型、请求结构、合法与非法响应、token、响应模型、成本、图片、多关键帧、配置读取、路由计划和API安全边界。本Round没有真实API调用、没有新增费用。

当前Phase仍为`IN_PROGRESS`：三张图片真实质量/延迟证据及一个视频多关键帧证据尚未生成。不能根据离线测试宣称Qwen3.5-OCR优于PaddleOCR或已经通过默认后端验收。

# Phase 4.3 Execution Round 2：三图真实验证

真实批次`output/batch_qwen35_ocr_3images_trial/`完成3次Qwen3.5-OCR调用，全部成功且请求/响应模型一致。OCR平均延迟6039ms、P95为11458ms；真实API估算成本0.004417元，尚未账单对账。

人工基准显示三图加权完整段落召回率70.10%、字符错误率10.64%，未通过现有90%/5%质量阈值。`img_9.jpg`产生大量短ASCII伪字符，旧低质量闸门漏判；本轮已补充该真实失败模式，旧批次复核能只标记`img_9.jpg`，完整离线测试375项通过。

当前状态仍为`IN_PROGRESS`，窗口状态为`IMAGE_GATE_NOT_PASSED`。根据先图片后视频的执行边界，本Round没有继续调用视频。不能把Qwen3.5-OCR设为已验收默认，也不能将估算成本描述为实际扣费。

已执行预定回退：默认OCR恢复为PaddleOCR，Qwen3.5-OCR保留为已评估、可显式选择的云候选。

Execution Window在Round 2提前结束，状态为`COMPLETED`，Phase状态为`WARNING`。原因不是接口接入失败，而是核心图片质量与延迟门禁未通过，且按既定边界不应继续扩大到视频真实调用。必须返回Planning决定结束、重规划或把候选优化转技术债。
