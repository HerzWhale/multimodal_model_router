# OCR 后端取舍判断报告

## 一、当前结论

- 当前后端：`paddleocr_local`
- 切换判断：`evaluate_alternative_backends`
- 下一优先评估后端：`cloud_ocr_service`
- 下一步：RapidOCR已实测未通过当前闸门；不要接入主流程。如继续追求生产可用OCR，下一轮只能在用户授权后小样本评估 `cloud_ocr_service`；如暂不授权外部API，则保留PaddleOCR作为当前本地基线。

## 二、当前PaddleOCR证据

| 指标 | 数值 | 作用 |
|---|---:|---|
| 完整段落召回率 | 78.05% | 判断OCR是否漏掉关键业务文字 |
| 字符错误率 | 11.01% | 判断OCR错字、漏字和多字程度 |
| OCR P95延迟 | 28261ms | 判断批次慢调用是否超过目标 |
| 质量是否失败 | True | 判断是否需要寻找识别更稳的方案 |
| 延迟是否失败 | True | 判断是否需要寻找更快的方案 |
| 延迟瓶颈 | `first_predict` | 判断优化方向 |

当前阈值：完整段落召回率 ≥ 90.00%，字符错误率 ≤ 5.00%，OCR延迟 ≤ 2000ms。

## 三、判断理由

- 当前PaddleOCR关键帧批次质量闸门未通过，需要评估替代方案。
- 当前PaddleOCR关键帧批次延迟闸门未通过，需要评估更快的本地或服务化方案。
- 延迟拆分显示主要瓶颈在模型推理，单纯优化文件读取或结果写入意义不大。
- RapidOCR候选后端已完成同批样本实测，但质量和延迟仍未通过当前闸门，不应接入主流程。

## 四、候选后端排序

| 后端ID | 部署类型 | 当前状态 | 优先级分数 | 下一步测试 |
|---|---|---|---:|---|
| cloud_ocr_service | cloud_api | candidate_after_local_eval | 90 | 如果用户授权，再选择一家服务做3张关键帧小样本live test，并记录成本、延迟和质量。 |
| tesseract_local | local | reference_baseline | 35 | 只选一张弱样本 `img_9.jpg` 做最小对照，不先纳入主流程。 |
| paddleocr_local | local | current_backend | 25 | 仅在接受当前质量与延迟边界时继续保留；否则作为基线对照。 |
| rapidocr_onnxruntime_local | local | evaluated_not_passed | 15 | 候选后端已实测未通过当前闸门，不应接入主流程。 |

## 五、已评估候选后端

| 后端ID | 依赖状态 | 闸门状态 | 完整段落召回率 | 字符错误率 | OCR P95延迟 | 外部API成本 |
|---|---|---|---:|---:|---:|---:|
| rapidocr_onnxruntime_local | available | not_passed | 82.93% | 10.64% | 4294ms | 0.0000元 |

## 六、信息来源

- `cloud_ocr_service`：https://cloud.baidu.com/product/ocr
- `tesseract_local`：https://github.com/tesseract-ocr/tesseract
- `paddleocr_local`：https://github.com/PaddlePaddle/PaddleOCR
- `rapidocr_onnxruntime_local`：https://github.com/RapidAI/RapidOCR

## 七、边界说明

- 本报告只基于已有PaddleOCR评估证据、已有候选评估报告和公开候选方案信息生成取舍建议。
- 已评估候选只代表当前样本上的本地对照结果，不等于已接入主流程。
- 如果后续要验证服务化OCR，必须先确认API Key、费用、网络和数据合规风险。

## 八、字段说明

| 字段 | 含义与作用 |
|---|---|
| `backend_id` | OCR候选后端的唯一标识，用来区分当前后端和待评估后端 |
| `switch_signal` | 是否需要从当前PaddleOCR转向替代方案评估的判断信号 |
| `evaluation_order` | 下一步建议评估的OCR候选顺序，只表示测试优先级，不表示已接入 |
| `quality_failed` | 质量是否未达当前闸门，用来判断是否需要寻找识别质量更稳的方案 |
| `latency_failed` | 延迟是否未达当前闸门，用来判断是否需要寻找更快或服务化的方案 |
| `latency_bottleneck` | 当前延迟瓶颈位置，用来判断优化方向是模型推理、解码、解析还是引擎创建 |
| `candidate_evaluations` | 候选OCR后端的已评估结果摘要，用来避免重复推荐已经实测未通过的后端 |
