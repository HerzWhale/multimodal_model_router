# 文件级分析结果解读

批次编号：batch_paddleocr_smoke_20260723

说明：本文件用于人工阅读；机器读取请使用 `results.jsonl`。

---

## 结果 1：file_0001 | img.png | image

- 处理状态：partial_success
- 主分类：other
- 副分类：无
- 关键词：内容分析
- 摘要：基于现有文本证据，该内容主要涉及：模拟视觉描述：img.png 展示了一段待分析内容。
- 业务用途：可用于内容归档、素材检索和结构化结果验证。
- 使用证据：visual_description
- 缺失证据：ocr_text
- 使用模型：ocr: paddlepaddle/PP-OCRv5_mobile（failed, file_0001_call_0001）；visual_understanding: qwen/mock-vision（success, file_0001_call_0002）；text_analysis: deepseek/mock-text（success, file_0001_call_0003）
- 关联模型调用：file_0001_call_0001、file_0001_call_0002、file_0001_call_0003
- 文件处理成本：0.010026 元
- 文件处理耗时：15298 ms
- 错误信息：PaddleOCR 本地推理失败：[json.exception.parse_error.101] parse error at line 1, column 1: attempting to parse an empty input; check that your input string or stream contains the expected JSON
- 风险提示：OCR 分支失败，最终分析未使用图片文字证据。

---

## 结果 2：file_0002 | img_1.png | image

- 处理状态：partial_success
- 主分类：other
- 副分类：无
- 关键词：内容分析
- 摘要：基于现有文本证据，该内容主要涉及：模拟视觉描述：img_1.png 展示了一段待分析内容。
- 业务用途：可用于内容归档、素材检索和结构化结果验证。
- 使用证据：visual_description
- 缺失证据：ocr_text
- 使用模型：ocr: paddlepaddle/PP-OCRv5_mobile（failed, file_0002_call_0001）；visual_understanding: qwen/mock-vision（success, file_0002_call_0002）；text_analysis: deepseek/mock-text（success, file_0002_call_0003）
- 关联模型调用：file_0002_call_0001、file_0002_call_0002、file_0002_call_0003
- 文件处理成本：0.010028 元
- 文件处理耗时：8 ms
- 错误信息：PaddleOCR 本地推理失败：[json.exception.parse_error.101] parse error at line 1, column 1: attempting to parse an empty input; check that your input string or stream contains the expected JSON
- 风险提示：OCR 分支失败，最终分析未使用图片文字证据。
