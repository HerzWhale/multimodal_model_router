# 文件级分析结果解读

批次编号：batch_controlled_paddleocr_gate_20260729

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
- 缺失证据：无
- 使用模型：ocr: paddlepaddle/PP-OCRv5_mobile（success, file_0001_call_0001）；visual_understanding: qwen/mock-vision（success, file_0001_call_0002）；text_analysis: deepseek/mock-text（success, file_0001_call_0003）
- 关联模型调用：file_0001_call_0001、file_0001_call_0002、file_0001_call_0003
- 文件处理成本：0.010026 元
- 文件处理耗时：24552 ms
- 错误信息：None
- 风险提示：OCR 返回了非空文字，但文本疑似乱码或过度碎片化，最终分析未把 OCR 文字作为可靠证据。

---

## 结果 2：file_0002 | img_1.png | image

- 处理状态：success
- 主分类：entertainment
- 副分类：无
- 关键词：内容分析
- 摘要：基于现有文本证据，该内容主要涉及：食影双修 抖音号：63873336098 已关注 1125.3万获赞 134.6万粉丝 百科 食饮双修，抖音视频创作者，其IP属地为河北。专注于影视解说视频的创作。截至2024年7月11日，抖音账号 “食影双修” 已发布127个作品，拥有3
- 业务用途：可用于内容归档、素材检索和结构化结果验证。
- 使用证据：ocr_text、visual_description
- 缺失证据：无
- 使用模型：ocr: paddlepaddle/PP-OCRv5_mobile（success, file_0002_call_0001）；visual_understanding: qwen/mock-vision（success, file_0002_call_0002）；text_analysis: deepseek/mock-text（success, file_0002_call_0003）
- 关联模型调用：file_0002_call_0001、file_0002_call_0002、file_0002_call_0003
- 文件处理成本：0.010462 元
- 文件处理耗时：12845 ms
- 错误信息：None
- 风险提示：无

---

## 结果 3：file_0011 | ai_content_sample.txt | text

- 处理状态：success
- 主分类：technology
- 副分类：finance_business、news
- 关键词：AI团队、多模态处理、素材结构化、模型调用、成本核算
- 摘要：基于现有文本证据，该内容主要涉及：某内容平台的 AI 团队正在整理一批历史素材，其中包括产品发布会稿件、短视频字幕、图文运营文案和技术教程。团队希望把这些素材统一转换成结构化数据，方便后续做内容检索、素材归档和模型训练样本筛选。 在现有流程中，不同格式的文件需要人工判断类型
- 业务用途：可用于技术素材归档、模型调用流程验证、内容检索和批次统计分析。
- 使用证据：raw_text
- 缺失证据：无
- 使用模型：text_analysis: deepseek/mock-text（success, file_0011_call_0001）
- 关联模型调用：file_0011_call_0001
- 文件处理成本：0.000376 元
- 文件处理耗时：1 ms
- 错误信息：None
- 风险提示：无
