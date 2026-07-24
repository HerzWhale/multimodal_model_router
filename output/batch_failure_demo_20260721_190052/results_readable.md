# 文件级分析结果解读

批次编号：batch_failure_demo_20260721_190052

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
- 使用模型：ocr: doubao/mock-ocr（failed, file_0001_call_0001）；visual_understanding: qwen/mock-vision（success, file_0001_call_0002）；text_analysis: deepseek/mock-text（success, file_0001_call_0003）
- 关联模型调用：file_0001_call_0001、file_0001_call_0002、file_0001_call_0003
- 文件处理成本：0.020026 元
- 文件处理耗时：0 ms
- 错误信息：演示用 OCR 失败：图片文字识别服务超时。
- 风险提示：OCR 分支失败，最终分析未使用图片文字证据。

---

## 结果 2：file_0002 | ai_content_sample.txt | text

- 处理状态：failed
- 主分类：None
- 副分类：无
- 关键词：无
- 摘要：None
- 业务用途：None
- 使用证据：raw_text
- 缺失证据：无
- 使用模型：text_analysis: deepseek/mock-text（failed, file_0002_call_0001）
- 关联模型调用：file_0002_call_0001
- 文件处理成本：0.000376 元
- 文件处理耗时：0 ms
- 错误信息：演示用文本分析失败：模型返回不可解析结果。
- 风险提示：文本分析模型调用失败，无法产出有效分类、标签和摘要。

---

## 结果 3：file_0003 | 例子.mp4 | video

- 处理状态：partial_success
- 主分类：other
- 副分类：无
- 关键词：内容分析
- 摘要：基于现有文本证据，该内容主要涉及：模拟 OCR 文字：例子_frame_0001.jpg 模拟视觉描述：例子_frame_0001.jpg 展示了一段待分析内容。
- 业务用途：可用于内容归档、素材检索和结构化结果验证。
- 使用证据：ocr_text、visual_description
- 缺失证据：audio_transcript
- 使用模型：ocr: doubao/mock-ocr（success, file_0003_call_0001）；visual_understanding: qwen/mock-vision（success, file_0003_call_0002）；speech_to_text: doubao/mock-asr（failed, file_0003_call_0003）；text_analysis: deepseek/mock-text（success, file_0003_call_0004）
- 关联模型调用：file_0003_call_0001、file_0003_call_0002、file_0003_call_0003、file_0003_call_0004
- 文件处理成本：0.020064 元
- 文件处理耗时：1 ms
- 错误信息：演示用语音识别失败：音频转写服务超时。
- 风险提示：语音识别分支失败，最终分析未使用音频转写证据。
