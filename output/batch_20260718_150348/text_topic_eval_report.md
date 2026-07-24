# 文本主分类人工评估报告

说明：本报告只评估文本文件的 `topic` 主分类，不评估摘要、关键词、副分类、图片或视频结果。

| 指标 | 数值 | 含义 |
|---|---:|---|
| 模板行数 | 1 | 需要人工判断的文本样本数 |
| 已评估样本数 | 1 | 已填写 gold_topic 的样本数 |
| 缺少标签样本数 | 0 | 尚未填写 gold_topic 的样本数 |
| 预测正确数 | 1 | predicted_topic 与 gold_topic 相同的样本数 |
| Accuracy | 100.00% | 文本主分类准确率 |

## 明细

| file_id | file_name | predicted_topic | gold_topic | 是否正确 | 备注 |
|---|---|---|---|---|---|
| file_0002 | ai_content_sample.txt | technology | technology | 是 | 分类正确 |

## 字段说明

| 字段 | 含义与作用 |
|---|---|
| `gold_topic` | 人工标注的正确主分类，用于和模型预测的 predicted_topic 对比。 |
| `predicted_topic` | 模型输出的主分类，用于衡量文本分类结果是否命中人工标签。 |
| `accuracy` | 文本主分类准确率，计算方式为 correct_count / evaluated_count。 |
| `evaluated_count` | 已经填写 gold_topic 并纳入统计的文本样本数。 |
