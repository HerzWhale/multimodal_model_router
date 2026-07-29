# 图片 OCR 错误归因报告：img_9.jpg

## 一、指标概览

- 批次：`batch_paddleocr_keyframes_20260724_retry`
- 必选文字块数：42
- 完整命中文字块数：20
- 未完整命中文字块数：22
- 完整段落召回率：47.62%
- 字符错误率：20.27%
- OCR延迟：28261ms

## 二、闸门判断

结论：`not_passed`

- 完整段落召回率为47.62%，低于当前MVP观察阈值90%。
- 字符错误率为20.27%，高于当前MVP观察阈值5%。
- OCR延迟为28261ms，高于图片任务2000ms目标。

下一步：继续留在图片OCR功能内，先解释弱样本和延迟瓶颈。

## 三、按文字块类型聚合

| 文字块类型 | 总段数 | 错误段数 | 错误段占比 | 编辑距离合计 | 字符错误率 |
|---|---:|---:|---:|---:|---:|
| pipeline_module | 14 | 7 | 50.00% | 45 | 21.84% |
| buffer_size | 6 | 6 | 100.00% | 28 | 32.56% |
| tlb_size | 4 | 4 | 100.00% | 24 | 41.38% |
| cache_module | 2 | 2 | 100.00% | 6 | 23.08% |
| diagram_group_label | 6 | 2 | 33.33% | 3 | 25.00% |
| scheduler_module | 6 | 1 | 16.67% | 1 | 0.85% |
| diagram_title | 4 | 0 | 0.00% | 0 | 0.00% |

## 四、按错误类型聚合

| 错误类型 | 段数 | 编辑距离合计 | 解释 |
|---|---:|---:|---|
| value_retained_label_lost | 10 | 46 | 数值被识别到，但标签或模块名称缺失，常见于小字号指标说明。 |
| label_retained_value_lost | 8 | 56 | 标签或模块名称被识别到，但容量、页数、Entry等数值信息缺失。 |
| partial_text_match | 2 | 3 | 只匹配到较短片段，说明文字块被截断或被复杂布局拆散。 |
| character_substitution_or_layout_noise | 2 | 2 | 有接近片段但存在字符替换、符号误读或布局噪声。 |

## 五、编辑距离最高的错误段

| segment_id | segment_type | edit_distance | error_bucket | gold_text | comparison_text |
|---|---|---:|---|---|---|
| left_fp_simd_prf | pipeline_module | 9 | label_retained_value_lost | FP & SIMD PRF 160 Entry | FPSSIMDPRF |
| right_fp_simd_prf | pipeline_module | 9 | label_retained_value_lost | FP & SIMD PRF 222 Entry | FPGSIMDPRF |
| left_integer_prf | pipeline_module | 8 | label_retained_value_lost | Integer PRF 158 Entry | IntegerPRF |
| left_reorder_buffer | buffer_size | 8 | label_retained_value_lost | Re-order Buffer 504 Entry | Re-orderBuffer |
| right_integer_prf | pipeline_module | 8 | label_retained_value_lost | Integer PRF 222 Entry | IntegerPRF |
| right_reorder_buffer | buffer_size | 8 | label_retained_value_lost | Re-order Buffer 442 Entry | Re-orderBuffer |
| left_l1d_tlb | tlb_size | 6 | value_retained_label_lost | L1D TLB 128 pages | 128pages |
| left_l2d_tlb | tlb_size | 6 | value_retained_label_lost | L2D TLB 1024 pages | 1024pages |
| right_l1d_tlb | tlb_size | 6 | value_retained_label_lost | L1D TLB 256 pages | 256pages |
| right_l2d_tlb | tlb_size | 6 | value_retained_label_lost | L2D TLB 1024 pages | 1024pages |

## 六、人工解读

- img_9.jpg 的错误主要集中在：pipeline_module, buffer_size, tlb_size。
- 该图属于小字号、双栏结构图，OCR更容易把模块标签、容量数值和相邻文字拆散或压缩。
- 本图OCR延迟为28261ms，高于图片任务2秒目标。

## 七、边界说明

- 本报告只解释已有OCR评估结果，不重新运行PaddleOCR，也不调用DeepSeek。
- 错误归因来自分段人工基准与OCR文本的比较，是工程诊断，不是模型泛化质量结论。
- 视觉理解、语音识别和视频真实处理仍未接入，不能把本报告解释为完整多模态质量评估。
