# 图片 OCR 预处理实验报告：img_9.jpg

## 一、实验目标

只围绕 `img_9.jpg` 验证两种最小预处理是否能改善小字号结构图 OCR：整图放大2倍、左右分区后各自放大2倍。

## 二、原图基线

- 完整段落召回率：47.62%
- 字符错误率：20.27%
- OCR延迟：28261ms
- 编辑距离合计：107

## 三、实验结果

| 变体 | 状态 | 完整段落召回率 | 字符错误率 | OCR延迟 | 编辑距离合计 |
|---|---|---:|---:|---:|---:|
| full_image_2x | success | 52.38% | 20.27% | 64146ms | 107 |
| vertical_halves_2x | success | 50.00% | 20.27% | 32421ms | 107 |

## 四、实验结论

结论：`improved_but_not_passed`

最佳变体：`full_image_2x`

- 最佳变体完整段落召回率从47.62%提升到52.38%。
- 最佳变体字符错误率未低于原图基线20.27%。

下一步：预处理方向有一定价值，但不能直接通过闸门；下一步只做一个更窄的分区或裁剪实验。

## 五、生成的预处理图片

- `output\batch_paddleocr_keyframes_20260724_retry\image_ocr_preprocess_experiment_img_9\img_9_full_2x.png`
- `output\batch_paddleocr_keyframes_20260724_retry\image_ocr_preprocess_experiment_img_9\img_9_left_2x.png`
- `output\batch_paddleocr_keyframes_20260724_retry\image_ocr_preprocess_experiment_img_9\img_9_right_2x.png`

## 六、边界说明

- 本实验只调用本地PaddleOCR，不调用DeepSeek，也不新增视觉理解、语音识别或视频处理能力。
- 本实验只比较同一张图片在不同预处理方式下的OCR结果，不能证明PaddleOCR线上泛化质量。
- 本实验产生的预处理图片是实验中间产物，用于复现本次OCR输入，不应混入正式业务输入目录。
