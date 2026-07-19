# Release Checklist

这份清单用于 GitHub 发布前检查。当前项目已经完成本地 Git 初始化；在用户明确授权前，不推送、不创建远程仓库、不创建 PR。

## 1. 代码清理

- [ ] 确认不修改核心业务代码后再发布。
- [ ] 确认 `src` 目录只包含项目源码，不包含缓存文件。
- [ ] 确认 `tests` 目录只包含测试源码，不包含缓存文件。
- [ ] 确认 `.idea`、`.vscode`、`__pycache__`、`.pytest_cache` 不进入版本库。
- [ ] 确认没有临时脚本、临时日志或本地调试文件。

## 2. 敏感信息检查

- [ ] 确认没有 `.env` 或 `.env.*` 文件被纳入版本库。
- [ ] 确认没有 API Key、真实密钥或 bearer token 出现在代码、配置、文档和输出文件中。
- [ ] 确认 `DEEPSEEK_API_KEY` 只通过本机环境变量读取。
- [ ] 确认 README 中只出现占位示例，不出现真实密钥。
- [ ] 确认发布前不要把本机环境变量截图上传到仓库。

## 3. README 检查

- [ ] 项目一句话定位清楚。
- [ ] 目标用户清楚：内容平台 AI 团队技术负责人。
- [ ] 当前能力边界清楚：真实接入 DeepSeek 文本分析；OCR、视觉理解、语音识别仍是 mock 或占位。
- [ ] 安装命令准确。
- [ ] 运行命令准确。
- [ ] 测试命令准确。
- [ ] Demo 路径准确。
- [ ] 图片路径可以在 GitHub README 中正常显示。

## 4. Demo 输出检查

- [ ] 保留 `output/batch_20260718_150348/` 作为作品集展示证据。
- [ ] 确认 `results_readable.md` 可读，能展示文件级分析结果。
- [ ] 确认 `batch_report.json` 包含批次成本、延迟、成功率和错误质量统计。
- [ ] 确认 `model_calls.jsonl` 包含模型调用明细。
- [ ] 确认没有把其他临时批次误作为展示证据。

关键文件说明：

| 文件 | 作用 |
|---|---|
| `results_readable.md` | 人工可读结果页，用来展示每个文件的分类、摘要、证据、模型、成本和耗时 |
| `batch_report.json` | 批次统计报告，用来展示总成本、成功率和 P95 延迟 |
| `model_calls.jsonl` | 模型调用明细，用来追踪每次调用的供应商、模型、用量、成本和延迟 |

## 5. 图片路径检查

- [ ] 确认 `assets/demo_results_readable.png` 存在。
- [ ] 确认 `assets/batch_report_summary.png` 存在。
- [ ] 确认 `assets/model_call_chain.png` 存在。
- [ ] 确认 `assets/architecture_diagram.png` 存在。
- [ ] 确认 README 中图片路径使用相对路径，例如 `assets/demo_results_readable.png`。
- [ ] 确认 `docs/portfolio_showcase.md` 中图片路径使用 `../assets/...`。

## 6. 测试检查

- [ ] 运行离线测试命令：

```powershell
python -m unittest discover -s tests
```

- [ ] 确认测试结果为 OK。
- [ ] 确认测试不会触发 DeepSeek API。
- [ ] 确认没有把 live test 当作默认测试。
- [ ] 确认 `docs/tests.md` 已说明已有测试、未覆盖风险和后续测试计划。

## 7. GitHub 发布前检查

- [x] 本地 Git 仓库已初始化。
- [ ] 获得用户明确授权后再创建远程仓库或执行 push。
- [ ] 检查 `.gitignore` 是否已经覆盖 Python 缓存、虚拟环境、IDE 配置、环境变量文件、日志和临时输出。
- [ ] 确认 `.gitignore` 没有忽略当前展示批次 `output/batch_20260718_150348/`。
- [ ] 初始化仓库后先检查将要提交的文件列表，再提交。
- [ ] 不要提交 `.idea`、`__pycache__`、`.env`、`.venv`、临时日志或其他本地文件。

## 8. 简历/面试材料检查

- [ ] README 中有简历可用表述。
- [ ] `docs/portfolio_showcase.md` 能在 3 分钟内讲清项目。
- [ ] Demo 图能展示结果、调用链、架构、成本和延迟。
- [ ] 面试讲法不夸大当前能力。
- [ ] 能主动说明真实与 mock 边界。

## 9. 提交前风险清单

| 风险 | 当前状态 | 处理方式 |
|---|---|---|
| 真实 API Key 被提交 | 未发现真实密钥，仍需提交前复查 | `.gitignore` 排除 `.env`，发布前人工检查 |
| IDE 配置进入仓库 | 当前存在 `.idea` | `.gitignore` 已排除 `.idea/` |
| Python 缓存进入仓库 | 当前存在 `src/__pycache__` 和 `tests/__pycache__` | `.gitignore` 已排除缓存目录 |
| 展示批次被误忽略 | 当前展示批次必须保留 | `.gitignore` 忽略其他输出批次，但显式保留 `output/batch_20260718_150348/` |
| 项目能力被夸大 | 文档已标注真实与 mock 边界 | 发布前继续检查措辞 |
