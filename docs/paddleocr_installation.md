# PaddleOCR 本地安装与验证指南

本文只说明如何在 Windows PowerShell 中为本项目安装 PaddleOCR，并把运行环境放在 H 盘项目目录内。

## 一、进入项目目录

```powershell
Set-Location "H:\实习\multimodal_model_router"
```

确认当前 Python：

```powershell
python --version
python -m pip --version
```

当前 Python 3.13 在 PaddlePaddle Windows 支持范围内。

## 二、创建独立虚拟环境

```powershell
python -m venv .venv
```

激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

如果提示“禁止运行脚本”，先执行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

确认 Python 已切换到项目内：

```powershell
where.exe python
```

第一条路径应类似：

```text
H:\实习\multimodal_model_router\.venv\Scripts\python.exe
```

## 三、更新安装工具

```powershell
python -m pip install --upgrade pip setuptools wheel
```

## 四、安装 PaddlePaddle CPU 版

本项目暂时使用 CPU 推理，不安装 GPU 版本：

```powershell
python -m pip install paddlepaddle==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

安装完成后检查：

```powershell
python -c "import paddle; print('PaddlePaddle版本：', paddle.__version__)"
```

再运行官方环境检查：

```powershell
python -c "import paddle; paddle.utils.run_check()"
```

正常情况下会看到类似：

```text
PaddlePaddle is installed successfully!
```

PaddleOCR 要求 PaddlePaddle 3.0 及以上版本。参见 [PaddleOCR 官方安装说明](https://paddlepaddle.github.io/PaddleOCR/main/version3.x/installation.html)。

## 五、安装项目依赖和 PaddleOCR

项目的 `requirements.txt` 已包含 PaddleOCR：

```powershell
python -m pip install -r .\requirements.txt
```

检查具体安装情况：

```powershell
python -m pip show paddlepaddle
python -m pip show paddleocr
```

验证导入：

```powershell
python -c "import paddle; import paddleocr; print('PaddleOCR导入成功')"
```

## 六、解决模型下载受阻问题

PaddleOCR 第一次真实运行时会下载模型权重。

如果访问默认模型源失败，可在当前 PowerShell 窗口设置百度对象存储源：

```powershell
$env:PADDLE_PDX_MODEL_SOURCE = "BOS"
```

然后重新运行 OCR。官方文档也建议在无法访问 Hugging Face 时切换到 BOS。参见 [PaddleOCR 模型下载说明](https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/module_usage/text_recognition.html)。

该环境变量只对当前 PowerShell 窗口生效，不会写入项目配置。

### Windows 中文模型缓存路径

当前机器实测发现：Paddle底层运行时可能无法稳定读取中文路径下的模型缓存文件，表现为模型初始化时报告空JSON或模型文件无法解析。项目代码已经支持中文输入图片路径，但模型权重路径仍可能受底层运行时影响。

不要把模型缓存迁移到C盘。可以临时把同一个H盘项目目录映射为英文盘符：

```powershell
subst P: "H:\实习\multimodal_model_router"
Set-Location "P:\"
$env:PADDLE_PDX_CACHE_HOME = "P:\.venv\paddlex_cache"
$env:PADDLE_PDX_MODEL_SOURCE = "BOS"
.\.venv\Scripts\python.exe .\src\main.py --ocr-backend paddleocr --input-dir .\input\sample_images --batch-id batch_paddleocr_smoke
Set-Location "H:\实习\multimodal_model_router"
subst P: /d
```

这里的 `P:` 只是临时别名，模型和输出仍实际保存在H盘项目目录。运行结束后用 `subst P: /d` 解除映射；如果 `P:` 已被占用，应换一个未占用盘符。

项目当前还显式关闭了MKLDNN，以避开PaddlePaddle 3.3.0在本机Windows CPU环境中遇到的oneDNN运行时兼容错误。这是当前机器的兼容处理，不等于所有环境都必须关闭。

## 七、用项目运行 PaddleOCR

先确认输入目录及其子目录中存在有效图片：

```powershell
Get-ChildItem .\input -File -Recurse
```

然后执行：

```powershell
python .\src\main.py --ocr-backend paddleocr --batch-id batch_paddleocr_smoke
```

参数说明：

| 参数 | 含义与作用 |
|---|---|
| `--ocr-backend paddleocr` | 明确选择本地 PaddleOCR 处理图片 |
| `--batch-id` | 指定本次批处理编号，便于找到对应输出结果 |
| `batch_paddleocr_smoke` | 本次验证使用的批次名称，可自行修改 |

这条命令：

- 会在本地运行 PaddleOCR；
- 不需要 API Key；
- 不需要增加 `--allow-live-api`；
- 不会调用 DeepSeek，因为未显式选择 DeepSeek；
- 第一次运行可能需要下载 PP-OCRv5 mobile 权重；
- 图片视觉理解仍然是 mock；
- 视频 OCR 仍然是 mock。

## 八、检查输出

运行成功后查看本次命令中指定的批次目录，例如：

```text
H:\实习\multimodal_model_router\output\batch_paddleocr_smoke
```

重点检查两个文件：

```text
results_readable.md
model_calls.jsonl
```

相关字段：

| 字段 | 含义与作用 |
|---|---|
| `ocr_text` | PaddleOCR 从图片中识别出的真实文字 |
| `provider` | OCR 模型提供方，应记录为 `paddlepaddle` |
| `model_name` | 使用的模型组合，应记录为 `PP-OCRv5_mobile` |
| `cost_cny` | 外部 API 费用，应为 0；不包含本机资源成本 |
| `latency_ms` | 单张图片的本地推理耗时 |
| `processing_status` | 文件处理状态，用于判断整体成功、部分成功或失败 |
| `evidence_used` | 最终分析实际使用了哪些证据，用于确认识别文字是否进入下游分析 |

## 九、以后重新进入环境

关闭 PowerShell 后，下一次只需：

```powershell
Set-Location "H:\实习\multimodal_model_router"
.\.venv\Scripts\Activate.ps1
```

退出虚拟环境：

```powershell
deactivate
```

## 十、建议的完整执行顺序

依次执行：

```powershell
Set-Location "H:\实习\multimodal_model_router"
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install paddlepaddle==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install -r .\requirements.txt
python -c "import paddle; paddle.utils.run_check()"
python -c "import paddleocr; print('PaddleOCR导入成功')"
subst P: "H:\实习\multimodal_model_router"
Set-Location "P:\"
$env:PADDLE_PDX_CACHE_HOME = "P:\.venv\paddlex_cache"
$env:PADDLE_PDX_MODEL_SOURCE = "BOS"
.\.venv\Scripts\python.exe .\src\main.py --ocr-backend paddleocr --input-dir .\input\sample_images --batch-id batch_paddleocr_smoke
Set-Location "H:\实习\multimodal_model_router"
subst P: /d
```

PaddleOCR 3.x 使用 `PaddleOCR(...).predict(...)` 完成推理，与项目当前实现一致。参见 [PaddleOCR Python 使用说明](https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/pipeline_usage/OCR.html)。
