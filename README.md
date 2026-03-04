# VQ-VAD: Task-Driven Video Anomaly Detection Pipeline
**(基于 CLIP 与多模态大模型的两阶段视频异常检测流水线)**

本项目是一个结合了 **CLIP 特征提取（召回）** 与 **多模态大语言模型（精排验证）** 的自动化视频异常事件检测与定位系统。系统通过预检模型快速筛选视频中的疑似异常片段，利用 FFmpeg 自动切片，最后输入到 Qwen2-VL 进行智能判定，从而实现高精度、任务驱动的异常行为定位。

##  核心功能 (Features)

1. **两阶段检测架构 (Coarse-to-Fine)**:
   - **召回阶段 (Recall)**: 基于定制化的 CLIP 视频异常检测模型（`CLIPVAD`），支持检测 20 种预定义的异常事件（如打架、纵火、抢劫、车祸等），快速定位疑似异常帧。
   - **验证阶段 (Verify)**: 调用多模态大语言模型（默认使用 `Qwen2-VL-7B-Instruct`），作为“安防专家”对切片视频进行二次验证，有效过滤误报（False Positives）。
2. **自动化切片与处理**: 集成 `ffmpeg-python`，根据召回模型的预测帧，精准裁剪视频片段。
3. **任务驱动的灵活性**: 支持通过描述文件（如 `video_annotations.txt`）为不同视频指定特定需要检测的异常事件。

##  项目结构 (Project Structure)

```text
VQ-VAD/
├── main.py             # 主程序入口，串联整个流水线（VAD推断 -> 视频切片 -> MLLM验证）
├── vad_detect.py       # 异常召回模块，负责视频特征提取并使用 CLIPVAD 预测疑似异常片段
├── mllm.py             # 多模态大模型验证模块，封装了 Qwen2-VL 的推理逻辑
├── videocut.py         # 视频切片工具，封装 FFmpeg 的视频精准截取功能
├── model.py            # CLIPVAD 模型架构定义
├── result.txt          # 流水线自动生成的检测结果（包含视频名、判定结果及帧区间）
└── ...
```

##  环境依赖 (Prerequisites)

为了顺利运行该项目，请确保系统中已安装以下依赖：

### 系统级依赖
需要安装 `ffmpeg` 以支持视频处理与切片：
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

### Python 库依赖
建议使用 Python 3.10 或以上版本。
```bash
pip install torch torchvision numpy opencv-python Pillow tqdm
pip install ffmpeg-python transformers qwen-vl-utils
pip install git+https://github.com/openai/CLIP.git
```

## 准备与配置 (Configuration)

在运行项目之前，需要根据实际环境修改代码中的相关硬编码路径。

### 1. 权重准备
* **VAD 模型权重**: 需将训练好的 CLIP-VAD 模型放置于指定目录，并在 `vad_detect.py` 的 `CONFIG` 中修改对应的权重路径。
* **大模型权重**: 下载 `Qwen2-VL-7B-Instruct` 模型，并将本地路径修改到 `mllm.py` 中的 `MODEL_PATH`（默认：`/workspace/Qwen2-VL-7B-Instruct`）。

### 2. 数据准备
在 `main.py` 中配置以下路径以匹配你的本地环境：
* **视频文件夹**: `VIDEO_FOLDER` (存放待检测的 `.mp4` 或 `.avi` 视频)
* **任务描述文件**: `ANNOTATION_FILE`

`video_annotations.txt` 格式示例（以空格分隔视频名和任务描述）：
``` text
video_001 Fighting
video_002 RoadAccidents
video_003 normal surveillance scene
```

## 使用方法 (Usage)

直接运行主入口脚本启动端到端检测流水线：

```bash
python main.py
```

### 处理流程简述
1. 解析 `video_annotations.txt` 获取目标视频和任务描述。
2. 调用 `vad_detect.py` 提取视频特征，通过 CLIP 模型获取候选异常片段 (Candidate Segments)。
3. 在 `batch_temp` 目录下将原视频按预测区间进行切片。
4. 调用 `mllm.py`，将切片与具体的任务描述输入给 Qwen2-VL 模型。大模型会输出 `1` (确认为异常) 或 `0` (误报)。
5. 将通过验证的片段帧索引写入 `result.txt`，并在运行结束后自动清理临时切片文件。

## 输出格式 (Output Format)

执行完毕后，项目会在当前目录生成 `result.txt`，每一行代表一个视频的检测结果。

格式如下：
* **存在异常**: `[视频名] 1 [起始帧1] [结束帧1] [起始帧2] [结束帧2] ... -1`
* **无异常**: `[视频名] 0 -1`

示例：
```text
video_001 1 240 360 400 520 -1
video_002 0 -1
```

## ⚠️ 注意事项
* **显存占用**: 该流水线同时集成了 CLIP 特征提取器（默认 ViT-B/16）、自定义 VAD 模型和 Qwen2-VL-7B，运行过程中会进行动态显存清理（`torch.cuda.empty_cache()`），但仍建议在具有充足显存 (至少 16GB-24GB VRAM) 的 GPU 设备上运行。
* **大模型推理**: `mllm.py` 中限制了视频的分辨率输入 (`max_pixels: 360 * 420`) 和抽帧率 (`fps: 2.0`) 以节省显存，可根据设备算力自行调整。