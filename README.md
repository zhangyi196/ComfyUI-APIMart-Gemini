# ComfyUI-APIMart-Gemini

这是一个用于 ComfyUI 的自定义节点，通过 APImart 接入 Gemini 图像生成模型。

## 项目简介

当前仓库提供 1 个自定义节点：

- `Gemini Image Generation (APImart)`

支持能力：

- 文生图
- 图生图，最多支持 10 张参考图
- 轮询 APImart 异步任务直到完成
- 下载生成结果并返回给 ComfyUI

## 文件说明

- `gemini_node.py`：ComfyUI 节点的主要实现
- `__init__.py`：导出 ComfyUI 所需的节点映射
- `requirements.txt`：本项目依赖列表

## 环境要求

- Python 3.10 及以上
- ComfyUI
- APImart API Key

安装依赖：

```bash
pip install -r requirements.txt
```

说明：

- 仓库中的 `torch` 按可选依赖处理，因为很多 ComfyUI 环境本身已经自带它。
- 如果你的 ComfyUI 运行环境没有 `torch`，请按你的本地环境自行安装匹配版本。

## 安装方式

1. 将本仓库放到 ComfyUI 的 `custom_nodes` 目录下。
2. 根据 `requirements.txt` 安装依赖。
3. 重启 ComfyUI。

示例路径：

```text
ComfyUI/custom_nodes/ComfyUI-APIMart-Gemini
```

## 节点输入

必填输入：

- `mode`：`text-to-image` 或 `image-to-image`
- `api_key`：APImart API Key
- `prompt`：生成提示词
- `model`：Gemini 图像模型
- `n`：生成图片数量
- `resolution`：输出分辨率
- `size`：画面比例
- `seed`：可选随机种子

可选输入：

- `image_1` 到 `image_10`：`image-to-image` 模式下使用的参考图

## 节点输出

- `image`：返回给 ComfyUI 的图像张量
- `image_url`：APImart 返回的最终图片地址
- `response`：格式化后的接口响应文本

## 使用方法

1. 在 ComfyUI 中添加 `Gemini Image Generation (APImart)` 节点。
2. 输入你的 APImart API Key。
3. 选择模型和生成模式。
4. 如果使用 `image-to-image`，连接一张或多张输入图像。
5. 运行工作流，等待任务轮询完成。

## 常见问题

- 如果节点立即报错，请先检查 API Key 和请求参数是否正确。
- 如果轮询超时，请检查 APImart 任务状态和本地网络连接。
- 如果图像转换失败，请确认接入的是合法的 ComfyUI 图像张量。

## 开发说明

- `NODE_CLASS_MAPPINGS` 和 `NODE_DISPLAY_NAME_MAPPINGS` 中的名称属于 ComfyUI 集成契约，除非明确要做破坏性变更，否则不要随意改动。
- 新增或修改逻辑时，尽量同时兼容 NumPy 数组和 PyTorch 张量输入。
