# AGENTS.md

## 先看这里

- 保持精简，只写当前仓库真正需要的约定。
- 重要要求写在开头，避免分散。
- 每次更新本文件后，继续保持精简，不堆砌模板内容。
- 保持 ComfyUI 节点接口稳定，不随意修改 `NODE_CLASS_MAPPINGS`、`NODE_DISPLAY_NAME_MAPPINGS` 和 `generate` 的兼容性。
- 不要硬编码 API Key 等敏感信息。

## 仓库说明

这是一个 ComfyUI 自定义节点仓库，用于通过 APImart 调用 Gemini 图像生成接口。

## 关键文件

- `gemini_node.py`：核心节点与 API 调用逻辑
- `__init__.py`：ComfyUI 导出入口
- `requirements.txt`：依赖列表
- `README.md`：中文使用说明

## 修改约定

- 优先做小改动，避免无必要重构。
- API、轮询、图像张量 shape 相关修改要格外小心。
- 文档内容必须和当前代码一致。
