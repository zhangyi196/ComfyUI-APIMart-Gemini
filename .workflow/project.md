# Project: ComfyUI-APIMart-Gemini

## What This Is

ComfyUI 自定义节点包，通过 APImart 统一 API 接入 Gemini 和 GPT-Image-2 图像生成模型。面向需要在 ComfyUI 工作流中直接调用云端图像生成能力的用户。

## Core Value

**在 ComfyUI 中一站式使用 APImart 集成的多模型图像生成能力** — 如果一切功能都失效，至少 Gemini 和 GPT-Image-2 的文生图/图生图必须能稳定运行。

## Requirements

### Validated

<!-- 已发布并确认有价值的功能 -->

- Gemini 文生图（text-to-image）与图生图（image-to-image）
- GPT Image 2 文生图与图生图，支持 official_fallback 自动兜底
- GPT Image 2 Official 文生图、图生图、局部重绘（mask_image）
- 异步任务提交 + 轮询等待 + 结果下载的完整链路
- 最多 10 张参考图的图生图输入支持
- 图像张量格式兼容 NumPy 和 PyTorch

### Active

<!-- 当前正在构建的目标，发布前仍为假设 -->

- [ ] 修复 GPT Image 2 Official 节点 model 参数报错（upstream error: Missing required parameter: 'model'）
- [ ] 提升整体节点稳定性和错误处理质量

### Out of Scope

- 视频生成 — 仅专注图像生成
- APImart 以外的 API 渠道 — 不接入原生 OpenAI/Gemini API
- 独立 GUI — 节点仅在 ComfyUI 内使用
- 不引入新的第三方依赖 — 保持轻量

## Context

- 项目为个人维护，无团队协作开销
- ComfyUI 节点集成契约：`NODE_CLASS_MAPPINGS` 和 `NODE_DISPLAY_NAME_MAPPINGS` 不可随意变更
- APImart 是 API 代理/聚合平台，上游对接 OpenAI、Google 等，参数格式受 APImart 转发规则影响
- 三个节点共享类似的架构模式：提交任务 → 轮询 → 下载 → 转张量

## Constraints

- **兼容性**: 必须兼容 ComfyUI 的自定义节点加载机制
- **依赖**: 保持轻量，torch 为可选依赖（ComfyUI 自带）
- **API Key**: 禁止硬编码，通过节点参数传入
- **架构**: 优先小改动，避免无必要重构

## Tech Stack

- **Language**: Python 3.10+
- **Framework**: ComfyUI Custom Node
- **API**: APImart (`api.apimart.ai`)
- **Libraries**: requests, Pillow, NumPy, torch（可选）

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| 通过 APImart 统一接入多个模型 | 单一 API Key、统一鉴权、统一任务查询格式 | — Pending |
| 三个独立节点文件而非单一节点 | 每个模型的参数集差异较大，分离便于维护 | — Pending |
| torch 作为可选依赖 | ComfyUI 环境自带 torch，不应强制重装 | — Pending |

## Stakeholders

- zhangyi196（个人维护者）

---
*Last updated: 2026-05-18 after initialization*
