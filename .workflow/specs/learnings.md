---
title: "Learnings"
readMode: optional
priority: medium
category: learning
keywords:
  - bug
  - lesson
  - gotcha
  - learning
---

# Learnings

Add entries with: `/spec-add learning <description>`

## Entries

<spec-entry category="learning" keywords="openai,multipart,request-summary" date="2026-05-22" source="execute">
在 `xx_gpt_image_2_official_node.py` 中，优先把图片统一转换为 `(filename, bytes, mime_type)` 资源，再分别派生 JSON base64 与 multipart 请求；调试日志只保留 `upload_mode`、`field_strategy`、`image_count`、`estimated_bytes` 等元数据，避免把 base64 正文写入日志。
</spec-entry>

