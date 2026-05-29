# OpenAI Image API 兼容接口 (https://reachapi.ai/zh/docs/api-reference/image/openai-image-api)



本平台对外提供兼容 OpenAI Image API 的图片接口，当前推荐模型为 `gpt-image-2`。

截至 `2026-04-28`，OpenAI 官方模型页将 `gpt-image-2` 标记为默认的旗舰图片生成模型。在 ReachAPI 上，请求体与响应结构保持与 OpenAI 原生接口一致，集成时主要只需要替换基础域名。

如果需要任务查询、终态回调，或从 `data[].url` 获取最终图片地址，请使用 [GPT Image 2 异步图片接口](/docs/api-reference/image/openai-image-async-api)，并传入 `model: "gpt-image-2-async"`。

## 接口映射 [#接口映射]

| 能力   | OpenAI 原生地址                                    | ReachAPI 地址                                     |
| ---- | ---------------------------------------------- | ----------------------------------------------- |
| 图片生成 | `https://api.openai.com/v1/images/generations` | `https://api.reachapi.ai/v1/images/generations` |
| 图片编辑 | `https://api.openai.com/v1/images/edits`       | `https://api.reachapi.ai/v1/images/edits`       |

## 认证方式 [#认证方式]

使用 ReachAPI Key：

```http
Authorization: Bearer YOUR_REACH_API_KEY
```

图片生成请求使用：

```http
Content-Type: application/json
```

图片编辑请求沿用 OpenAI 原生 `multipart/form-data`，使用 `curl -F` 或 SDK/FormData 即可，不需要手动写死 `Content-Type` 边界。

## 兼容性说明 [#兼容性说明]

* 推荐模型：`gpt-image-2`
* 请求字段和响应对象遵循 OpenAI 官方 Image API 文档
* `POST /v1/images/edits` 使用 OpenAI 原生图片编辑请求格式，Content-Type 为 `multipart/form-data`
* 编辑请求字段沿用官方命名，例如 `image`、`mask`、`prompt`、`model`
* 官方 `gpt-image-2` 模型页当前标注支持图片生成与图片编辑，并注明不支持流式返回

## 最小示例 [#最小示例]

### 图片生成 [#图片生成]

```bash
curl -X POST "https://api.reachapi.ai/v1/images/generations" \
  -H "Authorization: Bearer YOUR_REACH_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-image-2",
    "prompt": "一张钴蓝色机械键盘的编辑风格产品图，放在石板灰桌面上"
  }'
```

### 图片编辑 [#图片编辑]

```bash
curl -X POST "https://api.reachapi.ai/v1/images/edits" \
  -H "Authorization: Bearer YOUR_REACH_API_KEY" \
  -F "model=gpt-image-2" \
  -F "prompt=将背景替换为干净的摄影棚背景，并优化整体布光" \
  -F "image=@/path/to/source.png"
```

## 官方文档 [#官方文档]

* 模型说明：[GPT Image 2](https://developers.openai.com/api/docs/models/gpt-image-2)
* 使用指南：[Image generation](https://platform.openai.com/docs/guides/images)
* 接口参考：[Create image](https://platform.openai.com/docs/api-reference/images/create)
* 接口参考：[Create image edit](https://platform.openai.com/docs/api-reference/images/createEdit)

像 `background`、`moderation`、`output_format`、`output_compression`、`size`、`quality`、`mask`、`input_fidelity` 等具体参数，以及完整响应字段，请以以上 OpenAI 官方文档为准；其中编辑接口请按官方原生 `multipart/form-data` 写法调用。
