---
title: "Architecture Constraints"
readMode: required
priority: high
category: arch
keywords:
  - architecture
  - module
  - layer
  - boundary
  - dependency
  - structure
---

# Architecture Constraints

Auto-generated from project structure. Update manually as architecture evolves.

## Module Structure

- Type: single-package (flat, no subdirectories)
- Key modules:
  - `gemini_node.py` — Gemini image generation node
  - `gpt_image_2_node.py` — GPT Image 2 generation node (with official_fallback)
  - `gpt_image_2_official_node.py` — GPT Image 2 Official generation node
  - `__init__.py` — ComfyUI entry point, aggregates all node mappings

## Layer Boundaries

```
ComfyUI Framework → Node Class (INPUT_TYPES/generate) → APImart API → Image Processing → ComfyUI Tensor
```

Each node file is a self-contained vertical slice — no shared utilities module.

## Dependency Rules

- Nodes do NOT import from each other
- `__init__.py` is the only file that imports from other project modules
- External dependencies: `requests`, `Pillow`, `numpy` (required), `torch` (optional)
- No circular dependencies

## Technology Constraints

- Runtime: Python >= 3.10
- Module system: standard Python imports
- Strict mode: no (no mypy/pyright config)
- API endpoint: `api.apimart.ai`
- Must not hardcode API keys

## Entries
