---
title: "Coding Conventions"
readMode: required
priority: high
category: coding
keywords:
  - style
  - naming
  - import
  - pattern
  - convention
  - formatting
---

# Coding Conventions

Auto-generated from project analysis. Update manually as patterns evolve.

## Formatting

- Indentation: 4 spaces
- Line length: not configured
- Trailing commas: none
- Semicolons: none

## Naming

- Variables/functions: snake_case
- Classes/types: PascalCase
- Constants: UPPER_SNAKE_CASE
- Files: snake_case

## Imports

- Style: named imports
- Path aliases: none (relative imports from package root)
- Order: standard library, third-party, optional (try/except for torch)
- Optional dependencies gated with `try/except ImportError`

## Patterns

- ComfyUI node contract: `INPUT_TYPES`, `RETURN_TYPES`, `RETURN_NAMES`, `FUNCTION`, `CATEGORY` class attributes
- Image tensor handling: compatible with both NumPy arrays and PyTorch tensors
- API polling pattern: submit task → poll with max retries → download → convert to tensor
- Logging: `print()` with `[ClassName]` prefix
- Error handling: try/except with Chinese error messages, re-raise with context
- Type hints: `typing` module (`Dict`, `List`, `Tuple`, `Any`, `Optional`)

## Entries
