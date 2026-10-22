## Commands

Use `uv add <package>` to add runtime packages. Use `uv add --dev <package>` for development-only tools and dependencies.

```bash
uv run ruff format .
uv run ruff check .
uv run pyright .
```

## Git and production

- Format all files before committing.
- Make atomic commits: commit titles should tell the story of your work process.
- Commit titles start with a capitalized verb, e.g. "Add", "Merge", "Update", "Fix"... Any meaningful verb is allowed.

## Code style

- Never use `getattr`, `hasattr`, `__dict__`, `dir`, or `try/except` (except `KeyboardInterrupt` while showing plots or awaiting text input).
- Keep imports out of `if` blocks. Imports at the top of functions are fine when used only there. Do not use `from __future__ import annotations`.
- Annotate variables whose types cannot be inferred by linting. Prefer dataclasses and functions to OOP.
