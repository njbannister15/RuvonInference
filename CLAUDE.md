# CLAUDE.md

## Workflow (Claude)
1. **Explore → Plan → Code → Commit → PR** (default loop).
   - Always plan diffs before editing.

2. **PRs**:
   - Small, reversible diffs.
   - Include *why*, link to issue/ticket, and describe test plan.


# Writing Code
When writing code please ensure that you write pydocs for each function, and educational comments, (The WHY is VERY importal), this ensure that the future editor can make changes to the code with context.

PLease use absolute imports and not relative imports

## Code Style & Tooling
- **Python version**: 3.12 (use uv for venv + deps).
- **Formatter**: `ruff` (via `make fmt`)
- **Linter**: `ruff` (via `make lint`)
- **Tests**: `pytest` (via `make test`)
- **Coverage**: encouraged; add when touching core logic.
- **Pydocs**: all functions my have comprehensive pydocs

👉 **Claude must run tools through Makefile:**
- Format: `make fmt`
- Lint: `make lint`
- Tests: `make test`
- CLI: `make run-cli`
- API: `make run-api`
