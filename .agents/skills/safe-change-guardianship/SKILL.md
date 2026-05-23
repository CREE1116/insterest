---
name: safe-change-guardianship
description: >-
  Safe change guardianship skill covering blast radius mitigation, change reviews,
  refactoring risk assessment, backward compatibility guarantees, schema migrations checks,
  dependency drift auditing, and unnecessary change mitigation.
---

# Safe Change Guardianship

This skill document defines rules and methodologies to mitigate risks when deploying code modifications, schema migrations, and library updates to production.

---

## 1. Unnecessary Change Mitigation (Minimal Diff Rule)

To maintain a clean git history and simplify review pipelines, all changes must strictly respect the **Minimal Diff Rule**.

### A. Core Directives
*   **Focus on the Goal**: Modify only the lines of code directly required to implement a fix or feature.
*   **No Arbitrary Formatting**: Avoid running auto-formatters (e.g. `black`, `ruff`) across entire files unless explicitly instructed. Refrain from changing indentation, line wrapping, or spacing in unrelated blocks.
*   **Preserve Context & Comments**: Never delete docstrings, comments, or annotations that are unrelated to your current task.
*   **No Feature Creep**: Do not refactor adjacent code blocks or fix unrelated "code smells" unless they directly impact the correctness of your planned modification.

---

## 2. Backward Compatibility & Blast Radius Control

In microservices architectures, updating an API endpoint or database schema can break upstream consumers if not handled safely.

### A. API Contract Versioning
*   **Never Modify Existing Fields**: Do not delete or rename fields in JSON request/response Pydantic models.
*   **Add Optional Fields Only**: If new data is required, add fields as optional (`Optional[type] = None`) to prevent older client versions from crashing.
*   **Graceful Deprecation**: Mark deprecated endpoints with `deprecated=True` in FastAPI, allowing clients time to migrate before removal.

### B. Database Migration Safety (Expand-Contract Pattern)
For schema migrations (e.g., adding or altering columns via Alembic), always follow the **Expand-Contract (Parallel Run)** strategy:
1.  **Phase 1 (Expand)**: Deploy the database change (e.g., add new projected vector column `vector_128d`). Ensure the application writes to both columns but reads from the old column.
2.  **Phase 2 (Backfill)**: Backfill data from the old column to the new column asynchronously.
3.  **Phase 3 (Transition)**: Update the application to read from the new column.
4.  **Phase 4 (Contract)**: Drop or archive the old column and remove dual-write code.

---

## 3. Dependency Drift Auditing

Library versions must be strictly managed to prevent supply chain bugs or dependency drift.

### A. Lockfile Compliance
*   **Pin exact dependencies** in `requirements.txt` or `pyproject.toml` (e.g., `torch==2.1.2`, `redis[hiredis]==5.0.1`).
*   Always run `pip-compile` or `uv pip compile` to update lock files and verify there are no sub-dependency conflicts.
*   Avoid adding high-weight dependencies unless absolutely necessary.

---

## 4. GitOps & CI-Driven Verification Loop

In modern GitOps setups, local verification is secondary to the remote Continuous Integration (CI) environment. Agents must prioritize pushing changes quickly and relying on remote CI runners (e.g., GitHub Actions) for final test validation and environment debugging.

For detailed guidelines and tool commands, refer to [gitops-workflow-management](file:///Users/leejongmin/code/insterest/.agents/skills/gitops-workflow-management/SKILL.md).

### A. Core Directives
*   **Push Early, Test Remote**: Instead of spending excessive time configuring complex databases, caches, or GPU runtimes locally, push code modifications to the feature branch and trigger remote CI actions.
*   **GitHub CLI Integration**:
    *   Use `gh pr create` to establish a Pull Request (PR) as early as draft stage.
    *   Monitor the status of tests using `gh run list` and `gh run view --log` to capture execution output directly from the CI runners.
*   **Iterative Patching Loop**:
    *   If a remote CI job fails, inspect the runner logs, isolate the test crash (e.g., Python exceptions, DB constraints, docker failures), apply a focused fix, commit, and push again.
    *   Repeat this loop until all CI status checks pass cleanly (`success`).
*   **Never Bypass CI**: No code should be recommended for final merge or manual deployment unless all automated status checks on the GitHub Pull Request are completely green.

