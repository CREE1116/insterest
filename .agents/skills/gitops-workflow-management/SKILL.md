---
name: gitops-workflow-management
description: >-
  GitOps-driven development workflow covering branch creation, code modification,
  remote CI triggering, GitHub Actions run monitoring, iterative error diagnostics,
  and automated PR merging.
---

# GitOps Workflow Management

This skill document defines the standard operating procedures for making codebase modifications, verifying them via CI/CD, and merging them securely.

---

## 1. Branching & Scoped Modification

*   **Branch Creation**: Before making any code modifications, always create a descriptive feature branch from `master` (e.g., `feature/optimize-recommendations`).
*   **Minimal Diff**: Focus exclusively on the files and lines necessary to implement the implementation plan. Refrain from general cosmetic refactoring.

---

## 2. Remote-First Verification (No Local Tests)

*   **Skip Local Test Runs**: Do not spend time running unit tests or integration tests locally. Distribute the work to remote cloud runners (GitHub Actions CI) where environments (Postgres, Redis, Kafka, PyTorch, etc.) are already fully configured.
*   **Immediate Push**: As soon as code edits are complete and syntactically correct, stage and commit the changes, then push the branch to the remote repository.

---

## 3. Pull Request & CI Monitoring Loop

*   **Create PR Early**: Use the GitHub CLI to open a Pull Request targeting `master` immediately after pushing.
*   **Monitor Status Checks**: Use GitHub CLI commands to poll or monitor the triggered workflow:
    *   `gh pr status` to check the status of checks on the PR.
    *   `gh run list --limit 5` to find the active run ID.
    *   `gh run view <run-id>` to inspect job statuses.
*   **No Polling Sleep**: When waiting for CI runs, do not run shell `sleep` commands. Instead, use scheduling tools to set one-shot timers and release execution threads.

---

## 4. Iterative Debugging & Patching

*   **Inspect Remote Failures**: If a test job fails on GitHub Actions:
    *   Retrieve the failing job logs using `gh run view <run-id> --log` or inspect the issue created automatically by the workflow on failure.
    *   Locate the traceback, identify the bug, and apply a targeted fix in the local worktree.
*   **Push & Trigger Loop**: Commit the fix, push to the remote branch, and wait for the new CI run to execute. Repeat this cycle until all status checks pass.

---

## 5. Merging & Integration

*   **All Green Constraint**: Never merge or propose merging a PR unless all status checks on the pull request are completely green (`success`).
*   **Merge Execution**: Once all checks are green, execute the merge (e.g. `gh pr merge --merge` or via GitOps automation) to integrate the changes into `master`.
