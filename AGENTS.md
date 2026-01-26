# AGENTS.md — Project rules for Codex

## Non-negotiables
- plan.tex is the single source of truth; execute it end-to-end.
- Always run tests / pipeline steps after edits.
- Document everything in docs/*.md (traceability, reproducibility, results, command logs).
- Do not do data cleaning unless plan.tex explicitly requires it.
- Repo must end clean: only plan-relevant files remain; archive others to archive/legacy_pre_plan/.

## Default commands
- Prefer `make setup`, `make test`, `make run`, `make repro` if present.
- If not present, create them and then use them.
