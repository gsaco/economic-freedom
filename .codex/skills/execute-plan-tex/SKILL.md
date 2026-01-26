---
name: execute-plan-tex
description: Use when the user asks to execute plan.tex end-to-end with verified runs, tests, and Markdown documentation, leaving the repo clean and plan-focused.
---

# Execute plan.tex end-to-end

Follow the “Execute plan.tex end-to-end” workflow:
- Build a traceability matrix mapping plan.tex items → artifacts → verification.
- Implement incrementally: code → run → test → document → repeat.
- Require make targets: setup/test/run/repro/clean.
- Log every command and outcome in docs/05_execution_log.md.
- Finish with repo cleanup: archive unrelated files to archive/legacy_pre_plan/ and write docs/08_repo_cleanup_report.md.
- Definition of done: all plan items DONE + make repro passes.
