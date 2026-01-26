# Validation and Tests

## Test suite
- **Run all tests:**
  ```bash
  make test
  ```

## Linting
```bash
make lint
```

## Pipeline verification
- **Full pipeline:**
  ```bash
  make run
  ```
- **Clean reproduction:**
  ```bash
  make repro
  ```

## Tests included
- `tests/test_close_elections_unique_keys.py` — uniqueness of (`iso3c`, `election_year`) in close‑election sample.
- `tests/test_tail_outcomes_presence.py` — tail outcomes table exists after pipeline.
- `tests/test_exclusion_bounds_presence.py` — exclusion bounds outputs exist after pipeline.
- `tests/test_clea_manifest.py` — CLEA manifest schema (skips if CLEA absent).
- `tests/test_lp_reduced_form_presence.py` — LP reduced‑form output exists after pipeline.

## Latest pass evidence
- `make lint` — passed.
- `make run` — completed successfully (paper pipeline).
- `make test` — 14 passed, 4 skipped, 5 warnings (divide‑by‑zero warnings in `tests/test_spec_search.py`).
- `make repro` — completed successfully (clean + run + test).
