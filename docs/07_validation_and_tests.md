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

## Latest pass evidence
- `make lint` — passed (ruff on `src` and `tests`).
- `make run` — completed successfully (paper pipeline).
- `make test` — 16 tests passed; warnings in `tests/test_spec_search.py` about divide‑by‑zero in statsmodels (logged in execution log).
- `make repro` — completed successfully (clean + run + test).
