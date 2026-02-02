import pandas as pd

from elections.audit import audit_rowcount, run_audit_path


def test_audit_rowcount_run_scoped(tmp_path) -> None:
    df = pd.DataFrame({
        "iso3": ["UTO"],
        "election_year": [2000],
    })
    run_id = "20260129_000000"
    audit_root = tmp_path / "reports" / "audit"
    audit_path = run_audit_path(audit_root, run_id, "rowcount_coverage_audit.csv")

    audit_rowcount(df, "table", "step", audit_path, key_cols=["iso3"], run_id=run_id)
    audit_rowcount(df, "table", "step2", audit_path, key_cols=["iso3"], run_id=run_id)

    out = pd.read_csv(audit_path)
    assert "run_id" in out.columns
    assert len(out) == 2
    assert (out["run_id"] == run_id).all()
    assert not (audit_root / "rowcount_coverage_audit.csv").exists()
