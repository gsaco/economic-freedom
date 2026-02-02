import pandas as pd

from elections.build_sources import aggregate_clea_lc, SourcePaths


def test_clea_aggregation_weighted_fallback(monkeypatch, tmp_path) -> None:
    df = pd.DataFrame({
        "id": [1, 1],
        "ctr": ["UTO", "UTO"],
        "ctr_n": ["Utopia", "Utopia"],
        "yr": [2000, 2000],
        "mn": [1, 1],
        "pty": [1, 2],
        "pty_n": ["Party A", "Party B"],
        "pv1": [pd.NA, pd.NA],
        "pvs1": [60.0, 40.0],
        "vv1": [1000.0, 1000.0],
        "seat": [50.0, 50.0],
    })

    def fake_read(_path):
        return df

    monkeypatch.setattr("elections.build_sources.read_clea_lc", fake_read)

    paths = SourcePaths(raw_dir=tmp_path, interim_dir=tmp_path, audit_dir=tmp_path)
    _agg, out = aggregate_clea_lc(paths)

    assert out.loc[0, "share_1"] == 60.0
    assert out.loc[0, "share_2"] == 40.0
    assert out.loc[0, "vote_share_method"] == "pvs1_weighted"
