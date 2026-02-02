from elections.standardize import normalize_name


def test_normalize_name_diacritics() -> None:
    assert normalize_name("Curaçao") == "curacao"
    assert normalize_name("Côte d'Ivoire") == "cote d ivoire"
    assert normalize_name("Türkiye") == "turkiye"
