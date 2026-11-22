from src.code_validation import get_valid_taric_prefix, validate_cn8


VALID_CODES = {"72111400", "94039910"}


def test_validate_cn8_does_not_truncate_taric_code():
    code, status = validate_cn8("7211 1400 95", VALID_CODES)
    assert code == "7211140095"
    assert status == "Format invalide"


def test_taric_prefix_is_only_returned_when_official():
    assert get_valid_taric_prefix("7211 1400 95", VALID_CODES) == "72111400"
    assert get_valid_taric_prefix("1234567890", VALID_CODES) is None


def test_regular_cn8_is_not_a_taric_candidate():
    assert get_valid_taric_prefix("94039910", VALID_CODES) is None
