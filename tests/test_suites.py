import pytest

from vlashift.suites import SUITES, TASKS_PER_SUITE, instructions, locate

ALL = [(suite, i, text) for suite in SUITES for i, text in enumerate(instructions()[suite])]


def test_four_suites_with_ten_instructions_each():
    table = instructions()
    assert tuple(table) == SUITES
    assert all(len(texts) == TASKS_PER_SUITE for texts in table.values())


def test_the_forty_instructions_are_unique():
    texts = [text for _, _, text in ALL]
    assert len(texts) == 40
    assert len(set(texts)) == 40


@pytest.mark.parametrize(("suite", "position", "text"), ALL)
def test_locate_round_trip(suite, position, text):
    assert locate(text) == (suite, position)


@pytest.mark.parametrize(("suite", "position", "text"), ALL)
def test_instruction_is_clean_text(suite, position, text):
    assert text == text.strip()
    assert "  " not in text
    assert text == text.lower()


def test_locate_rejects_unknown_text():
    with pytest.raises(KeyError, match="not an instruction"):
        locate("pick up the black bowl")


def test_locate_is_exact_about_case_and_spaces():
    text = instructions()["libero_goal"][7]
    with pytest.raises(KeyError):
        locate(text + " ")
    with pytest.raises(KeyError):
        locate(text.capitalize())
