import hashlib

import pytest

from vlashift.instructions import (
    MODES,
    blank,
    keep,
    other,
    paraphrase,
    paraphrases,
    shuffled,
    transform,
)
from vlashift.suites import SUITES, instructions, locate

ALL = [text for suite in SUITES for text in instructions()[suite]]


def test_keep_and_blank():
    assert keep("turn on the stove") == "turn on the stove"
    assert blank("turn on the stove") == ""


@pytest.mark.parametrize("text", ALL)
def test_other_is_the_next_task_of_the_same_suite(text):
    suite, position = locate(text)
    expected = instructions()[suite][(position + 1) % 10]
    assert other(text) == expected
    assert other(text) != text
    assert locate(other(text))[0] == suite


def test_other_wraps_around_the_last_task():
    last = instructions()["libero_object"][9]
    assert other(last) == instructions()["libero_object"][0]


@pytest.mark.parametrize("suite", SUITES)
def test_ten_steps_of_other_come_back_to_the_start(suite):
    text = start = instructions()[suite][3]
    for _ in range(10):
        text = other(text)
    assert text == start


@pytest.mark.parametrize("text", ALL)
def test_shuffled_keeps_the_words_and_changes_the_order(text):
    out = shuffled(text)
    assert sorted(out.split()) == sorted(text.split())
    assert out != text


@pytest.mark.parametrize("text", ALL)
def test_shuffled_is_fixed_for_a_text(text):
    assert shuffled(text) == shuffled(text)


def test_shuffled_follows_the_documented_key_order():
    text = "turn on the stove"
    keys = [hashlib.sha256(f"0|{text}|{i}".encode()).digest() for i in range(4)]
    expected = " ".join(w for _, w in sorted(zip(keys, text.split(), strict=True)))
    if expected == text:
        pytest.skip("identity permutation, the function moves to the next salt")
    assert shuffled(text) == expected


def test_shuffled_differs_between_tasks_of_one_suite():
    outs = {shuffled(t) for t in instructions()["libero_object"]}
    assert len(outs) == 10


def test_shuffled_refuses_a_text_it_cannot_reorder():
    with pytest.raises(ValueError, match="cannot reorder"):
        shuffled("stove")


def test_paraphrase_table_covers_exactly_the_forty_instructions():
    assert set(paraphrases()) == set(ALL)


@pytest.mark.parametrize("text", ALL)
def test_paraphrase_is_a_different_sentence(text):
    out = paraphrase(text)
    assert out.strip() == out
    assert out.lower() != text.lower()
    assert out not in ALL


def test_paraphrases_are_distinct():
    values = list(paraphrases().values())
    assert len(set(values)) == len(values)


@pytest.mark.parametrize("text", ALL)
def test_paraphrase_keeps_the_object_words(text):
    nouns = {
        "bowl",
        "soup",
        "cheese",
        "dressing",
        "bbq",
        "ketchup",
        "tomato",
        "butter",
        "milk",
        "pudding",
        "juice",
        "drawer",
        "stove",
        "plate",
        "bottle",
        "mug",
        "book",
        "moka",
    }
    original = {w for w in text.split() if w in nouns}
    assert original, text
    assert original <= set(paraphrase(text).split())


def test_unknown_text_is_refused_by_other_and_paraphrase():
    with pytest.raises(KeyError):
        other("wave at the camera")
    with pytest.raises(KeyError):
        paraphrase("wave at the camera")


def test_transform_applies_the_mode_to_every_text():
    texts = [ALL[0], ALL[11]]
    assert transform("keep", texts) == texts
    assert transform("blank", texts) == ["", ""]
    assert transform("other", texts) == [other(t) for t in texts]
    assert transform("paraphrase", texts) == [paraphrase(t) for t in texts]
    assert transform("shuffled", texts) == [shuffled(t) for t in texts]
    assert transform("keep", []) == []


def test_transform_rejects_unknown_modes():
    with pytest.raises(ValueError, match="unknown instruction mode"):
        transform("swap", ["turn on the stove"])


def test_mode_names():
    assert MODES == ("keep", "blank", "other", "shuffled", "paraphrase")
