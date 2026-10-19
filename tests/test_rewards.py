import pytest

from dqntrain.rewards import (
    _text,
    contains_reward,
    exact_match_reward,
    format_reward,
    length_penalty,
)


def test_text_extraction():
    assert _text("plain") == "plain"
    assert _text([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]) == "yo"
    assert _text([{"role": "user", "content": "hi"}]) == "hi"
    assert _text({"content": "solo"}) == "solo"
    assert _text([{"role": "assistant", "content": [{"type": "text", "text": "parts"}]}]) == "parts"


def test_format_reward():
    fn = format_reward(r"^yes")
    assert fn(["yes please", "no"]) == [1.0, 0.0]
    assert fn([{"role": "assistant", "content": "yes"}]) == [1.0]


def test_contains_reward():
    fn = contains_reward("42")
    assert fn(["the answer is 42", "nothing"]) == [1.0, 0.0]


def test_exact_match_reward():
    fn = exact_match_reward("answer")
    assert fn([" 42 ", "41"], answer=["42", "42"]) == [1.0, 0.0]
    assert fn(["x"], answer=[]) == [0.0]


def test_length_penalty():
    fn = length_penalty(3)
    scores = fn(["abc", "ab", "abcdef"])
    assert scores[0] == 1.0
    assert 0.0 < scores[1] < 1.0
    assert scores[2] == 0.0
