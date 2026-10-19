"""Built-in reward helpers for GRPO.

Reward functions are called by TRL as ``fn(completions, **dataset_columns)``
and must return one float per completion. Completions may be strings or
chat-format lists of messages.
"""
from __future__ import annotations

import re


def _content_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content)


def _text(completion):
    """Extract plain text from a string or chat-format completion."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return _content_text(completion.get("content", ""))
    if isinstance(completion, (list, tuple)):
        for message in reversed(completion):
            if isinstance(message, dict) and message.get("role") == "assistant":
                return _content_text(message.get("content", ""))
        if completion:
            last = completion[-1]
            return _content_text(last.get("content", "")) if isinstance(last, dict) else str(last)
        return ""
    return str(completion)


def format_reward(pattern):
    """1.0 when the completion matches the regex ``pattern``, else 0.0."""
    regex = re.compile(pattern)

    def fn(completions, **kwargs):
        return [1.0 if regex.search(_text(completion)) else 0.0 for completion in completions]

    fn.__name__ = f"format_reward({pattern!r})"
    return fn


def exact_match_reward(column):
    """1.0 when the stripped completion equals the stripped dataset ``column``."""

    def fn(completions, **kwargs):
        answers = list(kwargs.get(column) or [])
        scores = []
        for index, completion in enumerate(completions):
            if index < len(answers) and _text(completion).strip() == str(answers[index]).strip():
                scores.append(1.0)
            else:
                scores.append(0.0)
        return scores

    fn.__name__ = f"exact_match_reward({column!r})"
    return fn


def contains_reward(text):
    """1.0 when the completion contains ``text``. For simple substring checks."""

    def fn(completions, **kwargs):
        return [1.0 if text in _text(completion) else 0.0 for completion in completions]

    fn.__name__ = f"contains_reward({text!r})"
    return fn


def length_penalty(target):
    """Score falls linearly from 1.0 as the completion length moves from ``target`` characters."""

    def fn(completions, **kwargs):
        return [
            max(0.0, 1.0 - abs(len(_text(completion)) - target) / max(target, 1))
            for completion in completions
        ]

    fn.__name__ = f"length_penalty({target!r})"
    return fn
