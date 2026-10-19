# Rewards

## Contract

TRL calls each reward function with `completions` and dataset columns as
keyword arguments. A reward function must accept `completions` and `**kwargs`,
and return one float per completion. Completions may be strings or chat-format
lists of messages.

```python
def my_reward(completions, **kwargs):
    return [1.0 if should_score(c) else 0.0 for c in completions]
```

## Built-in helpers

`dqntrain.rewards` provides four helpers that follow the contract:

| Helper | Returns |
|---|---|
| `format_reward(pattern)` | 1.0 when a completion matches a regex, else 0.0 |
| `exact_match_reward(column)` | 1.0 when the stripped completion equals the stripped dataset column |
| `contains_reward(text)` | 1.0 when a completion contains the given text |
| `length_penalty(target)` | a score that falls as the completion length moves away from `target` characters |

```python
from dqntrain.rewards import format_reward, contains_reward

grpo = dq.GRPO("grpo-code",
    data="org/code-problems",
    reward_funcs=[format_reward(r"\\boxed\{.*\}"), contains_reward("\\boxed")],
    reward_weights=[0.2, 1.0],
    ...
)
```

Every built-in uses the `_text(completion)` helper, which extracts the string
from chat-format completions (the last assistant message) or returns the
string unchanged.

## Hashing

`reward_funcs` and `reward_weights` are part of the GRPO config hash. For each
function, the source is hashed with `inspect.getsource`. When source is
unavailable, for example for some notebook lambdas, the bytecode
(`co_code` and `co_consts`) is hashed instead and `summary()`/`plan()` print a
warning naming the function.

## Security warning

**Reward functions run in the training process** and have full access to it:
files, network, environment variables, and GPU memory.

If your reward code executes model-generated code — for example, running
generated Python to check whether it passes tests — do **not** call `exec` or
`eval` on model output inside the reward function. Run that code in a
subprocess with a timeout and resource limits, and return a score based on the
subprocess result. Sandboxing of reward code is out of scope for v1, so this
is the caller's responsibility.
