# Data pipeline

`load_stage_data(data_config, stage_kind)` runs these steps in this fixed
order:

1. Load the dataset with `datasets.load_dataset` using `name`, `subset`,
   `split`, and `revision`.
2. Shuffle with `seed`.
3. Apply `limit`.
4. Apply `filter` (a function taking one example, returning `bool`).
5. Rename `columns`.
6. Apply `map` (a function taking one example, returning a dict).
7. Check required columns for the stage kind. Fail fast with
   `DataColumnError` that lists the required, missing, and found columns.
8. Build the eval split from `eval`, if given.

Step 7 runs before the model loads, so mistakes fail in seconds. `summary()`
runs this whole pipeline without loading a model, which makes it the cheapest
place to catch dataset problems.

## Required columns

| Stage | Required | Notes |
|---|---|---|
| `SFT` | `messages` or `text` | `format` controls templating |
| `DPO` | `prompt`, `chosen`, `rejected` | Used as given |
| `GRPO` | `prompt` | Other columns pass through to reward functions |
| `Eval` | none | Uses `tasks` |
| `Custom` | none | Your function decides |
| `Merge` | none | No data |

## Column renaming

`columns` maps expected names to source names:

```python
dq.Data("org/code-problems", columns={"prompt": "question", "answer": "solution"})
```

renames the dataset's `question` to `prompt` and `solution` to `answer`. This
is the mapping the required-column check and the rest of the pipeline see.

## Errors

A dataset that cannot satisfy a stage raises:

```
DataColumnError: GRPO data is missing required columns.
required: [['prompt']]; missing: ['prompt']; found: ['question', 'solution']
```

## Eval splits

There is no automatic held-out split in v1. To build one, describe it
explicitly:

```python
data = dq.Data("org/train-set", eval={"name": "org/eval-set", "split": "test", "limit": 200})
```

`eval` accepts a dict or a `Data` instance and supports its own `name`,
`split`, and `limit`. The eval dataset skips required-column checks.
