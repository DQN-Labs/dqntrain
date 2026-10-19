"""Error types raised by dqntrain."""


def _short(value, limit=80):
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


class LockHeldError(Exception):
    def __init__(self, holder, heartbeat):
        super().__init__(
            f"run is locked by {holder!r} (heartbeat {heartbeat}); "
            "retry once the heartbeat is older than 15 minutes"
        )
        self.holder = holder
        self.heartbeat = heartbeat


class ConfigChangedError(Exception):
    def __init__(self, stage, changes):
        details = "; ".join(f"{key}: {_short(old)} -> {_short(new)}" for key, (old, new) in sorted(changes.items()))
        super().__init__(
            f"config for stage {stage!r} changed; call rerun({stage!r}) to accept it. "
            f"changed keys: {details}"
        )
        self.stage = stage
        self.changes = changes


class DataColumnError(Exception):
    def __init__(self, stage_kind, required, missing, found):
        super().__init__(
            f"{stage_kind} data is missing required columns. "
            f"required: {required}; missing: {missing}; found: {found}"
        )
        self.required = required
        self.missing = missing
        self.found = found


class DuplicateStageError(Exception):
    def __init__(self, name):
        super().__init__(f"duplicate stage name: {name!r}")
        self.name = name
