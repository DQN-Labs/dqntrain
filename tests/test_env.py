import dqntrain as dq
from dqntrain.env import env_info, env_report


def test_env_info_has_expected_keys():
    info = env_info()
    for key in ("python", "torch", "unsloth", "bitsandbytes", "trl", "transformers", "dqntrain"):
        assert key in info
    assert info["dqntrain"] == dq.__version__
    assert info["python"]


def test_env_report_prints_and_returns(capsys):
    info = env_report()
    out = capsys.readouterr().out
    assert "python:" in out
    assert "dqntrain: 1.0.0" in out
    assert info["trl"]
