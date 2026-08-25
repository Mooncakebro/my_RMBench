# Re-export the RMBench policy contract so that
# script/eval_policy.py's `importlib.import_module(policy_name)` +
# `getattr(module, "get_model"/"eval"/"reset_model")` works with
# policy_name: CompactMoDE.
from .deploy_policy import eval, get_model, reset_model  # noqa: F401
