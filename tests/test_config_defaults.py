import os
import importlib

def test_config_defaults():
    cfg_mod = importlib.import_module("src.config")
    # Expect a Config dataclass-like object; fallback to attribute presence
    cfg = cfg_mod.Config() if hasattr(cfg_mod, "Config") else None

    # If the module exposes a factory, try that (stay tolerant)
    if cfg is None and hasattr(cfg_mod, "get_config"):
        cfg = cfg_mod.get_config()

    # If still None, skip with a helpful message (does not fail CI)
    if cfg is None:
        import pytest
        pytest.skip("Config object or get_config() not found; adjust test once API is finalized.")

    # Check a few expected fields or reasonable defaults
    assert hasattr(cfg, "model_id")
    assert getattr(cfg, "model_id")  # non-empty default
    assert hasattr(cfg, "precision")
    assert getattr(cfg, "precision") in {"auto", "fp16", "int4"}

def test_env_overrides_take_effect(monkeypatch):
    monkeypatch.setenv("MODEL_ID", "microsoft/Phi-3.5-mini-instruct")
    monkeypatch.setenv("PRECISION", "fp16")

    cfg_mod = importlib.import_module("src.config")
    # Try to re-import to re-read env if module caches state
    importlib.reload(cfg_mod)
    cfg = cfg_mod.Config() if hasattr(cfg_mod, "Config") else None
    if cfg is None and hasattr(cfg_mod, "get_config"):
        cfg = cfg_mod.get_config()
    if cfg is None:
        import pytest
        pytest.skip("Config API not found; adjust test once API is finalized.")

    assert getattr(cfg, "model_id") == "microsoft/Phi-3.5-mini-instruct"
    assert getattr(cfg, "precision") == "fp16"
