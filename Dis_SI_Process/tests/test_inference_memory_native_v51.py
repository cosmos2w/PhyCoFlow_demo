from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "Dis_SI_Process/scripts/benchmark_inference_memory_native_v51.py"


def load_module():
    spec = importlib.util.spec_from_file_location("benchmark_inference_memory_native_v51", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_call_requires_disabled_gradients():
    module = load_module()

    class DummyOutput:
        family = "dummy"

    try:
        module.run_call(DummyOutput(), {}, {}, "dummy")
    except RuntimeError as exc:
        assert "gradients enabled" in str(exc)
    else:
        raise AssertionError("run_call accepted an inference invocation with gradients enabled")


def test_unique_tensor_bytes_deduplicates_shared_storage():
    module = load_module()
    tensor = torch.ones(8, dtype=torch.float32)
    assert module.unique_tensor_bytes([tensor, tensor.view(2, 4)]) == tensor.untyped_storage().nbytes()

