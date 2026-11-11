"""Host lifecycle tests; real model output/memory checks live in verify_conditioner.py."""
from contextlib import nullcontext
import threading
import weakref

import pytest
import infinilm.multimodal._h3_residency as residency


@pytest.fixture
def host_runtime(monkeypatch):
    loads = []

    class Engine:
        dtype = "bf16"

        def __init__(self, path, **kwargs):
            self.path = path
            loads.append(path)

    monkeypatch.setattr(residency, "InferEngine", Engine)
    monkeypatch.setattr(residency, "load_model_state_dict_by_file", lambda *a, **k: None)
    monkeypatch.setattr(residency.torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(residency.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(residency.torch.cuda, "device", lambda *a: nullcontext())
    monkeypatch.setattr(residency.infinicore, "get_device", lambda: None)
    monkeypatch.setattr(residency.infinicore, "set_device", lambda *a: None)
    monkeypatch.setattr(residency.infinicore.device, "_from_infinicore_device", lambda *a: None)
    monkeypatch.setattr(residency.infinicore, "trim_memory", lambda: None)
    monkeypatch.setattr(residency, "_sessions", weakref.WeakSet())
    return loads


def test_two_generations_reload_conditioner_and_transformer(host_runtime):
    qwen = residency.H3EngineResidency("qwen", devices=[0, 1], sequential=True, release_after=True)
    dit = residency.H3EngineResidency("dit", devices=[0, 1], sequential=True)
    assert host_runtime == []
    for _ in range(2):
        with qwen.use():
            assert dit.engine is None
            qwen_ref = weakref.ref(qwen.engine)
        assert qwen_ref() is None
        for _ in range(3):
            with dit.use():
                assert dit.engine.path == "dit"
                assert qwen.engine is None
    assert host_runtime == ["qwen", "dit", "qwen", "dit"]


@pytest.mark.parametrize("release_after", [True, False])
def test_failure_releases_engine_and_allows_retry(host_runtime, release_after):
    qwen = residency.H3EngineResidency("qwen", devices=[0], sequential=True, release_after=release_after)
    with pytest.raises(ValueError):
        with qwen.use():
            raise ValueError("bad request")
    assert qwen.engine is None
    with qwen.use():
        assert qwen.engine is not None
    assert host_runtime == ["qwen", "qwen"]


def test_live_forward_cannot_be_evicted(host_runtime):
    first = residency.H3EngineResidency("first", devices=[0, 1], sequential=True)
    second = residency.H3EngineResidency("second", devices=[1], sequential=True)
    attempted, entered = threading.Event(), threading.Event()

    def other_request():
        attempted.set()
        with second.use():
            entered.set()

    with first.use():
        worker = threading.Thread(target=other_request, daemon=True)
        worker.start()
        assert attempted.wait(2)
        assert not entered.wait(0.1)
        assert first.engine is not None
    worker.join(5)
    assert not worker.is_alive()
    assert entered.is_set() and first.engine is None


def test_disjoint_devices_keep_weights(host_runtime):
    first = residency.H3EngineResidency("first", devices=[0], sequential=True)
    second = residency.H3EngineResidency("second", devices=[1], sequential=True)
    with first.use():
        pass
    with second.use():
        assert first.engine is not None
    assert host_runtime == ["first", "second"]
