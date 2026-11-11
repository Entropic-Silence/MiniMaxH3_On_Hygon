#!/usr/bin/env python3
"""Smoke test for the Mutual Awareness Analyzer Python bindings.

Builds with `xmake f --mutual-awareness=y` are expected to expose the
`infinicore.analyzer` submodule. Builds without the option simply have it
absent and this script will skip.
"""

from __future__ import annotations

import sys


def main() -> int:
    # Load _infinicore.so directly to avoid pulling torch and other heavy
    # dependencies through the infinicore Python wrapper. This keeps the
    # smoke test focused on analyzer/dispatch bindings.
    import glob
    import importlib.util
    import os
    candidates = []
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.normpath(os.path.join(here, "..", ".."))
    for pattern in (
        os.path.join(repo_root, "python/infinicore/lib/_infinicore*.so"),
        os.path.join(repo_root, "build/*/release/_infinicore*.so"),
    ):
        candidates.extend(glob.glob(pattern))
    if not candidates:
        try:
            from infinicore.lib import _infinicore  # noqa: F401
        except ImportError as exc:
            print(f"[ERROR] cannot locate _infinicore.so: {exc}")
            return 1
    else:
        spec = importlib.util.spec_from_file_location("_infinicore", candidates[0])
        _infinicore = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_infinicore)

    if not hasattr(_infinicore, "analyzer"):
        print(
            "[SKIP] mutual-awareness build option is off (analyzer submodule "
            "missing). Re-configure with --mutual-awareness=y to run."
        )
        return 0

    analyzer = _infinicore.analyzer

    failed = 0

    def expect(name: str, cond: bool, hint: str = "") -> None:
        nonlocal failed
        if cond:
            print(f"  [PASS] {name}")
        else:
            print(f"  [FAIL] {name} {hint}")
            failed += 1

    print("=== Section 1: analyzer enums round-trip ===")
    expect("PhaseType has DECODE", hasattr(analyzer.PhaseType, "DECODE"))
    expect("OptimizationGoal has LATENCY_FIRST",
           hasattr(analyzer.OptimizationGoal, "LATENCY_FIRST"))
    expect("OpType has FLASH_ATTENTION", hasattr(analyzer.OpType, "FLASH_ATTENTION"))

    print("\n=== Section 2: trace + analyze ===")
    analyzer.set_enabled(True)
    analyzer.clear_trace()
    analyzer.trace_op_for_test(
        analyzer.OpType.ATTENTION,
        [1, 32, 1],
        dtype=0,
        device_type=0,
        device_id=0,
    )
    analyzer.trace_op_for_test(
        analyzer.OpType.FLASH_ATTENTION,
        [1, 32, 1],
        dtype=0,
        device_type=0,
        device_id=0,
    )
    phase = analyzer.get_current_phase()
    expect("decode shape (seq_len=1) detected as DECODE",
           phase == analyzer.PhaseType.DECODE,
           f"got: {phase}")

    intent = analyzer.analyze()
    expect("analyze() returns OptimizationIntent",
           hasattr(intent, "global_intent") and hasattr(intent, "per_device"))
    expect("decode goal -> LATENCY_FIRST",
           intent.global_intent.goal == analyzer.OptimizationGoal.LATENCY_FIRST)
    analyzer.clear_trace()

    print("\n" + "=" * 50)
    print(f"Result: {failed} failure(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
