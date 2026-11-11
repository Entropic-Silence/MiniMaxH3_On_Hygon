"""Conservative compile-only fallback for DAS Triton 3.5.1's older GCVM parser.

This leaves the generated arithmetic intact. The optional LLVM optimization
round trip is disabled, and GEP no-wrap promises unsupported by GCVM are dropped.
Correctness and performance must be measured before using resulting artifacts.
"""
import re


def normalize_llvm_ir(source):
    def gep(match):
        return " ".join(word for word in match.group().split() if word not in {"nuw", "nusw"})
    return re.sub(r"\bgetelementptr(?:\s+(?:inbounds|nuw|nusw))+", gep, source)


def install():
    import triton
    import sys
    print("COMPILER_PROCESS", sys.executable, sys.version, flush=True)
    print("TRITON_MODULE", getattr(triton, "__file__", None), getattr(triton, "__version__", None),
          list(getattr(triton, "__path__", [])), flush=True)
    from triton._C.libtriton import llvm
    from triton.backends.amd.compiler import HIPBackend
    if getattr(HIPBackend, "_ifab_compat", False):
        return
    if triton.__version__ != "3.5.1":
        raise RuntimeError("This compatibility path is scoped to DAS Triton 3.5.1")
    original = HIPBackend.make_amdgcn

    def assembly(source, metadata, options):
        return original(normalize_llvm_ir(source), metadata, options)

    # This vendor binding has no parse_ir/module replacement interface. Its
    # optimize_module round trip passes newer LLVM text to the older GCVM parser.
    llvm.optimize_module = lambda *args, **kwargs: None
    HIPBackend.make_amdgcn = staticmethod(assembly)
    HIPBackend._ifab_compat = True
    print("IFAB_GCVM_COMPAT llvm_opt=disabled gep_no_wrap=dropped", flush=True)
