from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def mamba_selective_scan(
    x: Tensor,
    dt: Tensor,
    b: Tensor,
    c: Tensor,
    a_log: Tensor,
    d: Tensor,
    gate: Tensor,
    dt_bias: Tensor,
    state: Tensor,
) -> Tensor:
    """Run Mamba selective scan and update ``state`` in-place."""
    return Tensor(
        _infinicore.mamba_selective_scan(
            x._underlying,
            dt._underlying,
            b._underlying,
            c._underlying,
            a_log._underlying,
            d._underlying,
            gate._underlying,
            dt_bias._underlying,
            state._underlying,
        )
    )
