from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def fused_gated_delta_net_gating(
    A_log: Tensor,
    a: Tensor,
    b: Tensor,
    dt_bias: Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
    *,
    out: tuple[Tensor, Tensor] | None = None,
) -> tuple[Tensor, Tensor]:
    if out is None:
        g, beta_output = _infinicore.fused_gated_delta_net_gating(
            A_log._underlying,
            a._underlying,
            b._underlying,
            dt_bias._underlying,
            beta,
            threshold,
        )
        return Tensor(g), Tensor(beta_output)

    g, beta_output = out
    _infinicore.fused_gated_delta_net_gating_(
        g._underlying,
        beta_output._underlying,
        A_log._underlying,
        a._underlying,
        b._underlying,
        dt_bias._underlying,
        beta,
        threshold,
    )
    return g, beta_output
