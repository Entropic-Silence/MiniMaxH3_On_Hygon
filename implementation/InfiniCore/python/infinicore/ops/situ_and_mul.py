from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def situ_and_mul(gate, up, beta=4.0, linear_beta=25.0, *, out=None):
    beta = float(beta)
    linear_beta = float(linear_beta)
    if out is None:
        return Tensor(
            _infinicore.situ_and_mul(
                gate._underlying,
                up._underlying,
                beta,
                linear_beta,
            )
        )

    _infinicore.situ_and_mul_(
        out._underlying,
        gate._underlying,
        up._underlying,
        beta,
        linear_beta,
    )
    return out
