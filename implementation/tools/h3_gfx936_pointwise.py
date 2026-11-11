"""gfx936 BF16 RMSNorm with FP32 accumulation and final BF16 weighting.
Matches the measured PyTorch 2.9 BW reference; all arithmetic uses NineToothed.
"""
import ninetoothed.language as ntl

def rms_apply(x, weight, eps, out):
    value = x
    value = value.to(ntl.float32)
    squared = value * value
    squared = ntl.where(x.offsets(-1) < x.source.shape[-1], squared, 0.)
    mean = (ntl.sum(squared, 0) / x.source.shape[-1])
    epsilon = eps
    epsilon = epsilon.to(ntl.float32)
    variance = mean + epsilon
    reciprocal = ntl.rsqrt(variance)
    normalized = value * reciprocal
    out = normalized * weight

def rms_modulate_apply(x, weight, eps, scale, shift, out):
    value = x
    value = value.to(ntl.float32)
    squared = value * value
    squared = ntl.where(x.offsets(-1) < x.source.shape[-1], squared, 0.)
    mean = (ntl.sum(squared, 0) / x.source.shape[-1])
    epsilon = eps
    epsilon = epsilon.to(ntl.float32)
    variance = mean + epsilon
    reciprocal = ntl.rsqrt(variance)
    normalized = value * reciprocal
    weighted = (normalized * weight).to(x.dtype)
    factor = (1. + scale).to(x.dtype)
    product = (weighted * factor).to(x.dtype)
    out = product + shift

def gated_residual_rms_modulate_apply(x, gate, y, weight, eps, scale, shift, residual, out):
    # `residual = x + gate * y` exactly as `gated_residual_apply`, then the
    # same RMSNorm/AdaLN arithmetic as `rms_modulate_apply` on that residual.
    product = (gate * y).to(x.dtype)
    summed = (x + product).to(x.dtype)
    residual = summed  # noqa: F841
    value = summed.to(ntl.float32)
    squared = value * value
    squared = ntl.where(x.offsets(-1) < x.source.shape[-1], squared, 0.)
    mean = (ntl.sum(squared, 0) / x.source.shape[-1])
    epsilon = eps
    epsilon = epsilon.to(ntl.float32)
    variance = mean + epsilon
    reciprocal = ntl.rsqrt(variance)
    normalized = value * reciprocal
    weighted = (normalized * weight).to(x.dtype)
    factor = (1. + scale).to(x.dtype)
    modulated = (weighted * factor).to(x.dtype)
    out = modulated + shift

def qk_norm_rope_apply(x0, x1, xp, w0, w1, wp, eps, cos, sin, y0, y1, yp):
    # h3_rms_exact (width 128) followed by h3_rope_exact, per head row.
    a = x0
    b = x1
    p = xp
    a = a.to(ntl.float32)
    b = b.to(ntl.float32)
    p = p.to(ntl.float32)
    sa = a * a
    sb = b * b
    sp = p * p
    sa = ntl.where(x0.offsets(-1) < 48, sa, 0.)
    sb = ntl.where(x1.offsets(-1) < 96, sb, 0.)
    sp = ntl.where(xp.offsets(-1) < 128, sp, 0.)
    mean = ((ntl.sum(sa, 1) + ntl.sum(sb, 1) + ntl.sum(sp, 1)) / 128)
    epsilon = eps
    epsilon = epsilon.to(ntl.float32)
    variance = mean + epsilon
    reciprocal = ntl.rsqrt(variance)
    na = (a * reciprocal[:, None] * w0).to(x0.dtype).to(ntl.float32)
    nb = (b * reciprocal[:, None] * w1).to(x0.dtype).to(ntl.float32)
    npass = (p * reciprocal[:, None] * wp).to(x0.dtype)
    c = cos
    s = sin
    c = c.to(ntl.float32)
    s = s.to(ntl.float32)
    ac = (na * c).to(x0.dtype).to(ntl.float32)
    bs = (-nb * s).to(x0.dtype).to(ntl.float32)
    bc = (nb * c).to(x0.dtype).to(ntl.float32)
    ass = (na * s).to(x0.dtype).to(ntl.float32)
    y0 = ac + bs  # noqa: F841
    y1 = bc + ass  # noqa: F841
    yp = npass

def qk_norm_rope_ranks_apply(x0, x1, xp, w0, w1, wp, eps, cos, sin, y0, y1, yp):
    # Same arithmetic as qk_norm_rope_apply on a [ranks, heads, 64] block of
    # one token: per-head reductions over the last axis, angles broadcast.
    a = x0
    b = x1
    p = xp
    a = a.to(ntl.float32)
    b = b.to(ntl.float32)
    p = p.to(ntl.float32)
    sa = a * a
    sb = b * b
    sp = p * p
    sa = ntl.where(x0.offsets(-1) < 48, sa, 0.)
    sb = ntl.where(x1.offsets(-1) < 96, sb, 0.)
    sp = ntl.where(xp.offsets(-1) < 128, sp, 0.)
    mean = ((ntl.sum(sa, 2) + ntl.sum(sb, 2) + ntl.sum(sp, 2)) / 128)
    epsilon = eps
    epsilon = epsilon.to(ntl.float32)
    variance = mean + epsilon
    reciprocal = ntl.rsqrt(variance)
    na = (a * reciprocal[:, :, None] * w0).to(x0.dtype).to(ntl.float32)
    nb = (b * reciprocal[:, :, None] * w1).to(x0.dtype).to(ntl.float32)
    npass = (p * reciprocal[:, :, None] * wp).to(x0.dtype)
    c = cos
    s = sin
    c = c.to(ntl.float32)
    s = s.to(ntl.float32)
    ac = (na * c).to(x0.dtype).to(ntl.float32)
    bs = (-nb * s).to(x0.dtype).to(ntl.float32)
    bc = (nb * c).to(x0.dtype).to(ntl.float32)
    ass = (na * s).to(x0.dtype).to(ntl.float32)
    y0 = ac + bs  # noqa: F841
    y1 = bc + ass  # noqa: F841
    yp = npass

def qk_norm_rope_block_apply(x0, x1, xp, w0, w1, wp, eps, cos, sin, y0, y1, yp):
    # Same arithmetic as qk_norm_rope_apply on a [tokens, heads, 64] block:
    # the per-head reductions run over the last axis and the per-token angles
    # broadcast over the heads.
    a = x0
    b = x1
    p = xp
    a = a.to(ntl.float32)
    b = b.to(ntl.float32)
    p = p.to(ntl.float32)
    sa = a * a
    sb = b * b
    sp = p * p
    sa = ntl.where(x0.offsets(-1) < 48, sa, 0.)
    sb = ntl.where(x1.offsets(-1) < 96, sb, 0.)
    sp = ntl.where(xp.offsets(-1) < 128, sp, 0.)
    mean = ((ntl.sum(sa, 2) + ntl.sum(sb, 2) + ntl.sum(sp, 2)) / 128)
    epsilon = eps
    epsilon = epsilon.to(ntl.float32)
    variance = mean + epsilon
    reciprocal = ntl.rsqrt(variance)
    na = (a * reciprocal[:, :, None] * w0).to(x0.dtype).to(ntl.float32)
    nb = (b * reciprocal[:, :, None] * w1).to(x0.dtype).to(ntl.float32)
    npass = (p * reciprocal[:, :, None] * wp).to(x0.dtype)
    c = cos
    s = sin
    c = c.to(ntl.float32)[:, None, :]
    s = s.to(ntl.float32)[:, None, :]
    ac = (na * c).to(x0.dtype).to(ntl.float32)
    bs = (-nb * s).to(x0.dtype).to(ntl.float32)
    bc = (nb * c).to(x0.dtype).to(ntl.float32)
    ass = (na * s).to(x0.dtype).to(ntl.float32)
    y0 = ac + bs  # noqa: F841
    y1 = bc + ass  # noqa: F841
    yp = npass

APPLICATIONS = {
    "gated_residual_rms_modulate_apply": gated_residual_rms_modulate_apply,
    "qk_norm_rope_apply": qk_norm_rope_apply,
    "qk_norm_rope_block_apply": qk_norm_rope_block_apply,
    "qk_norm_rope_ranks_apply": qk_norm_rope_ranks_apply,
    "rms_apply": rms_apply,
    "rms_modulate_apply": rms_modulate_apply,
}
