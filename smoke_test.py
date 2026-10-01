"""
Smoke tests for sked_core.py.

These check that the implementation is *internally consistent*: masks are
causal, incremental decoding matches a full forward pass, quantization is
actually ternary, and gradients flow. They say nothing about model quality --
the model has never been trained.

Run:  python smoke_test.py
"""

import torch
import torch.nn.functional as F

from sked_core import (
    SKEDModel,
    BitLinear158,
    count_parameters,
    estimate_static_bytes,
)

torch.manual_seed(0)
FAILURES = []


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except AssertionError as e:
        FAILURES.append(name)
        print(f"  FAIL  {name}: {e}")
    except Exception as e:  # noqa: BLE001
        FAILURES.append(name)
        print(f"  ERROR {name}: {type(e).__name__}: {e}")


def tiny_model(**overrides):
    cfg = dict(vocab_size=512, dim=128, num_heads=4, enc_layers=2, dec_layers=2,
               enc_ffn=256, moe_hidden=192, num_experts=8, top_k=2, window_size=8)
    cfg.update(overrides)
    return SKEDModel(**cfg)


# ---------------------------------------------------------------------------
# 1. quantization
# ---------------------------------------------------------------------------
def test_bitlinear_is_exactly_ternary():
    layer = BitLinear158(16, 8)
    x = torch.randn(4, 16)

    w = layer.weight
    gamma = w.abs().mean().clamp(min=layer.eps)
    w_q = torch.clamp(torch.round(w / gamma), -1.0, 1.0)
    beta = x.abs().amax(dim=-1, keepdim=True).clamp(min=layer.eps)
    x_q = torch.clamp(torch.round(x * 127.0 / beta), -128.0, 127.0)
    expected = F.linear(x_q, w_q) * (beta * gamma / 127.0)

    got = layer(x)
    assert torch.allclose(got, expected, atol=1e-5), "forward is not the quantized computation"
    assert set(w_q.unique().tolist()) <= {-1.0, 0.0, 1.0}, "weights are not ternary"


def test_bitlinear_gradients_flow():
    layer = BitLinear158(32, 16)
    layer(torch.randn(4, 32)).pow(2).mean().backward()
    g = layer.weight.grad
    assert g is not None and torch.isfinite(g).all(), "non-finite weight gradient"
    assert g.abs().sum() > 0, "gradient is identically zero (STE mask may be dead)"


# ---------------------------------------------------------------------------
# 2. shapes / loss
# ---------------------------------------------------------------------------
def test_forward_and_backward_shapes():
    m = tiny_model()
    ids = torch.randint(0, 512, (2, 16))
    logits = m(ids)
    assert logits.shape == (2, 16, 512), f"bad logits shape {tuple(logits.shape)}"

    loss = F.cross_entropy(logits[:, :-1].reshape(-1, 512), ids[:, 1:].reshape(-1))
    loss.backward()
    assert torch.isfinite(loss), "non-finite loss"
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert len(grads) > 0 and all(torch.isfinite(g).all() for g in grads), "non-finite gradients"


def test_no_nan_in_outputs():
    m = tiny_model()
    logits = m(torch.randint(0, 512, (2, 32)))
    assert torch.isfinite(logits).all(), "NaN/Inf in logits"


# ---------------------------------------------------------------------------
# 3. causality  (the property the earlier drafts violated)
# ---------------------------------------------------------------------------
def test_causality_no_future_leak():
    m = tiny_model().eval()
    ids = torch.randint(0, 512, (1, 24))
    perturbed = ids.clone()
    cut = 12
    perturbed[:, cut + 1:] = torch.randint(0, 512, perturbed[:, cut + 1:].shape)

    with torch.no_grad():
        a = m(ids)
        b = m(perturbed)

    # positions <= cut must be unaffected by tokens after `cut`
    assert torch.allclose(a[:, : cut + 1], b[:, : cut + 1], atol=1e-5), \
        "future tokens leaked into earlier positions"


def test_causality_holds_beyond_window():
    """With sliding-window attention the same property must still hold."""
    m = tiny_model(window_size=4).eval()
    ids = torch.randint(0, 512, (1, 20))
    perturbed = ids.clone()
    perturbed[:, 15:] = torch.randint(0, 512, perturbed[:, 15:].shape)
    with torch.no_grad():
        a, b = m(ids), m(perturbed)
    assert torch.allclose(a[:, :15], b[:, :15], atol=1e-5), "leak past window boundary"


# ---------------------------------------------------------------------------
# 4. incremental decode == full forward
# ---------------------------------------------------------------------------
def test_incremental_decode_matches_full_forward():
    m = tiny_model().eval()
    ids = torch.randint(0, 512, (1, 12))
    n_new = 6

    with torch.no_grad():
        generated = m.generate(ids, max_new_tokens=n_new)
    assert generated.shape == (1, 12 + n_new), f"bad generate shape {tuple(generated.shape)}"

    # replay: the greedy token at each step must match a fresh full forward pass
    with torch.no_grad():
        for i in range(n_new):
            prefix = generated[:, : 12 + i]
            ref = m(prefix)[:, -1, :].argmax(dim=-1)
            assert ref.item() == generated[0, 12 + i].item(), (
                f"step {i}: incremental decode ({generated[0, 12 + i].item()}) "
                f"!= full forward ({ref.item()})"
            )


def test_decode_respects_window():
    """Long sequences must still decode correctly once the cache saturates."""
    m = tiny_model(window_size=4).eval()
    ids = torch.randint(0, 512, (1, 16))
    with torch.no_grad():
        generated = m.generate(ids, max_new_tokens=5)
        ref = m(generated[:, :20])[:, -1, :].argmax(dim=-1)
    assert ref.item() == generated[0, 20].item(), "cache truncation broke decoding"


# ---------------------------------------------------------------------------
# 5. MoE
# ---------------------------------------------------------------------------
def test_moe_load_balancing_moves_bias():
    m = tiny_model()
    m.train()
    ids = torch.randint(0, 512, (4, 32))
    before = m.decoders[0].ffn.expert_bias.clone()
    for _ in range(5):
        F.cross_entropy(m(ids)[:, :-1].reshape(-1, 512), ids[:, 1:].reshape(-1)).backward()
        m.zero_grad(set_to_none=True)
    after = m.decoders[0].ffn.expert_bias
    assert not torch.allclose(before, after), "balancing bias never updated"


# ---------------------------------------------------------------------------
# 6. accounting
# ---------------------------------------------------------------------------
def test_parameter_accounting_matches_spec():
    """The 10.35B / 2.59B figures in the README, recomputed at full config.

    Built on the meta device: the full config is 10.35B params, which would be
    ~41 GB if materialised in fp32. Only shapes are needed for accounting.
    """
    with torch.device("meta"):
        m = SKEDModel()  # defaults == spec config
    total = count_parameters(m)
    print(f"        total params      : {total/1e9:.3f} B")

    mem = estimate_static_bytes(m)
    print(f"        ternary footprint : {mem['ternary_bytes']/2**30:.3f} GiB")
    print(f"        other  footprint  : {mem['other_bytes']/2**30:.3f} GiB")
    print(f"        static total      : {mem['total_bytes']/2**30:.3f} GiB")

    assert 10.0e9 < total < 10.7e9, f"total params {total/1e9:.2f}B outside expected band"


if __name__ == "__main__":
    print("SKED smoke tests\n")
    check("bitlinear forward is exactly ternary", test_bitlinear_is_exactly_ternary)
    check("bitlinear gradients flow", test_bitlinear_gradients_flow)
    check("forward/backward shapes + loss", test_forward_and_backward_shapes)
    check("no NaN/Inf in logits", test_no_nan_in_outputs)
    check("causality: no future leak", test_causality_no_future_leak)
    check("causality: holds beyond window", test_causality_holds_beyond_window)
    check("incremental decode == full forward", test_incremental_decode_matches_full_forward)
    check("decode respects window truncation", test_decode_respects_window)
    check("moe balancing bias updates", test_moe_load_balancing_moves_bias)
    check("parameter accounting matches spec", test_parameter_accounting_matches_spec)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} test(s) failed: {', '.join(FAILURES)}")
        raise SystemExit(1)
    print("all tests passed")