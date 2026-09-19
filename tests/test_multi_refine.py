"""
Unit tests for full-sequence multi-refinement training (--multi-refine).

python -m pytest tests/test_multi_refine.py -v
"""

import pytest
import torch

import nanochat.flash_attention as fa_module
from nanochat.gpt import GPT, GPTConfig

# These tests run tiny models on CPU: force the SDPA fallback (FA3 is CUDA-only).
# Set per-test (not at import) because other test modules reset the override.
@pytest.fixture(autouse=True)
def force_sdpa():
    prev = fa_module._override_impl
    fa_module._override_impl = 'sdpa'
    yield
    fa_module._override_impl = prev


def build_model(seed=0, **overrides):
    """Tiny CPU model with the feedback architecture enabled via use_multi_refine."""
    kwargs = dict(
        sequence_len=32, vocab_size=64, n_layer=2, n_head=2, n_kv_head=2,
        n_embd=32, window_pattern="L", use_multi_refine=True, multi_refine_num_passes=2,
    )
    kwargs.update(overrides)
    config = GPTConfig(**kwargs)
    torch.manual_seed(seed)
    model = GPT(config, pad_vocab_size_to=64)
    model.init_weights()
    # init_weights zeros c_proj / mlp.c_proj (and the gates), which kills most gradient
    # paths at exact init; randomize them so gradient-flow assertions are meaningful.
    with torch.no_grad():
        for block in model.transformer.h:
            torch.nn.init.normal_(block.attn.c_proj.weight, std=0.02)
            torch.nn.init.normal_(block.mlp.c_proj.weight, std=0.02)
    return model


def make_batch(model, B=2, T=16, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, model.config.vocab_size, (B, T), generator=g)
    y = torch.randint(0, model.config.vocab_size, (B, T), generator=g)
    return x, y


def test_output_structure():
    model = build_model()
    x, y = make_batch(model)
    out = model.forward_multi_refine_train(x, y)
    n_passes = 1 + model.config.multi_refine_num_passes
    assert len(out['per_pass_losses']) == n_passes
    assert out['per_pass_masks'] == [None] * n_passes  # every pass covers all positions
    assert out['loss'].ndim == 0 and torch.isfinite(out['loss'])
    assert not out['h_final'].requires_grad  # returned feedback is detached
    # equal-weight aggregation
    expected = sum(out['per_pass_losses']) / n_passes
    assert torch.allclose(out['loss'], expected)


def test_pass0_matches_plain_zero_feedback_forward():
    model = build_model()
    x, y = make_batch(model)
    out = model.forward_multi_refine_train(x, y)
    zero_fb = torch.zeros(x.size(0), x.size(1), model.config.n_embd)
    loss_plain, _ = model(x, y, h_final_prev=zero_fb)
    assert torch.allclose(out['per_pass_losses'][0], loss_plain, atol=1e-6)


def test_multi_refine_passes_differ_from_pass0():
    model = build_model()
    x, y = make_batch(model)
    out = model.forward_multi_refine_train(x, y)
    # nonzero feedback must change the loss (γ injection is 0.1 at init, gates ~1)
    assert not torch.allclose(out['per_pass_losses'][0], out['per_pass_losses'][1])


def test_loss_weights():
    weights = (0.5, 1.0, 2.0)
    model = build_model(multi_refine_loss_weights=weights)
    x, y = make_batch(model)
    out = model.forward_multi_refine_train(x, y)
    expected = sum(w * l for w, l in zip(weights, out['per_pass_losses'])) / sum(weights)
    assert torch.allclose(out['loss'], expected)


def test_loss_weights_length_validated():
    with pytest.raises(AssertionError):
        GPTConfig(sequence_len=32, vocab_size=64, n_layer=2, n_head=2, n_kv_head=2,
                  n_embd=32, use_multi_refine=True, multi_refine_num_passes=2, multi_refine_loss_weights=(1.0, 1.0))


def test_mutually_exclusive_with_interleaved():
    with pytest.raises(AssertionError):
        GPTConfig(sequence_len=32, vocab_size=64, n_layer=2, n_head=2, n_kv_head=2,
                  n_embd=32, use_multi_refine=True, use_interleaved=True)


def test_gradients_flow_to_feedback_params():
    model = build_model()
    x, y = make_batch(model)
    out = model.forward_multi_refine_train(x, y)
    out['loss'].backward()
    for proj in model.interleaved_projections.values():
        assert proj.W_v.weight.grad is not None and proj.W_v.weight.grad.abs().sum() > 0
    assert model.interleaved_lambdas.grad is not None and model.interleaved_lambdas.grad.abs().sum() > 0


def test_detach_blocks_cross_pass_gradients():
    # Weight ONLY the last pass. Without detach, its loss reaches earlier passes through
    # h_final, so dL/d(pass-0 activations) exists; with detach, earlier passes only run
    # under no-grad-needed leaves... simplest observable: compare wte grads. With detach
    # the last pass still uses wte directly, so instead compare against the no-detach
    # model: gradients must differ because the cross-pass paths are severed.
    x = None
    grads = {}
    for detach in (False, True):
        model = build_model(multi_refine_detach=detach, multi_refine_loss_weights=(0.0, 0.0, 1.0))
        if x is None:
            x, y = make_batch(model)
        out = model.forward_multi_refine_train(x, y)
        out['loss'].backward()
        grads[detach] = model.transformer.wte.weight.grad.clone()
    assert not torch.allclose(grads[False], grads[True])


def test_evaluate_bpb_headline_is_final_pass():
    from nanochat.loss_eval import evaluate_bpb
    model = build_model()
    model.eval()
    x, y = make_batch(model)
    token_bytes = torch.ones(model.config.vocab_size, dtype=torch.int64)
    with torch.no_grad():
        result = evaluate_bpb(model, iter([(x, y)]), 1, token_bytes)
    assert set(result) == {'bpb', 'per_pass_bpb'}
    assert len(result['per_pass_bpb']) == 1 + model.config.multi_refine_num_passes
    # every pass covers all positions, so the headline bpb is just the final pass
    assert result['bpb'] == pytest.approx(result['per_pass_bpb'][-1], rel=1e-5)
