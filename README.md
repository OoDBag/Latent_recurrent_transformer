# Latent Recurrent Transformer (LRT) — Parallel Interleaved Training

This repo implements **parallel interleaved training** for a latent-recurrent Transformer: the model's final hidden state is fed back into the network as recurrent input for later tokens, and this recurrence is trained efficiently with one full forward pass plus a few cheap partial ("interleaved") passes — instead of the T sequential passes that naive token-level recurrence would require.

The codebase is a fork of [karpathy/nanochat](https://github.com/karpathy/nanochat), keeping its minimal single-node training harness (tokenizer, pretraining, BPB/CORE evaluation) and adding the LRT mechanism on top.

## Method

**Recurrence.** At every layer, the final hidden state `h_final` from the previous position (shifted right by one, so token `t` only sees `t-1` — causality is preserved) is fed back into the network three ways:

1. **Residual injection**: added to the residual stream with a learned per-layer scale γ (`interleaved_lambdas`).
2. **Value feedback**: projected into V space by a per-layer linear map.
3. **Key feedback**: projected into K space by a per-layer linear map.

A learned **triple gate** mixes the value channels per KV head:

```
v = gate_local · v_local + gate_fb · v_feedback + gate_ve · v_init_embed
```

where `v_init_embed` is a token-indexed init value embedding (the model's "no feedback yet" prior).

**Parallel interleaved training.** Running true recurrence during training would need T sequential forward passes. Parallel interleaved training approximates it:

- **Pass 0 (full pass)**: a normal parallel forward over all positions with zero feedback, capturing every layer's K,V into a buffer.
- **Interleaved passes 1..N**: pass `i` recomputes only the positions with `pos % N == i`, using the latest merged `h_final` as feedback. Fresh K,V are written back into the shared buffer, and each pass's `h_final` is merged into the running feedback so the next pass sees the freshest state.
- Optionally (`--interleaved-kv-refresh`), between interleaved passes the buffered V/K at not-yet-visited positions are cheaply re-projected from the latest `h_final` to reduce staleness.

The training loss is the mean over all passes. Gradients flow through the KV buffer, so interleaved passes backpropagate into the full pass.

**Inference.** At inference the model decodes token by token with *true* recurrence (each position's `h_final` feeds the next). `evaluate_bpb_sequential` measures exactly this.

## Quick start

Environment setup follows nanochat (uv + PyTorch; see [runs/speedrun.sh](runs/speedrun.sh) for a reference setup). Then:

```bash
# 1. Download pretraining data (fineweb or climbmix)
python -m nanochat.dataset -d fineweb -n 240

# 2. Train the tokenizer (or reuse an existing one)
python -m scripts.tok_train --dataset fineweb

# 3. Pretrain with parallel interleaved training (8 GPUs; drop torchrun for single GPU)
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --run=d12-interleaved \
    --interleaved \
    --interleaved-passes=2 \
    --device-batch-size=4

# Baseline for comparison: same command without --interleaved
```

Evaluate a trained model:

```bash
# Parallel BPB (per-pass + merged), CORE, and samples
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --eval core,bpb

# Sequential (true recurrent) BPB, interleaved models only
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --eval bpb_seq
```

## Key CLI flags (`scripts.base_train`)

| Flag | Default | Description |
|------|---------|-------------|
| `--interleaved` | off | Enable parallel interleaved training (1 full pass + N interleaved passes) |
| `--interleaved-passes` | 2 | Number of interleaved passes; pass `i` covers positions `pos % N == i` |
| `--interleaved-kv-refresh` | off | Cheaply refresh buffered V/K at unvisited positions between interleaved passes |
| `--compile-interleaved` | off | Also `torch.compile` the interleaved-pass forward (runs eagerly by default) |
| `--eval-seq-every` | 250 | Run sequential (recurrent) BPB eval every N steps (interleaved models only) |
| `--eval-seq-batch-size` | 128 | Batch size for sequential eval (limited by KV cache memory) |
| `--eval-seq-full-pass-sanity` | off | Also run a zero-feedback token-by-token eval; should match the parallel full-pass BPB |
| `--dataset` | fineweb | Pretraining dataset: `fineweb` or `climbmix` |
| `--ve-gate-channels` | 32 | Input channels for the value/feedback gate (0 = full model dim) |
| `--no-value-embeds` / `--ve-every-layer` / `--no-x0-residual` / `--no-resid-lambdas` | — | Ablation toggles for the baseline architecture |

Setting `NANOCHAT_INTERLEAVED_FLEX=1` enables a FlexAttention fast path for interleaved-pass attention (falls back to SDPA on failure).

## Evaluation modes

- **Parallel BPB** (`evaluate_bpb`): for interleaved models, reports BPB per pass (`full`, `interleaved1`, …) plus a **merged** BPB where each position is scored by the last pass that computed it — the interleaved pipeline's actual output quality.
- **Sequential BPB** (`evaluate_bpb_sequential`): token-by-token decoding with true recurrent feedback, including a per-position-bucket breakdown to see how recurrence helps as context grows. A `full_pass_sanity` mode (zero feedback throughout) should reproduce the parallel full-pass BPB.
- **CORE** (`scripts.base_eval --eval core`): the DCLM CORE metric, evaluated with full-pass (zero-feedback) semantics.
- **Diagnostics**: at each eval, training prints the learned per-layer λ values (α residual, β x0, γ feedback) and the average triple-gate values (`gate_local` / `gate_fb` / `gate_ve`) per layer.

## What changed vs. upstream nanochat

| File | Change |
|------|--------|
| `nanochat/gpt.py` | Feedback projections, triple gate, `forward_interleaved_train` pipeline, optional KV refresh, diagnostics |
| `nanochat/flash_attention.py` | `flash_attn_interleaved` (queries at interleaved positions vs. a full KV buffer), FA3 padding path for unequal Q/KV lengths |
| `nanochat/loss_eval.py` | Per-pass + merged BPB; sequential recurrent BPB eval |
| `scripts/base_train.py` | Interleaved-training CLI flags, per-pass wandb logging, sequential eval loop |
| `scripts/base_eval.py` | `bpb_seq` eval mode, per-pass BPB reporting |
| `nanochat/dataset.py`, `scripts/tok_train.py` | Selectable pretraining dataset (fineweb / climbmix) |
| `nanochat/optim.py` | all_reduce fallback for params not shardable by world size |
| `nanochat/core_eval.py` | Minor refactor of loss/prediction computation |

## Acknowledgements

Built on [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy (MIT license) — the training harness, tokenizer, data pipeline, optimizer, and evaluation framework all come from upstream.

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that \$100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## License

MIT
