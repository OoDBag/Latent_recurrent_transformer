"""
A number of functions that help with evaluating a base model.
"""
import math
import time
import torch
import torch.nn.functional as F
import torch.distributed as dist
from nanochat.common import print0


def _accumulate_nats_bytes(loss2d, y, token_bytes, total_nats, total_bytes):
    """
    Helper to accumulate nats and bytes from per-token losses.
    Handles ignore_index (negative targets) correctly.
    """
    loss2d = loss2d.view(-1)  # flatten
    y = y.view(-1)  # flatten
    if (y.int() < 0).any():  # mps does not currently have kernel for < 0 for int64, only int32
        # slightly more complex code path if some target tokens are ignore_index (e.g. -1)
        # any target token < 0 is to be ignored: do NOT index token_bytes with negatives
        valid = y >= 0
        y_safe = torch.where(valid, y, torch.zeros_like(y))
        # map valid targets to their byte length; ignored targets contribute 0 bytes
        num_bytes2d = torch.where(
            valid,
            token_bytes[y_safe],
            torch.zeros_like(y, dtype=token_bytes.dtype)
        )
        total_nats += (loss2d * (num_bytes2d > 0)).sum()
        total_bytes += num_bytes2d.sum()
    else:
        # fast path: no ignored targets, safe to index directly
        num_bytes2d = token_bytes[y]
        total_nats += (loss2d * (num_bytes2d > 0)).sum()
        total_bytes += num_bytes2d.sum()


def _reduce_and_compute_bpb(total_nats, total_bytes):
    """Helper to reduce across ranks and compute BPB."""
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
    if total_bytes == 0:
        return float('inf')
    return total_nats / (math.log(2) * total_bytes)


@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    """
    Instead of the naive 'mean loss', this function returns the bits per byte (bpb),
    which is a tokenization vocab size-independent metric, meaning you are still comparing
    apples:apples if you change the vocab size. The way this works is that instead of just
    calculating the average loss as usual, you calculate the sum loss, and independently
    also the sum bytes (of all the target tokens), and divide. This normalizes the loss by
    the number of bytes that the target tokens represent.

    The added complexity is so that:
    1) All "normal" tokens are normalized by the length of the token in bytes
    2) No special tokens (e.g. <|bos|>) are included in the metric - they are masked out.
    3) No actively masked tokens (using ignore_index of e.g. -1) are included in the metric.

    In addition to evaluate_loss, we need the token_bytes tensor:
    It is a 1D tensor of shape (vocab_size,), indicating the number of bytes for
    each token id, or 0 if the token is to not be counted (e.g. special tokens).

    For interleaved models, this runs the full interleaved pipeline and returns a dict with:
    - 'bpb': merged BPB — each position scored by the LAST pass that computed it
             (the interleaved pipeline's actual output quality)
    - 'per_pass_bpb': list of BPB, [full, interleaved_1, ..., interleaved_N]; interleaved passes are
             scored only at their own positions
    For non-interleaved models, returns just the BPB float (backwards compatible).
    """
    device = model.get_device()
    use_interleaved = getattr(model.config, 'use_interleaved', False)

    if use_interleaved:
        n_passes = 1 + model.config.interleaved_num_passes
        # Track nats/bytes per pass, plus the merged (last-writer-wins) combination
        per_pass_nats = [torch.tensor(0.0, dtype=torch.float32, device=device) for _ in range(n_passes)]
        per_pass_bytes = [torch.tensor(0, dtype=torch.int64, device=device) for _ in range(n_passes)]
        merged_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
        merged_bytes = torch.tensor(0, dtype=torch.int64, device=device)

        batch_iter = iter(batches)
        for _ in range(steps):
            x, y = next(batch_iter)
            outputs = model.forward_interleaved_train(x, y, loss_reduction='none')
            per_pass_losses = outputs['per_pass_losses']  # list of [B*T] tensors
            per_pass_masks = outputs['per_pass_masks']    # list of None or [T] bool
            assert len(per_pass_losses) == n_passes

            merged_loss = per_pass_losses[0].clone()  # full pass covers all positions
            for p, (loss_flat, mask) in enumerate(zip(per_pass_losses, per_pass_masks)):
                if mask is None:
                    # full pass: covers all positions
                    _accumulate_nats_bytes(loss_flat, y, token_bytes, per_pass_nats[p], per_pass_bytes[p])
                else:
                    # interleaved pass: score only at its own (core) positions
                    y_pass = y.clone()
                    y_pass[:, ~mask] = -1
                    _accumulate_nats_bytes(loss_flat, y_pass, token_bytes, per_pass_nats[p], per_pass_bytes[p])
                    mask_flat = mask.unsqueeze(0).expand(y.size(0), -1).reshape(-1)
                    merged_loss = torch.where(mask_flat, loss_flat, merged_loss)
            _accumulate_nats_bytes(merged_loss, y, token_bytes, merged_nats, merged_bytes)

        per_pass_bpb = [_reduce_and_compute_bpb(per_pass_nats[p], per_pass_bytes[p]) for p in range(n_passes)]
        merged_bpb = _reduce_and_compute_bpb(merged_nats, merged_bytes)

        return {
            'bpb': merged_bpb,
            'per_pass_bpb': per_pass_bpb,
        }
    else:
        # Standard single-pass evaluation
        total_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
        total_bytes = torch.tensor(0, dtype=torch.int64, device=device)

        batch_iter = iter(batches)
        for _ in range(steps):
            x, y = next(batch_iter)
            loss2d, _ = model(x, y, loss_reduction='none')  # (B, T), h_final ignored
            _accumulate_nats_bytes(loss2d, y, token_bytes, total_nats, total_bytes)

        bpb = _reduce_and_compute_bpb(total_nats, total_bytes)
        return bpb


@torch.no_grad()
def evaluate_bpb_sequential(model, batches, steps, token_bytes, mode='recurrent', bucket_size=256):
    """
    Sequential token-by-token BPB evaluation for interleaved models (true recurrent inference).

    Args:
        mode: 'recurrent' = feed h_final from the previous position as feedback (interleaved-pass mode)
              'full_pass_sanity' = zero feedback throughout (should match the parallel full-pass bpb)
        bucket_size: position bucket size for per-bucket BPB breakdown (default 256)

    Returns dict with:
        'bpb': overall BPB (float)
        'bucket_bpb': list of (start, end, bpb) tuples for each position bucket
    """
    from nanochat.engine import KVCache

    device = model.get_device()
    config = model.config
    n_kv_head = config.n_kv_head
    head_dim = config.n_embd // config.n_head
    n_layer = config.n_layer
    # Match Engine convention: use bfloat16 on CUDA (autocast produces bf16 queries/keys)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    total_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=device)

    use_feedback = (mode == 'recurrent')

    # Per-bucket accumulators (allocated lazily once we know T)
    bucket_nats = None
    bucket_bytes = None
    n_buckets = None

    batch_iter = iter(batches)
    t_start = time.time()
    for batch_idx in range(steps):
        x, y = next(batch_iter)
        B, T = x.shape

        # Lazy init buckets on first batch
        if bucket_nats is None:
            n_buckets = (T + bucket_size - 1) // bucket_size
            bucket_nats = [torch.tensor(0.0, dtype=torch.float32, device=device) for _ in range(n_buckets)]
            bucket_bytes = [torch.tensor(0, dtype=torch.int64, device=device) for _ in range(n_buckets)]

        kv_cache = KVCache(B, n_kv_head, T, head_dim, n_layer, device, dtype)

        # Initialize feedback as zeros
        h_final_prev = torch.zeros(B, 1, config.n_embd, device=device, dtype=dtype)

        for t in range(T):
            token = x[:, t:t+1]  # [B, 1]

            # pass_idx is used for gate diagnostics only: label t=0 as the full pass
            # (zero feedback) and t>0 as a interleaved pass when feedback is being fed back.
            pidx = 1 if (use_feedback and t > 0) else 0

            # Always pass h_final_prev (never None): the training full pass receives an
            # explicit zeros tensor, so the projection/gate path always runs. Passing
            # None would skip that path entirely, causing a train/eval mismatch.
            logits, h_final = model(
                token, kv_cache=kv_cache,
                h_final_prev=h_final_prev,
                pass_idx=pidx,
                return_hidden=True,
            )

            # Compute per-token loss
            targets_t = y[:, t]  # [B]
            loss_t = F.cross_entropy(
                logits[:, 0, :], targets_t,
                ignore_index=-1, reduction='none',
            )  # [B]

            # Accumulate nats and bytes
            valid = targets_t >= 0
            safe_t = torch.where(valid, targets_t, torch.zeros_like(targets_t))
            nbytes_t = torch.where(
                valid,
                token_bytes[safe_t],
                torch.zeros_like(safe_t, dtype=token_bytes.dtype),
            )
            nats_t = (loss_t * (nbytes_t > 0).float()).sum()
            bytes_t = nbytes_t.sum()
            total_nats += nats_t
            total_bytes += bytes_t

            # Per-bucket accumulation
            bucket_idx = t // bucket_size
            bucket_nats[bucket_idx] += nats_t
            bucket_bytes[bucket_idx] += bytes_t

            # Update feedback for next position (only in recurrent mode)
            if use_feedback:
                h_final_prev = h_final  # [B, 1, n_embd]

        # Progress logging
        elapsed = time.time() - t_start
        avg_per_batch = elapsed / (batch_idx + 1)
        remaining = avg_per_batch * (steps - batch_idx - 1)
        print0(f"  bpb_seq({mode}): batch {batch_idx+1}/{steps} | {elapsed:.1f}s elapsed | eta: {remaining:.1f}s")

    overall_bpb = _reduce_and_compute_bpb(total_nats, total_bytes)

    # Compute per-bucket BPB
    bucket_bpb = []
    if n_buckets is not None:
        for b in range(n_buckets):
            bpb = _reduce_and_compute_bpb(bucket_nats[b], bucket_bytes[b])
            start = b * bucket_size
            end = min(start + bucket_size, T)
            bucket_bpb.append((start, end, bpb))

    return {'bpb': overall_bpb, 'bucket_bpb': bucket_bpb}
