import argparse
import math
import os
import time
import torch
import numpy as np
import numpy.typing as npt
import torch.nn as nn

from torch import Tensor
from einops import rearrange
from cs336_basics.experiment_logger import ExperimentLogger
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.adamw_module import AdamW


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """
    Compute mean cross-entropy loss.

    Args:
        logits: (..., vocab_size) unnormalized log-probabilities
        targets: (...,) integer class indices

    Returns:
        Scalar mean cross-entropy loss.
    """
    # Flatten to 2D: (N, vocab_size) so indexing is straightforward
    logits_2d = rearrange(logits, "... vocab -> (...) vocab")
    targets_1d = rearrange(targets, "... -> (...)")

    # Subtract max for numerical stability (cancels in log-softmax)
    m = logits_2d.max(dim=-1, keepdim=True).values
    # shape (N, vocab_size)
    shifted = logits_2d - m

    # log-sum-exp: log Σ exp(o_j - m)
    log_normalizer = shifted.exp().sum(dim=-1).log()

    # Target logit: o_{x_{i+1}} - m
    # fancy indexing — for each row b it selects column targets_1d[b].
    # shape (N,)
    target_logit = shifted[torch.arange(shifted.shape[0]), targets_1d]

    # ℓ_i = -(o_t - m) + log Σ exp(o_j - m)
    loss = -target_logit + log_normalizer

    return loss.mean()

def learning_rate_schedule(
    step: int,
    warmup_steps: int,
    cosine_steps: int,
    lr_max: float,
    lr_min: float,
) -> float:
    """
    Compute learning rate at a given step using a linear warmup followed by cosine decay.

    Args:
        step: Current training step (0-indexed).
        warmup_steps: Number of steps to linearly increase the learning rate.
        cosine_steps: Number of steps to apply cosine decay.
        lr_max: Maximum learning rate after warmup.
        lr_min: Minimum learning rate after cosine decay.

    Returns:
        Learning rate for the current step.
    """
    if step < warmup_steps:
        # Linear warmup: lr = lr_max * (step / warmup_steps)
        return lr_max * (step / warmup_steps)
    elif warmup_steps <= step < cosine_steps:
        # Cosine decay: lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * (step - warmup_steps) / (cosine_steps - warmup_steps)))
        progress = (step - warmup_steps) / (cosine_steps - warmup_steps)
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))
    else:
        # After decay: lr = lr_min
        return lr_min
    

def gradient_clipping(parameters: list[Tensor], max_norm: float) -> None:
    """
    Avoid larger gradients. Clip gradients of the given parameters to have a maximum L2 norm of max_norm.

    Args:
        parameters: List of tensors with gradients to be clipped.
        max_norm: Maximum allowed L2 norm of the gradients.

    Returns:
        None. The function modifies the gradients in-place.
    """
    # Compute total norm
    total_norm = torch.sqrt(sum(p.grad.norm() ** 2 for p in parameters if p.grad is not None))
    if total_norm.item() < max_norm:
        return  # No clipping needed
    
    for p in parameters:
        if p.grad is not None:
            p.grad.mul_(max_norm / (total_norm.item() + 1e-6))  # Scale down gradients in-place

def data_loader(input: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[Tensor, Tensor]:
    """
    Randomly sample a batch of input and target sequences for training.

    Args:
        input: 1D numpy array of token indices representing the training data.
        batch_size: Number of sequences per batch.
        context_length: Length of each input sequence (number of tokens).
        device: Device to move the tensors to ("cpu" or "cuda").

    Returns:
        A tuple (input_batch, target_batch) where each tensor has shape
        (batch_size, context_length) on the requested device.
    """
    max_start = len(input) - context_length
    # shape (batch_size,)
    starts = np.random.randint(0, max_start, size=batch_size)
    # shape (batch_size, context_length)
    x = np.stack([input[s:s + context_length] for s in starts])
    y = np.stack([input[s + 1:s + context_length + 1] for s in starts])
    input_batch = torch.tensor(x, dtype=torch.long).to(device)
    target_batch = torch.tensor(y, dtype=torch.long).to(device)
    return input_batch, target_batch

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, iteration: int, output_path: str) -> None:
    """
    Save model and optimizer state to a checkpoint file.

    Args:
        model: The model to save.
        optimizer: The optimizer to save.
        iteration: The current training iteration (used for naming the checkpoint).
        output_path: The path to save the checkpoint file.
    Returns:
        None. The function saves a file to disk.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, output_path)

def load_checkpoint(src: str, model: nn.Module, optimizer: torch.optim.Optimizer) -> int:
    """
    Load model and optimizer state from a checkpoint file.

    Args:
        src: Path to the checkpoint file.
        model: The model to load the state into.
        optimizer: The optimizer to load the state into.

    Returns:
        The iteration number stored in the checkpoint.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]


# ---------------------------------------------------------------------------
# Training script
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Transformer LM")

    # Data
    p.add_argument("--train_path", type=str, required=True,
                   help="Path to training data (.npy, 1-D int array of token IDs)")
    p.add_argument("--val_path",   type=str, default=None,
                   help="Path to validation data (.npy)")

    # Model
    p.add_argument("--vocab_size",     type=int,   default=10000)
    p.add_argument("--context_length", type=int,   default=256)
    p.add_argument("--d_model",        type=int,   default=512)
    p.add_argument("--num_layers",     type=int,   default=4)
    p.add_argument("--num_heads",      type=int,   default=16)
    p.add_argument("--d_ff",           type=int,   default=1344)
    p.add_argument("--rope_theta",     type=float, default=10000.0)

    # Optimizer
    p.add_argument("--lr_max",       type=float, default=1e-3)
    p.add_argument("--lr_min",       type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int,   default=500)
    p.add_argument("--cosine_steps", type=int,   default=5000)
    p.add_argument("--beta1",        type=float, default=0.9)
    p.add_argument("--beta2",        type=float, default=0.95)
    p.add_argument("--eps",          type=float, default=1e-8)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip",    type=float, default=1.0,
                   help="Max L2 norm for gradient clipping (0 = disabled)")

    # Training loop
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_iters",  type=int, default=5000)

    # Logging & checkpointing
    p.add_argument("--log_interval",        type=int, default=100)
    p.add_argument("--val_interval",        type=int, default=500)
    p.add_argument("--val_iters",           type=int, default=20,
                   help="Batches to average for validation loss estimate")
    p.add_argument("--checkpoint_dir",      type=str, default="checkpoints")
    p.add_argument("--checkpoint_interval", type=int, default=1000)
    p.add_argument("--resume",              type=str, default=None,
                   help="Checkpoint path to resume training from")

    # Misc
    p.add_argument("--device",     type=str, default="cpu")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--data_dtype", type=str, default="uint16",
                   help="numpy dtype of the encoded .npy files (e.g. uint16, int32, int64)")

    # Experiment tracking (JSONL)
    p.add_argument("--log_dir",  type=str, default="logs",
                   help="Directory for JSONL experiment logs")
    p.add_argument("--run_name", type=str, default=None,
                   help="Run identifier used as the JSONL filename. "
                        "Defaults to a timestamp-based name.")

    # Weights & Biases (optional)
    p.add_argument("--wandb",         action="store_true",
                   help="Enable Weights & Biases logging")
    p.add_argument("--wandb_project", type=str, default="cs336",
                   help="W&B project name")
    p.add_argument("--wandb_entity",  type=str, default=None,
                   help="W&B entity (team or username)")

    # Sanity check
    p.add_argument("--overfit_batch", action="store_true",
                   help="Overfit a single fixed batch as a sanity check. "
                        "Loss should drop to ~0.")
    p.add_argument("--overfit_steps", type=int, default=200,
                   help="Number of gradient steps for --overfit_batch (default 200)")

    return p.parse_args()


@torch.no_grad()
def _val_loss(model, val_data, batch_size, context_length, device, val_iters):
    """Estimate validation loss over `val_iters` random batches."""
    model.eval()
    total = 0.0
    for _ in range(val_iters):
        x, y = data_loader(val_data, batch_size, context_length, device)
        total += cross_entropy(model(x), y).item()
    model.train()
    return total / val_iters


def overfit_single_batch(args) -> None:
    """
    Sanity-check the training loop by overfitting a single fixed batch.
    Loss should fall from ~log(vocab_size) down toward 0.
    """
    dtype = np.dtype(args.data_dtype)
    print(f"Loading data: {args.train_path}")
    data = np.load(args.train_path, mmap_mode="r")
    assert data.dtype == dtype, f"Expected {dtype}, got {data.dtype}"

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
    ).to(args.device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # Fix one batch — never change it
    x, y = data_loader(data, args.batch_size, args.context_length, args.device)
    initial_loss = cross_entropy(model(x), y).item()
    print(f"Initial loss: {initial_loss:.4f}  "
          f"(random baseline ~{math.log(args.vocab_size):.2f})")
    print(f"\nOverfitting for {args.overfit_steps} steps...")
    print(f"{'step':>6}  {'loss':>10}")
    print("-" * 20)

    model.train()
    for step in range(args.overfit_steps):
        loss = cross_entropy(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            gradient_clipping(list(model.parameters()), args.grad_clip)
        optimizer.step()

        if (step + 1) % max(1, args.overfit_steps // 20) == 0 or step == 0:
            print(f"{step+1:>6}  {loss.item():>10.4f}")

    final_loss = loss.item()
    print("-" * 20)
    status = "PASS" if final_loss < initial_loss * 0.1 else "FAIL (loss did not drop 10x)"
    print(f"Final loss: {final_loss:.4f}  [{status}]")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.overfit_batch:
        overfit_single_batch(args)
        return

    # ------------------------------------------------------------------
    # Data  (memory-mapped -- only accessed slices are loaded into RAM)
    # ------------------------------------------------------------------
    dtype = np.dtype(args.data_dtype)

    print(f"Loading training data: {args.train_path}")
    train_data = np.load(args.train_path, mmap_mode="r")
    assert train_data.dtype == dtype, (
        f"Expected dtype {dtype}, got {train_data.dtype}. "
    )
    assert train_data.ndim == 1, f"Expected 1-D array, got shape {train_data.shape}"
    print(f"  shape={train_data.shape}  dtype={train_data.dtype}  "
          f"first tokens={train_data[:8].tolist()}")

    val_data = None
    if args.val_path:
        print(f"Loading validation data: {args.val_path}")
        val_data = np.load(args.val_path, mmap_mode="r")
        assert val_data.dtype == dtype, (
            f"Expected dtype {dtype}, got {val_data.dtype}."
        )
        assert val_data.ndim == 1, f"Expected 1-D array, got shape {val_data.shape}"
        print(f"  shape={val_data.shape}  dtype={val_data.dtype}  "
              f"first tokens={val_data[:8].tolist()}")

    # ------------------------------------------------------------------
    # Model & optimizer
    # ------------------------------------------------------------------
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
    ).to(args.device)
    model = torch.compile(model)  # may improve speed
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # ------------------------------------------------------------------
    # (Optional) Resume
    # ------------------------------------------------------------------
    start_iter = 0
    if args.resume:
        start_iter = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from {args.resume} at step {start_iter}")

    # ------------------------------------------------------------------
    # Experiment logger
    # ------------------------------------------------------------------
    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"{run_name}.jsonl")

    checkpoint_dir = os.path.join(args.checkpoint_dir, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger = ExperimentLogger(
        log_path,
        append=(args.resume is not None),
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=run_name,
        wandb_config=vars(args),
        wandb_resume="allow" if args.resume else None,
    )
    print(f"Logging to: {log_path}")

    # Write config record so the log is self-contained (JSONL)
    logger.log("config", step=0, wallclock=0.0, **vars(args))

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    model.train()
    train_start = time.time()   # wallclock anchor — never reset
    t0 = time.time()            # interval timer for tok/s

    try:
        for step in range(start_iter, args.max_iters):

            # Learning rate schedule
            lr = learning_rate_schedule(
                step=step,
                warmup_steps=args.warmup_steps,
                cosine_steps=args.cosine_steps,
                lr_max=args.lr_max,
                lr_min=args.lr_min,
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Forward / backward
            x, y = data_loader(train_data, args.batch_size, args.context_length, args.device)
            loss = cross_entropy(model(x), y)
            optimizer.zero_grad()
            loss.backward()

            if args.grad_clip > 0:
                gradient_clipping(list(model.parameters()), args.grad_clip)

            optimizer.step()

            # Logging
            if (step + 1) % args.log_interval == 0:
                elapsed_interval = time.time() - t0
                wallclock = time.time() - train_start
                tok_per_s = args.log_interval * args.batch_size * args.context_length / elapsed_interval
                train_loss = loss.item()
                print(f"step {step+1:>7d} | loss {train_loss:.4f} | lr {lr:.2e} | "
                      f"{tok_per_s:,.0f} tok/s | {wallclock:.1f}s")
                logger.log(
                    "train",
                    step=step + 1,
                    wallclock=wallclock,
                    loss=round(train_loss, 6),
                    lr=lr,
                    tok_per_s=round(tok_per_s, 1),
                )
                t0 = time.time()

            # Validation
            if val_data is not None and (step + 1) % args.val_interval == 0:
                wallclock = time.time() - train_start
                vl = _val_loss(model, val_data, args.batch_size, args.context_length,
                               args.device, args.val_iters)
                print(f"  [val] step {step+1:>7d} | val_loss {vl:.4f} | {wallclock:.1f}s")
                logger.log(
                    "val",
                    step=step + 1,
                    wallclock=wallclock,
                    val_loss=round(vl, 6),
                )

            # Checkpoint
            if (step + 1) % args.checkpoint_interval == 0:
                ckpt = os.path.join(checkpoint_dir, f"checkpoint_{step+1:07d}.pt")
                save_checkpoint(model, optimizer, step + 1, ckpt)
                print(f"  [checkpoint] {ckpt}")

        # Final checkpoint
        ckpt = os.path.join(checkpoint_dir, "checkpoint_final.pt")
        save_checkpoint(model, optimizer, args.max_iters, ckpt)
        print(f"Done. Final checkpoint: {ckpt}")

    finally:
        logger.close()


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    model: nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_p: float = 1.0,
    end_of_text_id: int | None = None,
    device: str = "cpu",
) -> list[int]:
    """
    Autoregressively sample tokens from `model` given a prompt.

    Args:
        model: A TransformerLM (or any nn.Module) that takes an int tensor of shape
               (1, seq_len) and returns logits of shape (1, seq_len, vocab_size).
        prompt_ids: List of integer token IDs forming the initial prompt.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Softmax temperature. Values < 1 sharpen the distribution
                     (more greedy); values > 1 flatten it (more diverse).
                     Set to 0 for fully greedy (argmax) decoding.
        top_p: Nucleus sampling threshold. Only the smallest set of tokens whose
               cumulative probability >= top_p are kept; the rest are zeroed out.
               Set to 1.0 to disable (keep all tokens).
        end_of_text_id: Token ID that signals end of generation. Generation stops
                        when this token is sampled. Pass None to ignore.
        device: PyTorch device string.

    Returns:
        List of generated token IDs (does NOT include the original prompt).
    """
    model.eval()
    context = list(prompt_ids)
    generated: list[int] = []

    # Infer context_length from the model if available
    context_length = getattr(model, "context_length", None)

    for _ in range(max_new_tokens):
        # Truncate to model's context window
        ids = context if context_length is None else context[-context_length:]
        x = torch.tensor([ids], dtype=torch.long, device=device)  # (1, seq_len)

        logits = model(x)                   # (1, seq_len, vocab_size)
        next_logits = logits[0, -1, :]      # (vocab_size,)

        # --- Temperature scaling ---
        if temperature == 0.0:
            # argmax return the index of max value from vocab
            next_id = int(next_logits.argmax())
        else:
            next_logits = next_logits / temperature

            # --- Top-p (nucleus) filtering ---
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cumprobs = torch.cumsum(probs, dim=-1)

                # Shift by one so the token that pushes cumsum over top_p is kept
                remove = (cumprobs - probs) > top_p
                sorted_logits[remove] = float("-inf")

                # Scatter back to original order
                next_logits = torch.empty_like(next_logits).scatter_(
                    0, sorted_idx, sorted_logits
                )

            probs = torch.softmax(next_logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())

        generated.append(next_id)
        context.append(next_id)

        if end_of_text_id is not None and next_id == end_of_text_id:
            break

    return generated


if __name__ == "__main__":
    main()
