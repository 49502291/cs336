import torch
import torch.nn as nn

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer_lm import TransformerLM

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


class Generator:
    """
    Stateful generator: load model + tokenizer once, call generate() repeatedly.

    Example
    -------
    gen = Generator(
        checkpoint="checkpoints/lr_3e-3/checkpoint_final.pt",
        vocab_path="data/tinystories_vocab.json",
        merges_path="data/tinystories_merges.json",
    )
    print(gen.generate("Once upon a time"))
    print(gen.generate("The dragon said", temperature=0.5))
    """

    def __init__(
        self,
        checkpoint: str,
        vocab_path: str,
        merges_path: str,
        special_tokens: list[str] | None = None,
        vocab_size: int = 10000,
        context_length: int = 256,
        d_model: int = 512,
        num_layers: int = 4,
        num_heads: int = 16,
        d_ff: int = 1344,
        rope_theta: float = 10000.0,
        device: str = "cpu",
    ):
        self.device = device

        self.tokenizer = Tokenizer.from_files(
            vocab_filepath=vocab_path,
            merges_filepath=merges_path,
            special_tokens=special_tokens,
        )

        self.model = TransformerLM(
            vocab_size=vocab_size,
            context_length=context_length,
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=rope_theta,
            device=device,
        ).to(device)

        ckpt = torch.load(checkpoint, map_location=device)
        state_dict = ckpt["model_state_dict"]
        # torch.compile() adds "_orig_mod." prefix — strip it if present
        if any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        print(f"Loaded checkpoint: {checkpoint}  (step {ckpt.get('iteration', '?')})")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_p: float = 1.0,
        end_of_text_id: int | None = None,
    ) -> str:
        """Encode prompt, sample tokens, decode and return the full text (prompt + continuation)."""
        prompt_ids = self.tokenizer.encode(prompt)
        generated_ids = generate(
            model=self.model,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            end_of_text_id=end_of_text_id,
            device=self.device,
        )
        return prompt + self.tokenizer.decode(generated_ids)
