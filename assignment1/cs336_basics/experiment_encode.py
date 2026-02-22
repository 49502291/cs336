"""Experiment: Compute compression ratio (bytes/token) for tokenizers."""

import numpy as np
from pathlib import Path
from cs336_basics.tokenizer import Tokenizer


def compute_compression_ratio(tokenizer: Tokenizer, text: str) -> dict:
    """Compute compression statistics for a given tokenizer and text."""
    num_bytes = len(text.encode("utf-8"))
    tokens = tokenizer.encode(text)
    num_tokens = len(tokens)
    ratio = num_bytes / num_tokens if num_tokens > 0 else 0
    return {
        "bytes": num_bytes,
        "tokens": num_tokens,
        "bytes_per_token": ratio,
    }


def encode_and_save(tokenizer: Tokenizer, input_path: str, output_path: str) -> dict:
    """
    Encode a text file and save token IDs as a NumPy array (uint16).
    
    Args:
        tokenizer: The tokenizer to use.
        input_path: Path to the input text file.
        output_path: Path to save the .npy file.
        
    Returns:
        Dictionary with encoding statistics.
    """
    print(f"Encoding {input_path}...")
    
    # Get file size
    num_bytes = Path(input_path).stat().st_size
    print(f"  File size: {num_bytes:,} bytes ({num_bytes / 1024 / 1024:.2f} MB)")
    
    # Encode using iterable for memory efficiency with large files
    with open(input_path, "r", encoding="utf-8") as f:
        token_ids = list(tokenizer.encode_iterable(f))
    
    num_tokens = len(token_ids)    
    # Convert to NumPy array and save
    token_array = np.array(token_ids, dtype=np.uint16)
    np.save(output_path, token_array)
    
    # Report file size
    output_size = Path(output_path).stat().st_size
    print(f"  Saved to: {output_path}")
    print(f"  Output size: {output_size:,} bytes ({output_size / 1024 / 1024:.2f} MB)")
    
    return {
        "input_bytes": num_bytes,
        "num_tokens": num_tokens,
        "output_bytes": output_size
    }


def calculate_compression_ratio():
    # TinyStories
    print("=" * 50)
    print("TinyStories")
    print("=" * 50)
    
    ts_tokenizer = Tokenizer.from_files(
        "../data/tinystories_vocab.json",
        "../data/tinystories_merges.json",
        special_tokens=["<|endoftext|>"]
    )
    
    with open("../data/sample_tinystories.txt", "r", encoding="utf-8") as f:
        ts_text = f.read()
    
    ts_stats = compute_compression_ratio(ts_tokenizer, ts_text)
    print(f"File size:         {ts_stats['bytes']:,} bytes")
    print(f"Number of tokens:  {ts_stats['tokens']:,}")
    print(f"Compression ratio: {ts_stats['bytes_per_token']:.4f} bytes/token")
    
    # OWT (OpenWebText)
    print()
    print("=" * 50)
    print("OWT (OpenWebText)")
    print("=" * 50)
    
    owt_tokenizer = Tokenizer.from_files(
        "../data/owt_vocab.json",
        "../data/owt_merges.json",
        special_tokens=["<|endoftext|>"]
    )
    
    with open("../data/sample_owt.text", "r", encoding="utf-8") as f:
        owt_text = f.read()
    
    owt_stats = compute_compression_ratio(owt_tokenizer, owt_text)
    print(f"File size:         {owt_stats['bytes']:,} bytes")
    print(f"Number of tokens:  {owt_stats['tokens']:,}")
    print(f"Compression ratio: {owt_stats['bytes_per_token']:.4f} bytes/token")

    # OWT text with TinyStories tokenizer (mismatched)
    print()
    print("=" * 50)
    print("OWT text with TinyStories tokenizer (mismatched)")
    print("=" * 50)
    
    owt_with_ts_stats = compute_compression_ratio(ts_tokenizer, owt_text)
    print(f"File size:         {owt_with_ts_stats['bytes']:,} bytes")
    print(f"Number of tokens:  {owt_with_ts_stats['tokens']:,}")
    print(f"Compression ratio: {owt_with_ts_stats['bytes_per_token']:.4f} bytes/token")
    print(f"Compression degradation: {owt_with_ts_stats['tokens']/owt_stats['tokens']:.2f}x more tokens")


def encode_datasets():
    """Encode all datasets and save as NumPy uint16 arrays."""
    data_dir = Path("../data")
    output_dir = Path("../data/encoded")
    output_dir.mkdir(exist_ok=True)
    
    # Load tokenizers with special token
    print("Loading tokenizers...")
    special_tokens = ["<|endoftext|>"]
    ts_tokenizer = Tokenizer.from_files(
        str(data_dir / "tinystories_vocab.json"),
        str(data_dir / "tinystories_merges.json"),
        special_tokens=special_tokens
    )
    owt_tokenizer = Tokenizer.from_files(
        str(data_dir / "owt_vocab.json"),
        str(data_dir / "owt_merges.json"),
        special_tokens=special_tokens
    )
    
    # Define datasets to encode
    datasets = [
        # (tokenizer, input_file, output_file)
        #(ts_tokenizer, "TinyStoriesV2-GPT4-train.txt", "tinystories_train.npy"),
        (ts_tokenizer, "TinyStoriesV2-GPT4-valid.txt", "tinystories_valid.npy"),
        #(owt_tokenizer, "owt_train.txt", "owt_train.npy"),
        (owt_tokenizer, "owt_valid.txt", "owt_valid.npy"),
    ]
    
    results = []
    
    for tokenizer, input_file, output_file in datasets:
        input_path = data_dir / input_file
        output_path = output_dir / output_file
        
        if not input_path.exists():
            print(f"\nSkipping {input_file} (file not found)")
            continue
        
        print()
        print("=" * 60)
        stats = encode_and_save(tokenizer, str(input_path), str(output_path))
        results.append((input_file, stats))
    
    # Summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for name, stats in results:
        print(f"{name}:")
        print(f"  Tokens: {stats['num_tokens']:,}")
        print(f"  Bytes/token: {stats['input_bytes'] / stats['num_tokens']:.4f}")


if __name__ == "__main__":
    encode_datasets()
