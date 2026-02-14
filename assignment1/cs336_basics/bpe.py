"""Byte-level BPE tokenizer training."""

import json
import os
import regex
from multiprocessing import Pool, cpu_count
from typing import BinaryIO

# GPT-2 style pre-tokenization pattern
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def pretokenize(text: str) -> dict[tuple[bytes, ...], int]:
    """
    Pre-tokenize text using the GPT-2 style regex pattern and build frequency table.
    Uses finditer to avoid storing all pre-tokens in memory.
    Splits on <|endoftext|> to ensure no merging across document boundaries.

    Args:
        text: Input text to pre-tokenize.

    Returns:
        A frequency table mapping tuples of bytes to their counts.
        e.g., {(b'l', b'o', b'w'): 5, (b'l', b'o', b'w', b'e', b'r'): 2, ...}
    """
    # normalize newlines
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    freq_table: dict[tuple[bytes, ...], int] = {}
    # Split on special token to prevent merging across documents
    for segment in text.split("<|endoftext|>"):
        for match in regex.finditer(PAT, segment):
            pretoken = match.group().encode("utf-8")
            # Split pretoken into tuple of individual bytes
            token_tuple = tuple(bytes([b]) for b in pretoken)
            freq_table[token_tuple] = freq_table.get(token_tuple, 0) + 1
    return freq_table


def merge_freq_tables(
    tables: list[dict[tuple[bytes, ...], int]]
) -> dict[tuple[bytes, ...], int]:
    """
    Merge multiple frequency tables into one.

    Args:
        tables: List of frequency tables to merge.

    Returns:
        A single merged frequency table.
    """
    merged: dict[tuple[bytes, ...], int] = {}
    for table in tables:
        for key, count in table.items():
            merged[key] = merged.get(key, 0) + count
    return merged


def pretokenize_chunk(args: tuple[str, int, int]) -> dict[tuple[bytes, ...], int]:
    """
    Pre-tokenize a chunk of a file.

    Args:
        args: Tuple of (file_path, start_offset, end_offset)

    Returns:
        A frequency table for this chunk.
    """
    file_path, start, end = args
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return pretokenize(chunk)


def parallel_pretokenize(
    input_path: str,
    num_processes: int | None = None,
    split_special_token: bytes = b"<|endoftext|>",
) -> dict[tuple[bytes, ...], int]:
    """
    Pre-tokenize a file in parallel using multiprocessing.

    Args:
        input_path: Path to the input file.
        num_processes: Number of processes to use. Defaults to CPU count.
        split_special_token: Token to split chunks on.

    Returns:
        A frequency table mapping tuples of bytes to their counts.
    """
    if num_processes is None:
        num_processes = cpu_count()

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_special_token)

    # Create args for each chunk
    chunk_args = [
        (input_path, start, end)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    # Process chunks in parallel
    with Pool(num_processes) as pool:
        chunk_tables = pool.map(pretokenize_chunk, chunk_args)

    # Merge all frequency tables
    return merge_freq_tables(chunk_tables)


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer.

    Args:
        input_path: Path to a text file with BPE tokenizer training data.
        vocab_size: A positive integer that defines the maximum final vocabulary size
            (including the initial byte vocabulary, vocabulary items produced from
            merging, and any special tokens).
        special_tokens: A list of strings to add to the vocabulary. These special
            tokens do not otherwise affect BPE training.

    Returns:
        vocab: The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes).
        merges: A list of BPE merges produced from training. Each list item is a tuple
            of bytes (<token1>, <token2>), representing that <token1> was merged with
            <token2>. The merges should be ordered by order of creation.
    """
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []

    # Initialize vocabulary with all byte values
    for i in range(256):
        vocab[i] = bytes([i])   

    # Add special tokens to vocabulary
    for token in special_tokens:
        token_id = len(vocab)
        vocab[token_id] = token.encode("utf-8")

    # Pre-tokenize in parallel, building frequency table
    freq_table = parallel_pretokenize(input_path)

    # Build initial pair frequency index and reverse index (pair -> token_tuples containing it)
    pair_freq: dict[tuple[bytes, bytes], int] = {}
    pair_to_tuples: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}
    for token_tuple, count in freq_table.items():
        for i in range(len(token_tuple) - 1):
            pair = (token_tuple[i], token_tuple[i + 1])
            pair_freq[pair] = pair_freq.get(pair, 0) + count
            if pair not in pair_to_tuples:
                pair_to_tuples[pair] = set()
            pair_to_tuples[pair].add(token_tuple)

    # Merge loop until reaching desired vocab size
    while len(vocab) < vocab_size:
        # If no pairs found, break early
        if not pair_freq:
            break

        # Find the most frequent pair (break ties by lexicographically greater pair)
        most_frequent_pair = max(pair_freq.items(), key=lambda x: (x[1], x[0]))[0]

        # Create new token by merging the most frequent pair (concatenate bytes)
        new_token = most_frequent_pair[0] + most_frequent_pair[1]
        new_token_id = len(vocab)
        vocab[new_token_id] = new_token
        merges.append(most_frequent_pair)

        # Only process tuples that contain the merged pair
        affected_tuples = list(pair_to_tuples.get(most_frequent_pair, set()))
        
        for token_tuple in affected_tuples:
            count = freq_table[token_tuple]
            
            # Remove old pairs from indices
            for i in range(len(token_tuple) - 1):
                old_pair = (token_tuple[i], token_tuple[i + 1])
                pair_freq[old_pair] -= count
                if pair_freq[old_pair] <= 0:
                    del pair_freq[old_pair]
                if old_pair in pair_to_tuples:
                    pair_to_tuples[old_pair].discard(token_tuple)
                    if not pair_to_tuples[old_pair]:
                        del pair_to_tuples[old_pair]
            
            # Merge adjacent pairs in the token tuple
            new_tuple: list[bytes] = []
            i = 0
            while i < len(token_tuple):
                if (
                    i < len(token_tuple) - 1
                    and token_tuple[i] == most_frequent_pair[0]
                    and token_tuple[i + 1] == most_frequent_pair[1]
                ):
                    new_tuple.append(new_token)
                    i += 2
                else:
                    new_tuple.append(token_tuple[i])
                    i += 1
            new_key = tuple(new_tuple)
            
            # Update freq_table
            del freq_table[token_tuple]
            freq_table[new_key] = count
            
            # update pair_freq and pair_to_tuples with new pairs
            for i in range(len(new_key) - 1):
                new_pair = (new_key[i], new_key[i + 1])
                pair_freq[new_pair] = pair_freq.get(new_pair, 0) + count
                if new_pair not in pair_to_tuples:
                    pair_to_tuples[new_pair] = set()
                pair_to_tuples[new_pair].add(new_key)

    return vocab, merges


if __name__ == "__main__":
    import time
    print('Training BPE on TinyStories dataset...')
    start_time = time.time()
    vocab, merges = train_bpe(
        input_path='data/TinyStoriesV2-GPT4-train.txt',
        vocab_size=10000,
        special_tokens=['<|endoftext|>'],
    )
    end_time = time.time()
    print(f'Training time: {end_time - start_time:.2f} seconds')
    # Serialize vocab: convert bytes to repr strings for readability
    vocab_serialized = {k: repr(v) for k, v in vocab.items()}
    with open('data/tinystories_vocab.json', 'w') as f:
        json.dump(vocab_serialized, f, indent=2)

    # Serialize merges: convert tuples of bytes to repr pairs
    merges_serialized = [[repr(m[0]), repr(m[1])] for m in merges]
    with open('data/tinystories_merges.json', 'w') as f:
        json.dump(merges_serialized, f, indent=2)

    print(f'Vocab size: {len(vocab)}')
    print(f'Number of merges: {len(merges)}')