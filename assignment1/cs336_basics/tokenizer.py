"""Byte-level BPE Tokenizer implementation."""

from collections.abc import Iterable, Iterator
import json
import regex


# GPT-2 style pre-tokenization pattern
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    """
    A byte-level BPE tokenizer that encodes text into integer IDs and decodes integer IDs into text.
    Supports user-provided special tokens which are preserved as single tokens during encoding.
    """
    
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and optionally special tokens.
        
        Args:
            vocab: Mapping from token ID (int) to token bytes.
            merges: List of BPE merges, each a tuple of (token1, token2) indicating 
                    token1 was merged with token2. Ordered by order of creation.
            special_tokens: Optional list of special token strings to add to vocabulary.
        """
        # Copy vocab to avoid mutating the original
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = special_tokens or []
        
        # Add special tokens to vocabulary if not already present
        for token in self.special_tokens:
            byte_encoded = token.encode("utf-8")
            if byte_encoded not in set(self.vocab.values()):
                self.vocab[len(self.vocab)] = byte_encoded
        
        # Create reverse mapping: bytes -> token ID
        self.bytes_to_id = {v: k for k, v in self.vocab.items()}
        
        # Create merge priority mapping: (token1, token2) -> priority (lower = higher priority)
        self.merge_priority = {
            merge: idx for idx, merge in enumerate(self.merges)
        }
        
        # Build special token regex pattern for splitting text
        # Sort special tokens by length (longest first) to handle overlapping tokens
        if self.special_tokens:
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            escaped = [regex.escape(t) for t in sorted_special]
            self._special_pattern = regex.compile("(" + "|".join(escaped) + ")")
        else:
            self._special_pattern = None
    
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        """
        Construct a Tokenizer from serialized vocabulary and merges files.
        
        Args:
            vocab_filepath: Path to the JSON vocabulary file.
            merges_filepath: Path to the JSON merges file.
            special_tokens: Optional list of special token strings.
            
        Returns:
            A Tokenizer instance.
        """
        # Load vocabulary: keys are string IDs, values are repr'd bytes
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_serialized = json.load(f)
        
        # Parse vocab: convert string keys to int, eval repr'd bytes back to bytes
        vocab: dict[int, bytes] = {}
        for key, value in vocab_serialized.items():
            token_id = int(key)
            # value is like "b'\\x00'" or "b' t'" - use eval to parse
            token_bytes = eval(value)
            vocab[token_id] = token_bytes
        
        # Load merges: list of [repr(bytes1), repr(bytes2)]
        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges_serialized = json.load(f)
        
        # Parse merges
        merges: list[tuple[bytes, bytes]] = []
        for merge_pair in merges_serialized:
            token1 = eval(merge_pair[0])
            token2 = eval(merge_pair[1])
            merges.append((token1, token2))
        
        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        """
        Encode a string of text into a list of integer token IDs.
        
        Args:
            text: The input text string to encode.
            
        Returns:
            A list of integer token IDs.
        """
        tokens = []
        
        # Split text by special tokens if any
        if self._special_pattern:
            segments = self._special_pattern.split(text)
        else:
            segments = [text]

        pretokenized_segments = []
        
        for segment in segments:
            if segment in self.special_tokens:
                token_bytes = segment.encode("utf-8")
                pretokenized_segments.append((True, token_bytes))
            else:
                # Pre-tokenize using GPT-2 style regex
                for match in regex.finditer(PAT, segment):
                    pretoken = match.group().encode("utf-8")
                    # Split pretoken into tuple of individual bytes
                    token_tuple = tuple(bytes([b]) for b in pretoken)
                    pretokenized_segments.append((False, token_tuple))

        # Apply BPE to each pretokenized segment
        for is_special, token_data in pretokenized_segments:
            if is_special:
                token_id = self.bytes_to_id[token_data]
                tokens.append(token_id)
            else:
                bpe_tokens = self._bpe(token_data)
                tokens.extend(bpe_tokens)
        
        return tokens
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings, return a generator that lazily yields token IDs.
        
        This is required for memory-efficient tokenization of large files that we
        cannot directly load into memory.
        
        Args:
            iterable: An iterable of strings (e.g., a Python file handle).
            
        Yields:
            Integer token IDs one at a time.
        """
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of integer token IDs back into a string of text.
        
        Args:
            ids: A list of integer token IDs to decode.
            
        Returns:
            The decoded text string.
        """
        byte_chunks = [self.vocab[token_id] for token_id in ids]
        combined_bytes = b"".join(byte_chunks)
        return combined_bytes.decode("utf-8", errors="replace")

    def _bpe(self, token_tuple: tuple[bytes, ...]) -> list[int]:
        """
        Apply Byte Pair Encoding (BPE) to a tuple of bytes representing a token.
        
        Args:
            token_tuple: A tuple of bytes representing the token to encode.
            
        Returns:
            A list of integer token IDs after BPE.
        """
        token_list = list(token_tuple)
        
        while True:
            # Find all adjacent pairs
            pairs = [
                (token_list[i], token_list[i + 1])
                for i in range(len(token_list) - 1)
            ]
            
            # Determine the best pair to merge based on priority
            best_pair = None
            best_priority = float("inf")
            for pair in pairs:
                if pair in self.merge_priority:
                    priority = self.merge_priority[pair]
                    if priority < best_priority:
                        best_priority = priority
                        best_pair = pair
            
            # If no pairs can be merged, break
            if best_pair is None:
                break
            
            # Merge the best pair
            new_token = best_pair[0] + best_pair[1]
            new_token_list = []
            i = 0
            while i < len(token_list):
                if i < len(token_list) - 1 and (token_list[i], token_list[i + 1]) == best_pair:
                    new_token_list.append(new_token)
                    i += 2
                else:
                    new_token_list.append(token_list[i])
                    i += 1
            token_list = new_token_list
        
        # Convert final tokens to IDs
        token_ids = [self.bytes_to_id[token] for token in token_list]
        return token_ids