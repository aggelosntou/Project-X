"""
tokenizer.py — Character-level and whitespace-level tokenizers.

Tokenization is the first step in any NLP pipeline: convert raw text
into integer sequences the model can process.

Two flavors:
  CharTokenizer  — one token per character, small vocab (~100), long sequences
  SimpleTokenizer — one token per word, large vocab, short sequences

Character-level pros: no unknown words, works for any language/code
Character-level cons: longer sequences (slower attention), harder to learn word semantics

Word-level pros: shorter sequences, each token carries rich semantic meaning
Word-level cons: out-of-vocabulary words, vocabulary can be huge
"""

import numpy as np
from typing import List, Optional


class CharTokenizer:
    """
    Character-level tokenizer.

    Maps each unique character in a corpus to an integer index.
    Encodes text as a list of those integers.
    Decodes back to the original string.

    Example:
        tok = CharTokenizer("hello world")
        ids = tok.encode("hello")   # [0, 1, 2, 2, 3]
        txt = tok.decode(ids)       # "hello"
    """

    def __init__(self, text: str):
        """
        Build vocabulary from all unique characters in `text`.
        Characters are sorted for reproducibility.
        """
        chars = sorted(set(text))
        self._char_to_id = {ch: i for i, ch in enumerate(chars)}
        self._id_to_char = {i: ch for i, ch in enumerate(chars)}
        self._vocab_size  = len(chars)

    def encode(self, text: str) -> List[int]:
        """Convert string to list of integer IDs."""
        return [self._char_to_id[ch] for ch in text
                if ch in self._char_to_id]

    def decode(self, ids) -> str:
        """Convert list of integer IDs back to string."""
        return ''.join(self._id_to_char.get(int(i), '?') for i in ids)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def __repr__(self):
        return f"CharTokenizer(vocab_size={self._vocab_size})"


class SimpleTokenizer:
    """
    Whitespace tokenizer with special tokens: <pad>, <bos>, <eos>, <unk>.

    Workflow:
      1. Call fit(texts) on a list of training strings to build vocabulary.
      2. Call encode(text) to get integer IDs.
      3. Call decode(ids) to get back text.

    Special tokens:
      <pad> — padding token for batching variable-length sequences
      <bos> — beginning of sequence
      <eos> — end of sequence
      <unk> — unknown word (not in vocabulary)

    Example:
        tok = SimpleTokenizer()
        tok.fit(["the cat sat on the mat", "the dog sat on the log"])
        ids = tok.encode("the cat")   # [4, 5]  (after special tokens)
        tok.decode(ids)               # "the cat"
    """

    PAD_TOKEN = '<pad>'
    BOS_TOKEN = '<bos>'
    EOS_TOKEN = '<eos>'
    UNK_TOKEN = '<unk>'

    def __init__(self):
        self._word_to_id: dict = {}
        self._id_to_word: dict = {}
        self._vocab_size: int  = 0

        # Special token IDs — set during fit
        self.pad_id: int = 0
        self.bos_id: int = 1
        self.eos_id: int = 2
        self.unk_id: int = 3

    def fit(self, texts: List[str]) -> 'SimpleTokenizer':
        """
        Build vocabulary from a list of strings.
        Inserts special tokens first so their IDs are stable (0, 1, 2, 3).
        """
        specials = [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        words = []
        for text in texts:
            words.extend(text.split())

        # Deduplicate, sort for reproducibility
        unique_words = sorted(set(words))

        vocab = specials + unique_words
        self._word_to_id = {w: i for i, w in enumerate(vocab)}
        self._id_to_word = {i: w for i, w in enumerate(vocab)}
        self._vocab_size  = len(vocab)

        self.pad_id = self._word_to_id[self.PAD_TOKEN]
        self.bos_id = self._word_to_id[self.BOS_TOKEN]
        self.eos_id = self._word_to_id[self.EOS_TOKEN]
        self.unk_id = self._word_to_id[self.UNK_TOKEN]

        return self

    def encode(self, text: str, max_len: Optional[int] = None,
               add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """
        Convert string to list of integer IDs.

        Args:
            text:    input string (whitespace-tokenized)
            max_len: if set, truncate or pad to this length
            add_bos: prepend <bos> token
            add_eos: append <eos> token

        Returns:
            List of integer token IDs.
        """
        tokens = [self._word_to_id.get(w, self.unk_id) for w in text.split()]

        if add_bos:
            tokens = [self.bos_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_id]

        if max_len is not None:
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            else:
                tokens = tokens + [self.pad_id] * (max_len - len(tokens))

        return tokens

    def decode(self, ids, skip_special: bool = True) -> str:
        """
        Convert integer IDs back to string.

        Args:
            ids:           iterable of integer token IDs
            skip_special:  if True, omit <pad>, <bos>, <eos>, <unk> tokens
        """
        specials = {self.pad_id, self.bos_id, self.eos_id, self.unk_id} if skip_special else set()
        words = [self._id_to_word.get(int(i), self.UNK_TOKEN)
                 for i in ids if int(i) not in specials]
        return ' '.join(words)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def __repr__(self):
        return f"SimpleTokenizer(vocab_size={self._vocab_size})"


if __name__ == "__main__":
    # ---- CharTokenizer demo ----
    print("=== CharTokenizer ===")
    text = "hello world, this is a test. 0123456789"
    tok = CharTokenizer(text)
    print(f"Vocab size: {tok.vocab_size}")
    encoded = tok.encode("hello")
    print(f"Encoded 'hello': {encoded}")
    print(f"Decoded back:    '{tok.decode(encoded)}'")

    # ---- SimpleTokenizer demo ----
    print("\n=== SimpleTokenizer ===")
    corpus = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "a quick brown fox jumps over the lazy dog",
    ]
    stok = SimpleTokenizer()
    stok.fit(corpus)
    print(f"Vocab size: {stok.vocab_size}")
    ids = stok.encode("the cat sat on the mat", add_bos=True, add_eos=True)
    print(f"Encoded: {ids}")
    print(f"Decoded: '{stok.decode(ids)}'")

    # Padding test
    ids_padded = stok.encode("the cat", max_len=10)
    print(f"Padded (max_len=10): {ids_padded}")

    # Round-trip test for addition tokenizer usage
    print("\n=== Addition format test ===")
    addition_corpus = [f"{a}+{b}={a+b}" for a in range(10) for b in range(10)]
    sample = "123+456=579"
    char_tok = CharTokenizer("".join(addition_corpus) + sample)
    enc = char_tok.encode(sample)
    dec = char_tok.decode(enc)
    print(f"Original: {sample}")
    print(f"Decoded:  {dec}")
    assert dec == sample, "Round-trip failed!"
    print("Round-trip: PASSED")
