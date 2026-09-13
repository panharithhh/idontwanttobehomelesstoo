def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1 
    return counts 


def merge(ids, pair, idx) -> list:
    news_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and pair[0] == ids[i] and pair[1] == ids[i+1]:
            news_ids.append(idx)
            i += 2
        else:
            news_ids.append(ids[i]) 
            i += 1

    return news_ids
# ==============================================================================
# CS336 LEARNING ROADMAP & PEDAGOGICAL MILESTONES (AGENTS.md)
# ==============================================================================
# Goal: Build understanding from first principles. Test each piece with `pytest`.
#
# ------------------------------------------------------------------------------
# PHASE 1: TOKENIZER (cs336_basics/tokenizer.py)
# ------------------------------------------------------------------------------
# Step 1.1 [COMPLETED]: `get_stats` & `merge`
#   - Counts adjacent pairs and replaces occurrences with a new token ID.
#
# Step 1.2 [NEXT]: `decode(ids: list[int]) -> str`
#   - Invariant: Map each token ID back to bytes using `self.vocab`.
#   - Concatenate all bytes and decode: `b"".join(...).decode("utf-8", errors="replace")`.
#
# Step 1.3: `encode(text: str) -> list[int]`
#   - Convert raw string to initial byte IDs: `list(text.encode("utf-8"))`.
#   - Find which adjacent pairs in `ids` exist in `self.merges`.
#   - Greedily merge the pair with the lowest merge rank (earliest learned).
#   - Repeat until no more eligible pairs can be merged.
#
# Step 1.4: Special Tokens Handling
#   - Tokens like "<|endoftext|>" must NEVER be split into individual byte tokens.
#   - During `encode`, split text on special tokens first, encode text chunks,
#     and keep special tokens as single IDs.
#
# Step 1.5: CS336 Speed Optimization (From minbpe -> Stanford fast BPE)
#   - Regex Pre-tokenization: Use GPT-2 regex so merges never cross words/punctuation.
#   - Word-Frequency Dictionary: Instead of scanning raw text repeatedly (O(N*V)),
#     count unique pre-tokenized words `dict[tuple[bytes, ...], int]`.
#   - Verified by: `uv run pytest tests/test_tokenizer.py tests/test_train_bpe.py`
#
# ------------------------------------------------------------------------------
# PHASE 2: TRANSFORMER ARCHITECTURE (cs336_basics/model.py)
# ------------------------------------------------------------------------------
# Step 2.1: `Embedding` & `Linear` (shape checks, weight initialization).
# Step 2.2: `RMSNorm` (Root Mean Square LayerNorm, no mean centering, no bias).
# Step 2.3: `SwiGLU` (Feed-forward layer: (xW1 * swish(xW3))W2).
# Step 2.4: `RoPE` (Rotary Position Embeddings: rotating Q and K by angle frequencies).
# Step 2.5: `ScaledDotProductAttention` & `MultiHeadSelfAttention` (causal masking).
# Step 2.6: `TransformerBlock` & `TransformerLM` (Pre-norm residual connections).
#   - Verified by: `uv run pytest tests/test_model.py`
#
# ------------------------------------------------------------------------------
# PHASE 3: TRAINING, OPTIMIZATION & SAMPLING (optimizer.py, data.py)
# ------------------------------------------------------------------------------
# Step 3.1: Custom `AdamW` optimizer (decoupled weight decay).
# Step 3.2: Learning rate scheduler (cosine decay with linear warmup).
# Step 3.3: Data loader & batching (chunking tokens into context length sequences).
# Step 3.4: Text Generation (temperature scaling, top-k and top-p sampling).
# ==============================================================================


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes] | None = None, merges: dict[tuple[int, int], int] | None = None, special_tokens: list[str] | None = None):
        # Base byte vocabulary: 0..255 map to their single-byte values
        self.vocab = {i: bytes([i]) for i in range(256)} if vocab is None else vocab
        self.merges = {} if merges is None else merges
        self.special_tokens = [] if special_tokens is None else special_tokens

    def train(self, text: str, vocab_size: int):
        """Train BPE on text until target vocab_size is reached."""
        pass

    def encode(self, text: str) -> list[int]:
        """Convert a string to a list of token IDs."""
        pass

    def decode(self, ids: list[int]) -> str:
        """Convert a list of token IDs back into a string."""
        pass


if __name__ == "__main__":
    test_ids = [1, 2, 3, 1, 2]
    print("Testing merge:")
    print(f"Original: {test_ids}")
    print(f"Merged (1, 2) -> 256: {merge(test_ids, (1, 2), 256)}")
