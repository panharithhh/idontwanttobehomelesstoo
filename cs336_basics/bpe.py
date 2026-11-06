import regex as re 

def merge(ids, pair, idx):
    new_ids = {}
    for k,v in ids.items():
        
        i = 0

        new_chunk_ids = []

        while i < len(k): 
            if len(k) - i > 1 and pair[0] == k[i] and pair[1] == k[i+1]:
                new_chunk_ids.append(idx)
                i += 2

            else :
                new_chunk_ids.append(k[i]) 
                i += 1

        new_ids[tuple(new_chunk_ids)] = v

    return new_ids

pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

compiled = re.compile(pattern)

def pretokenize(data: str, special_token : list) -> list: 

    if special_token:

        tokens = sorted(special_token, key=len, reverse=True)
        toks =  "|".join(re.escape(t) for t in tokens)
        chunks = re.split(toks, data)

    else :
        chunks = [data]

    pre_tokens = []
    for chunk in chunks:
        #compiled is where the regex pattern is 
        pre_tokens.extend(m.group(0) for m in compiled.finditer(chunk))
        

    return pre_tokens


def get_stats(ids):
    # The goal is
    counts = {}

    for k,v in ids.items():
        for pair in list(zip(k, k[1:])):
            counts[pair] = counts.get(pair,0) + v

   
    return counts

def get_unique_pretoken(text):

    word_counts = {} 

    for word in text:
        word_bytes = word.encode("utf-8") 
        word_counts[word_bytes] = word_counts.get(word_bytes,0) +1

    return word_counts 


def train_bpe(input_path, vocab_size,special_tokens):

    special_tokens = special_tokens or []

    length_special_token = len(special_tokens)
    
    with open (input_path, encoding="utf-8") as f:
        data = f.read()

     
    pre_tokens = pretokenize(data, special_tokens)

    new_token_amount = vocab_size - 256 # new token amount
    
    idx = 255 

    ids = get_unique_pretoken(pre_tokens)
    
    vocab = {}

    merges = []

    for i in range(256):
        vocab[i] = bytes([i])

    for _ in range(new_token_amount - len(special_tokens)):  # main training
        stats = get_stats(ids) # tuples : counts 
        if not stats:
            break 
        top_pair = max(stats, key=lambda p: (stats[p], vocab[p[0]], vocab[p[1]])) # return a tuple
        
        idx += 1 
        ids = merge(ids,top_pair,idx)
        vocab[idx] = vocab[top_pair[0]] + vocab[top_pair[1]]
        merges.append((vocab[top_pair[0]], vocab[top_pair[1]]))


    for i in range(len(special_tokens)):
        special_token= str(special_tokens[i])
        vocab[1 + i + idx] = special_token.encode("utf-8")
        
    
    return vocab, merges 


def tokenizer():
    pass

# =====================================================================
# WHAT THE TESTS USE: only two "doors" (tests/adapters.py)
# Everything else (helpers) is up to me. Tests never call them directly.
# =====================================================================
#
# DOOR 1: run_train_bpe(input_path, vocab_size, special_tokens)
#   -> calls my train
#   in:   input_path      str | PathLike   path to the training text file
#         vocab_size      int              final vocab size, INCLUDING
#                                          256 bytes + merges + special tokens
#         special_tokens  list[str]        e.g. ["<|endoftext|>"]
#   out:  vocab           dict[int, bytes] id -> token bytes
#         merges          list[tuple[bytes, bytes]]  in order of creation
#   rule: only ONE place opens the file (adapter OR train, not both)
#   tests (tests/test_train_bpe.py):
#     - speed: corpus.en, vocab 500, must finish < 1.5 s
#     - merges must EXACTLY equal the reference list (order + tie-break)
#     - special tokens never merged into other tokens (no b"<|" inside)
#
# DOOR 2: get_tokenizer(vocab, merges, special_tokens=None)
#   -> returns an object of my Tokenizer class
#   class methods (handout p. 12):
#     __init__(self, vocab, merges, special_tokens=None)
#         vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]],
#         special_tokens: list[str] | None
#         (append special tokens to vocab if not already there)
#     from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)
#         classmethod: load saved vocab + merges, return a Tokenizer
#     encode(self, text: str) -> list[int]
#     encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]
#         yields ids lazily (big files, low memory)
#     decode(self, ids: list[int]) -> str
#         use errors="replace" for broken bytes
#   tests (tests/test_tokenizer.py):
#     - loaded with GPT-2's real vocab + ~50,000 merges
#     - roundtrip: decode(encode(text)) == text
#     - ids must match tiktoken exactly
#     - special tokens (incl. overlapping), empty string, unicode
#     - encode_iterable + memory limits
#
# HELPERS (mine, behind the doors):
#   pretokenize          text -> list of pre-token strings (used by train + encode)
#   get_unique_pretoken  pre-tokens -> {word ids: count}
#   get_stats            {word ids: count} -> {pair: count}
#   merge                {word ids: count}, pair, new id -> new {word ids: count}
#
# Run tests:  uv run pytest tests/test_train_bpe.py
#             uv run pytest tests/test_tokenizer.py
# =====================================================================