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

import enum
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
       
        top_pair = max(stats, key=lambda p: (stats[p], vocab[p[0]], vocab[p[1]])) 

        idx += 1 
        ids = merge(ids,top_pair,idx)
        vocab[idx] = vocab[top_pair[0]] + vocab[top_pair[1]]
        merges.append((vocab[top_pair[0]], vocab[top_pair[1]]))


    for i in range(len(special_tokens)):
        special_token= str(special_tokens[i])
        vocab[1 + i + idx] = special_token.encode("utf-8")
        

    # last
    
    return vocab, merges 

token_ids = []
pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
re_chunk = []


    
class Tokenizer:
    def __init__(self,vocab,merges,special_tokens=None):
        self.vocab = vocab
        self.rev_vocab = {b: i for i,b in vocab.items()}
        self.ranks = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = special_tokens or []

        self.list_specialtoken_enc = []

        for i in range(len(self.special_tokens)):
            self.list_specialtoken_enc.append(self.special_tokens[i].encode("utf-8")) 

    def encode(self, text):
        token_ids = []
 
        special_pattern ="|".join(re.escape(t) for t in self.special_tokens)

        pieces = re.split(f"({special_pattern})",text )

        piece_bytes = []

        for i in range(len(pieces)):
            
            piece_bytes.append(pieces[i].encode("utf-8"))

        for i in range(len(piece_bytes)):
            if piece_bytes[i] in self.list_specialtoken_enc:
                token_ids.append(self.rev_vocab[piece_bytes[i]] )
            else:  

                # token_ids.append(pieces[i])
                chunk = pieces[i] 

                pre_tokens = pretokenize(chunk,False)


                for chunk in pre_tokens:
                    parts = [bytes([b]) for b in chunk.encode("utf-8")]

                    while len(parts) > 1:
                        best = min(zip(parts, parts[1:]), key=lambda p: self.ranks.get(p, float("inf")))
                        if best not in self.ranks:
                            break

                        i = 0
                        while i < len(parts) - 1:
                            if (parts[i], parts[i + 1]) == best:
                                parts[i:i + 2] = [parts[i] + parts[i + 1]]
                            i += 1
                    token_ids.extend(self.rev_vocab[p] for p in parts)
        
        return token_ids 

    def decode(self, ids):
        c = b''.join(self.vocab[token_id] for token_id in ids)
        text = c.decode("utf-8", errors="replace")
        return text