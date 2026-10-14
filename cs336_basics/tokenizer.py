import regex as re

def get_stats(ids)-> dict: # this is just 
    counts = {}
    for i in range(len(ids)):
        for pair in zip(ids[i], ids[i][1:]):
            counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):

    out = [[] for _ in range (len(ids))]
    for n in range(len(ids)):
        ids_n = ids[n]
        i = 0
        while i < len(ids_n):
            if i < len(ids_n) - 1 and ids_n[i] == pair[0] and ids_n[i + 1] == pair[1]:
                out[n].append(idx)
                i += 2
            else:
                out[n].append(ids_n[i])
                i += 1
    return out

pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def train(text, vocab_size, special_tokens,verbose=False): 

    remove_st = "|".join(map(re.escape, special_tokens))

    pieces = re.split(remove_st, text) 
 
    pre_tokens = [] # This just have no more of the special token 
    for piece in pieces:
        pre_tokens.extend(pattern.findall(piece)) # this create a list of new word


    ids = list(chunk.encode("utf-8") for chunk in pre_tokens)

    vocab = {i: bytes([i]) for i in range(256)}
    merges_list = []
    merges = {} # (97,98) -> 257 

    for idx in range(256, vocab_size - len(special_tokens)): # 256 -> 398 
        stats = get_stats(ids)
        if not stats:                                # nothing left to merge
            break
        pair = max(stats, key=lambda p: (stats[p], vocab[p[0]], vocab[p[1]]), )        # tie breaking base on the bytes if want to go ofr the lower use - ? 
        ids = merge(ids, pair, idx)
        merges[pair] = idx
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        merges_list.append((vocab[pair[0]], vocab[pair[1]]))
        if verbose:
            print(f"merge {idx - 255}/{vocab_size - 256}: {pair} -> {idx} {vocab[idx]} ({stats[pair]}x)")
 
    for i in range(len(special_tokens)):

        vocab[vocab_size-i-1] = special_tokens[i].encode("utf-8")
 
    return vocab, merges_list