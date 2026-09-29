def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train(text, vocab_size, verbose=False):

    ids = list(text.encode("utf-8"))                 # raw bytes as ints 0..255
    vocab = {i: bytes([i]) for i in range(256)}
    merges = {}

    for idx in range(256, vocab_size):
        stats = get_stats(ids)
        if not stats:                                # nothing left to merge
            break
        pair = max(stats, key=stats.get)             # most frequent pair
        ids = merge(ids, pair, idx)
        merges[pair] = idx
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        if verbose:
            print(f"merge {idx - 255}/{vocab_size - 256}: {pair} -> {idx} {vocab[idx]} ({stats[pair]}x)")

    return vocab, merges


vocab, merges = train("aaabdaaabac", 259)
print(merges)      
print(vocab[258])  