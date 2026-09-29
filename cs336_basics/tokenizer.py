import regex as re

def get_stats(ids)-> dict: # this is just 
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

pattern = re.compile("|".join([
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""\p{N}{1,3}""",
    r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
    r"""\s*[\r\n]+""",
    r"""\s+(?!\S)""",
    r"""\s+""",
]))


def train(text, vocab_size, verbose=False):

    #merges: list[tuple[bytes, bytes]]
    text = pattern.findall(text)
    ids = list(text.encode("utf-8"))                 # raw bytes as ints 0..255
    vocab = {i: bytes([i]) for i in range(256)}
    # 257 : b "cd" 
    merges_list = []
    merges = {} # (97,98) -> 257 

    for idx in range(256, vocab_size):
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

        t = pattern.findall(text) 
        

    return vocab, merges_list


pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

# pattern = re.compile("|".join([
#     r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
#     r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
#     r"""\p{N}{1,3}""",
#     r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
#     r"""\s*[\r\n]+""",
#     r"""\s+(?!\S)""",
#     r"""\s+""",
# ])) gpt 4o regex pattern

def encode():
    pass

def decode():
    pass

# Piece of the pattern	Matches	Example
# '(?:[sdmt]|ll|ve|re)	contraction endings	's, 'll, 're
# ?\p{L}+	an optional space, then a run of letters	" text", "some"
# ?\p{N}+	an optional space, then a run of digits	" 100"
# ?[^\s\p{L}\p{N}]+	an optional space, then a run of anything that isn't a space, letter or digit (punctuation)	"!!", " -"
# \s+(?!\S) and \s+	leftover whitespace, like double spaces or trailing spaces	" "

with open("/Users/cheapanharith/AI/standform-LLMSCRATCH/idontwanttobehomelesstoo/data/TinyStoriesV2-GPT4-train.txt", "r", encoding="utf-8") as f:
    text = f.read()

vocab, merges = train(text,  260)
print(merges)      
print(vocab[258]) 
