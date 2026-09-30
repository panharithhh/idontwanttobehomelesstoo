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
    text = pattern.findall(text) # this will return a list of string 
    ids = [list(chunk.encode("utf-8") for chunk in text)]
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

        

    return vocab, merges_list


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

# with open("/Users/cheapanharith/AI/standform-LLMSCRATCH/idontwanttobehomelesstoo/data/TinyStoriesV2-GPT4-train.txt", "r", encoding="utf-8") as f:
#     text = f.read()

text = "cd cd abab"
vocab, merges = train(text,  258)
print(merges)      


# Pre-tokenization pattern: 7 alternatives, tried in order at each position; the first one that matches wins.
# \p{...} is a Unicode category (needs the `regex` module, not stdlib `re`):
#   L letter, Lu upper, Ll lower, Lt titlecase, Lm modifier letter, Lo other letter (CJK etc.), N number, M combining mark
#   "upper-ish" below = Lu Lt Lm Lo M, "lower-ish" = Ll Lm Lo M
#
# 1. optional leading char (not newline/letter/digit: a space, quote, punctuation...) + optional upper-ish letters
#    + 1+ lower-ish letters + optional contraction ('s 't 're 've 'm 'll 'd, any case)
#    -> "Hello", " world", "don't"; also splits camelCase where lower meets upper
# 2. same, but 1+ upper-ish letters and the lower-ish part is optional -> ALL-CAPS words like " NASA"
# 3. 1 to 3 digits, no leading space attached -> "12345" becomes "123", "45"
# 4. optional space + 1+ chars that are not whitespace/letter/digit (punctuation, symbols, emoji)
#    + optional trailing \r \n or /  -> " ...", "!!\n"
# 5. a whitespace run up through its last newline -> "\n\n", "  \n"
# 6. a whitespace run, but if a non-space char follows it stops one short so the last space can attach to the next token
# 7. any leftover whitespace (fallback)
