# `run_train_bpe` — design brief & question dump

Working notes, not assignment code. Written to be handed to a fast model (Gemini) in the
side pane while you write the implementation yourself.

---

## 0. How this split works

- **Claude (slow, this file):** hard questions, spec archaeology, design pressure, code review.
- **Gemini (fast, side pane):** interrogate it with the questions below, get explanations
  fast, argue with it about tradeoffs, have it re-explain anything here you can't answer.
- **You:** write every line of `cs336_basics/tokenizer.py`.

Rule that makes this work: **do not accept an answer from either model you can't re-derive
on a 3-word toy corpus by hand.** The tests below are exact-match — a design you don't
understand will fail at merge #170 and you'll have no idea why.

---

## 1. Ground truth from this repo

Don't let any model guess these. Verified from the files:

| Fact | Source |
|---|---|
| Signature: `(input_path, vocab_size: int, special_tokens: list[str], **kwargs)` | `tests/adapters.py:565` |
| Returns `tuple[dict[int, bytes], list[tuple[bytes, bytes]]]` | same |
| `merges` compared with `==` — **exact order, exact length** | `tests/test_train_bpe.py:49` |
| `vocab` compared as `set(keys)` and `set(values)` — IDs need not match the reference positionally | `tests/test_train_bpe.py:60-62` |
| Speed budget: **1.5 s** on `tests/fixtures/corpus.en` (130 KB). Reference impl: 0.38 s. "Toy" impl: ~3 s | `tests/test_train_bpe.py:8-25` |
| Every non-special vocab entry must satisfy `b"<|" not in word_bytes` | `tests/test_train_bpe.py:78-80` |
| Second correctness test runs on **5 MB** of TinyStories against a pickled snapshot | `tests/test_train_bpe.py:65` |
| `regex` (not `re`) is a declared dependency, commented "more powerful regex than the builtin `re`" | `pyproject.toml:16` |
| `find_chunk_boundaries(file, desired_num_chunks, split_special_token)` is given to you | `cs336_basics/pretokenization_example.py` |

## 2. Arithmetic to state before writing a line

The reference merges file has **243** lines. The test asks for `vocab_size=500` with
`special_tokens=["<|endoftext|>"]`.

- Write the equation that produces 243 from 500. Three terms.
- Now: what should happen if someone calls this with `vocab_size=200`? With
  `vocab_size=257, special_tokens=["<|endoftext|>"]`? Is the loop bound a count of merges
  or a target size — and do those differ at the edges?
- `len(vocab)` at the end should equal what, exactly? Is that assertion cheap enough to
  leave in permanently?

---

## 3. The question dump

### A. Output contract

1. `vocab` is `dict[int, bytes]` and the test only checks **set** equality of keys and
   values. So ID assignment is free — but is it? Your `Tokenizer` class
   (`tests/adapters.py:542`) has to encode/decode with this same vocab. Name one ID
   convention that passes `test_train_bpe` but makes `encode`/`decode` painful later.
2. Where do the special token IDs sit — before the 256 byte tokens, or after all merges?
   Does the answer change if a *later* training run adds a second special token?
3. `merges` entries are `tuple[bytes, bytes]` — the two *pre-merge* tokens. Is the merged
   result recoverable from the tuple alone? What does that imply about whether `merges` and
   `vocab` can be built in a single pass or need two?
4. The reference merges file is serialized through `gpt2_bytes_to_unicode()`
   (`tests/common.py`) — the `Ġ` you see for a space. Is that mapping part of *your*
   implementation, or purely the on-disk format of the fixture? Justify from the test code:
   who calls `gpt2_byte_decoder`?

### B. Tie-breaking — the trap that costs people a day

5. Two candidate pairs have the same maximum count. Go find the exact rule in the handout
   (§2.5 area, the BPE training spec). Write it down verbatim before proceeding.
6. That rule implies a specific Python comparison. On `tuple[bytes, bytes]`, what does `>`
   actually compare — byte values, or something else? Try `(b"Z", b"a") > (b"A", b"z")` in a
   REPL and explain the result.
7. If your tie-break is wrong, does merge #1 differ from the reference, or merge #170? What
   does the **index of first divergence** tell you about which stage is buggy?
8. Construct a corpus of ≤4 short words where "highest count, first seen wins" and the
   handout's rule produce *different* merge #1. If you can't construct one, you don't
   understand the rule yet.
9. Is the tie-break ever load-bearing on a large corpus, or only on tiny ones? (Think about
   how many distinct pairs share a count of 2 in 5 MB of text.)

### C. Pretokenization

10. Why does BPE here never merge a pair that spans two pretokens? What would break in
    `encode` later if training ignored pretoken boundaries?
11. You apply the GPT-2 pattern to — `str` or `bytes`? Walk through what happens to the
    German fixture (`tests/fixtures/german.txt`) under each choice. Where does the
    UTF-8 → bytes conversion have to happen relative to the regex?
12. Why is `regex` a dependency instead of `re`? Find the specific construct in the GPT-2
    pattern that `re` rejects. (Don't take a model's word for it — paste the pattern into
    both and see.)
13. Should the regex be compiled once at module level or per call? Measure it on 130 KB
    before you decide it doesn't matter.
14. Counting: `dict[pretoken, count]` vs. a flat list of every pretoken occurrence. On
    TinyStories 2 GB, estimate the ratio between total pretokens and *distinct* pretokens.
    That ratio is the entire speedup — what is it, roughly, and how would you measure it
    cheaply on the 5 MB fixture?
15. What is a pretoken's internal representation while merging — `bytes`? `tuple[bytes,...]`?
    `list[int]` of token IDs? Score each on: cost of one merge step, cost of counting pairs,
    cost of producing the final vocab.

### D. Special tokens

16. The docstring says special tokens "will never be split into multiple tokens" *and* "if
    these special tokens occur in `input_path`, they are treated as any other string."
    Those look contradictory. Which clause is about **training** and which is about
    **encoding**? Re-read it until you can state the training behavior in one sentence.
17. `test_train_bpe_special_tokens` asserts `b"<|" not in word_bytes` for every non-special
    vocab entry. What implementation mistake does that catch? Sketch the merge sequence that
    would produce a vocab entry containing `b"<|"`.
18. Should `<|endoftext|>` bytes participate in pair counting at all? If not, at what stage
    do they leave the pipeline — before pretokenization, during, or after?
19. `find_chunk_boundaries` takes a `split_special_token`. Is splitting on it a *performance*
    convenience or a *correctness* requirement? Answer by asking what happens to a pair that
    straddles a chunk boundary in each case.
20. Two special tokens where one is a prefix of the other (`<|eot|>`, `<|eot_id|>`) — does
    that matter during training? During encoding? (`test_overlapping_special_tokens` at
    `tests/test_tokenizer.py:248` exists for a reason.)

### E. Speed — the 1.5 s budget

21. Naive loop: recount all pairs from scratch after each merge. Write the complexity in
    terms of (corpus size N, merges V). Plug in N for `corpus.en` and V=243. Why does that
    land at ~3 s and not 0.38 s?
22. **The key question.** You just merged pair `(a, b)` inside one pretoken. Exactly which
    pair counts change, and by how much? Work it out by hand on the token sequence
    `a b a b a` — count pairs before, apply the merge, count after, and diff. Watch the
    overlapping case carefully; it's where the off-by-one lives.
23. Given #22, what index do you need to avoid rescanning the whole corpus — pair →
    what, exactly? What's the cost of *maintaining* that index per merge?
24. Do you need a priority queue, or is `max()` over a counter fine at V=243? At what corpus
    size does the answer flip? (Note: a heap with stale entries needs a validity check — is
    that complexity worth it here?)
25. Before optimizing anything: what fraction of your runtime on `corpus.en` is
    pretokenization vs. the merge loop? How do you measure that without a profiler? With one?
26. Which of these actually matters at 130 KB, and which only at 2 GB: pair-index,
    dedup-by-count, parallelism, heap? Rank them by expected payoff per line of code.

### F. Parallelism

27. Of the two phases — pretokenize/count, and the merge loop — which parallelizes and which
    is inherently sequential? Why is the second one sequential *by definition* of BPE?
28. What does a worker process return, and how do you combine results from 4 workers? Is the
    combination associative? Commutative? Does that matter for determinism?
29. Process startup on macOS is spawn, not fork. On a **130 KB** file with a 1.5 s budget,
    does `multiprocessing` help or hurt? What's your plan for making it conditional?
30. If workers return dicts in nondeterministic completion order, can your final merge list
    change run to run? Trace it: where could iteration order leak into a tie-break?

### G. Karpathy-style vs. this spec

You said you're following a minimal byte-level BPE. Three ways that style typically differs
from this assignment. For each, decide whether it passes or silently fails:

31. **No regex pretokenization** — merges run over the whole text. Which test catches this,
    and at roughly which merge index would the output start to diverge?
32. **Tie-break = whichever pair `max()` happens to hit first.** Does `test_train_bpe` catch
    it? Would it catch it *reliably*, or only on this particular corpus?
33. **Tokens as `int` IDs throughout, vocab as `dict[int, bytes]` built at the end.** Is this
    compatible with the required return type? Is it *better* than carrying `bytes` around?
    (This one may be a genuine improvement — argue it.)
34. Minimal implementations usually train on one string held in memory. This one takes a
    `input_path` and later gets pointed at a 2 GB file. What changes?

---

## 4. Debugging ladder

Build this before you need it, in order:

1. Toy corpus (3–4 words, repeated), first **3** merges computed by hand on paper. Assert.
2. `len(vocab) == vocab_size`, and all 256 single-byte tokens present.
3. `len(merges) == vocab_size - 256 - len(special_tokens)`.
4. **First-divergence index** against the reference merges: compare element-wise, print the
   first index where yours differs plus the 3 merges on either side from both lists.
   - Divergence at index 0–5 → pretokenization or byte handling.
   - Divergence deep in the list → count-update or tie-break.
   - Same merges, wrong vocab → ID assignment or the special-token insertion point.
5. Only once the 130 KB test passes: run the 5 MB snapshot test.
6. Only once both pass: touch the speed test. Correct-then-fast, never the reverse.

---

## 5. Paste this into the Gemini pane to frame the session

> I'm implementing BPE training for Stanford CS336 assignment 1. Course policy: you may
> explain, question, and review, but you must not write the implementation for me — I write
> every line.
>
> Constraints my code must satisfy (verified, don't second-guess them): returns
> `tuple[dict[int, bytes], list[tuple[bytes, bytes]]]`; the merges list is compared for exact
> equality against a reference, so order and tie-breaking must match the spec exactly; 243
> merges for vocab_size=500 with one special token; must train on a 130 KB corpus in under
> 1.5 s; no vocab entry other than the special token may contain the bytes `<|`.
>
> I'll paste questions from a design brief. For each: answer it, then ask me a follow-up that
> checks whether I actually understood, and tell me what a wrong answer would look like
> downstream. If I ask you for code, give me the conceptual step and a test I can write
> instead. Be blunt when my design is wrong.

Then feed it §3 a block at a time. Argue back. If Gemini and this file disagree, that's the
interesting case — bring it to Claude.

---

## 6. Checkpoints to bring back to Claude

- After §2 + §3A–D answered, **before** writing code: your design decisions in prose
  (representation, ID scheme, where specials get removed, tie-break rule). Cheapest possible
  place to catch a wrong turn.
- When merges match the reference but the speed test fails: bring timings, not code.
- When the first-divergence index is nonzero: bring the index + surrounding merges.
- When it all passes: bring the file for review — invariants, edge cases, and what'll break
  when you point it at 2 GB.
