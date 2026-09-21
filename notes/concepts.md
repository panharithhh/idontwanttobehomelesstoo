# CS336 Assignment 1: Concept Notes

Study notes for Assignment 1 (Spring 2025). They contain concepts, questions and sanity checks, but **no code and no solutions**. The handout (`cs336_assignment1_basics.pdf`) is the source of truth. If these notes disagree with it, the handout wins.

Last updated: 2026-09-21 (Claude). These notes follow the handout's section *names*; section numbers are not verified yet.

---

## For Gemini: read this first

- Follow `AGENTS.md` strictly. Don't write Python or pseudocode, give solutions, edit files, run commands, or point to third-party implementations.
- Use this file as your reference and keep your answers consistent with it.
- Each topic below has **Concept**, **Check the handout**, **Ask the student** and **Sanity checks**. Ask one or two questions at a time. Don't read a whole section back to the student.
- When the student is stuck, prefer a sanity check or a question over an explanation.
- If a question isn't covered here, answer briefly if you're sure. Otherwise tell the student: "ask Claude to extend the notes on <topic>."

## How the loop works

1. The student writes code in `cs336_basics/` and wires it into `tests/adapters.py`.
2. Quick questions go to Gemini, which answers from this file.
3. For a gap or a deep question, the student asks Claude to "extend notes: <topic>", and Claude adds to this file.
4. Gemini picks up the new section the next time it reads the file.

## Map: topic → adapter in `tests/adapters.py`

| Topic | Adapter(s) |
|---|---|
| BPE training | `run_train_bpe` |
| Tokenizer encode/decode | `get_tokenizer` |
| Linear, Embedding | `run_linear`, `run_embedding` |
| RMSNorm | `run_rmsnorm` |
| SiLU, SwiGLU | `run_silu`, `run_swiglu` |
| RoPE | `run_rope` |
| Softmax, attention | `run_softmax`, `run_scaled_dot_product_attention` |
| Multi-head attention | `run_multihead_self_attention`, `run_multihead_self_attention_with_rope` |
| Transformer block, LM | `run_transformer_block`, `run_transformer_lm` |
| Loss, clipping | `run_cross_entropy`, `run_gradient_clipping` |
| Optimizer, schedule | `get_adamw_cls`, `run_get_lr_cosine_schedule` |
| Data loading | `run_get_batch` |
| Checkpointing | `run_save_checkpoint`, `run_load_checkpoint` |

Suggested order is the handout's order: tokenizer → model pieces from the bottom up → loss and optimizer → data and training loop → decoding → experiments.

Lecture pointers on cs336.stanford.edu: the **tokenization** lecture (sections 1–3 below), the **PyTorch and resource accounting** lecture (FLOPs, memory, einsum, optimizers) and the **architectures and hyperparameters** lecture (pre-norm, RMSNorm, SwiGLU, RoPE).

---

## 1. Unicode and bytes

**Concept**
- Text is a sequence of Unicode code points. UTF-8 turns each code point into 1–4 bytes, and ASCII characters take exactly 1 byte.
- Byte-level BPE starts from the 256 possible byte values, so every string can be represented and there is never an "unknown" token.

**Ask the student**
- Why start from bytes instead of Unicode code points? How big would the starting vocabulary be with code points?
- Why train on UTF-8 rather than UTF-16 or UTF-32? Think about sequence length and how common ASCII is.
- Why does decoding a UTF-8 string one byte at a time break for characters like "é" or an emoji?

**Sanity checks**
- In a REPL, compare the byte length and character length of "hello", "héllo" and an emoji.
- Find a two-byte sequence that is not valid UTF-8, and see what happens when you try to decode it.

---

## 2. BPE tokenizer training (`run_train_bpe`)

**Concept**
- **Initial vocabulary:** the 256 single bytes plus the special tokens.
- **Training loop:** repeatedly count every adjacent pair of tokens, merge the most frequent pair into a new token and record that merge. Stop when the vocabulary reaches `vocab_size`.
- **Output:** `vocab` (id → bytes) and `merges`, an *ordered* list of the pairs that were merged.
- **Pre-tokenization:** before counting, text is split into pre-tokens with the regex pattern the handout gives (the GPT-2 pattern). Merges never cross a pre-token boundary. Because identical pre-tokens behave identically, you count each distinct pre-token once and track how often it occurs.
- **Special tokens** (for example `<|endoftext|>`) mark document boundaries. They get their own ids and are never merged with anything. The handout explains how to keep merges from crossing them.
- **Ties:** when several pairs have the same count, the handout gives a deterministic tie-breaking rule. The tests depend on following it exactly.

**Check the handout**
- The worked "stylized example" (a small corpus of words like low, lower, widest and newest). Do the first few merges by hand and compare your answer with the handout's.
- The exact tie-breaking rule.
- How special tokens should be handled before pre-tokenization.
- The "Experimenting with BPE tokenizer training" section on parallelizing pre-tokenization. The repo also has a runnable pre-tokenization example in `cs336_basics/`.
- The same section on making the merge step faster.

**Ask the student**
- What does one pre-token look like in your data structure during training? Why might a sequence of `bytes` objects be easier to work with than a string?
- What is the difference between `vocab` and `merges`, and why does the order of `merges` matter?
- After you merge one pair, which pair counts can change? Do you need to recount pre-tokens that don't contain that pair?
- Is your time going into pre-tokenization or into merging? How would you find out?

**Sanity checks**
- Work out the first three merges by hand on a tiny corpus, then compare with your code.
- **Invariants:**
  - The final vocabulary size equals `vocab_size` exactly.
  - The number of merges equals `vocab_size` minus 256 minus the number of special tokens.
  - Each merged token's bytes appear in the vocabulary.
  - The vocabulary contains no duplicate entries.
  - No merge contains a special token.
- Run on the small TinyStories validation file before the full training set.
- If your merges match the reference for the first N steps and then diverge, look closely at tie-breaking and at how you update counts after a merge.
- Profile before optimizing (`cProfile`, or `scalene`, which the handout mentions).

---

## 3. Tokenizer encoding and decoding (`get_tokenizer`)

**Concept**
- **Encoding:**
  1. Separate out the special tokens.
  2. Pre-tokenize the remaining text with the same pattern used in training.
  3. Turn each pre-token into bytes.
  4. Apply the learned merges *in the order they were created*.
  5. Map the resulting tokens to ids.
- **Decoding:** join the bytes for each id and decode them as UTF-8. The handout says how to handle byte sequences that aren't valid UTF-8, so decoding never crashes.
- **Streaming:** `encode_iterable` should handle files too big to fit in memory, producing ids lazily without splitting a token across chunks.

**Ask the student**
- Why must encoding apply merges in training order? What would go wrong if you merged whatever pair is most frequent in the input text?
- Why is `decode(encode(text)) == text` guaranteed, while `encode(decode(ids)) == ids` is not?
- Suppose one special token is a prefix of another (for example, a doubled end-of-text token). Which should win when both match?
- In a stream, where is it safe to split the input without changing the resulting tokens?

**Sanity checks**
- Round-trip these inputs: an empty string, a single character, an emoji, text with special tokens next to each other, and repeated special tokens.
- When a test fails, shrink the input to the smallest string where your ids differ from the expected ones.
- For the tokenizer experiments, measure compression (bytes per token) and throughput. Ask why `uint16` is a big enough type to store the ids.

---

## 4. Model building blocks

**Rules from the handout:** you may not use most of `torch.nn`, `torch.nn.functional` or `torch.optim`. The handout lists the exceptions (things like `nn.Parameter`, the container classes and the `Optimizer` base class). Built-ins *are* fine in your own tests to compare against.

**Conventions to check in the handout:** how vectors and weights are oriented (the adapters store weights as `(d_out, d_in)`), whether layers have a bias, and how to initialize weights (the handout gives the distributions). Einsum and einops are encouraged.

### Linear and Embedding

- **Linear concept:** a matrix multiply with no bias. It must work with any number of leading batch dimensions.
- **Embedding concept:** a lookup table where each token id selects a row.
- **Ask the student:**
  - Given weights stored as `(d_out, d_in)` and input `(..., d_in)`, which dimension gets contracted?
  - Why is an embedding lookup equivalent to multiplying a one-hot vector by a matrix, and why not do it that way?
  - Why do modern LMs drop the bias?
- **Sanity checks:**
  - Feed an input with several leading dimensions and assert the output shape.
  - In a test, compare against the built-in layer after loading the same weights into it.

### RMSNorm

- **Concept:** rescale each vector by its root-mean-square over `d_model`, then multiply by a learned gain. There is no mean subtraction and no bias (unlike LayerNorm). The handout asks you to do the computation in higher precision and then cast back.
- **Ask the student:**
  - Which dimension do you reduce over, and how do you keep the shape so broadcasting works?
  - Why upcast before squaring?
  - What is `eps` protecting against?
- **Sanity checks:**
  - With gain set to 1, the output vectors have an RMS of about 1.
  - The output shape matches the input shape.
  - A low-precision input (bf16) still gives sensible values.

### SiLU and SwiGLU

- **Concept:** SiLU is a smooth ReLU-like activation. SwiGLU is a gated feed-forward layer: one projection passes through SiLU and is multiplied element-wise by a second projection, then a third projection maps the result back to `d_model`. The handout gives the exact equation and explains how to choose `d_ff` relative to `d_model`.
- **Ask the student:**
  - What does the gate add compared with a plain two-layer MLP?
  - Why is `d_ff` smaller than the classic 4× `d_model` when there are three matrices instead of two? Hint: compare parameter counts.
- **Sanity checks:**
  - Use the adapter's weight shapes to work out which of `w1`, `w2` and `w3` does what.
  - Assert the shapes after each step.

### RoPE (rotary position embeddings)

- **Concept:** rotate pairs of dimensions in the queries and keys by an angle that depends on the token's position and on the pair's frequency. Because of this, the query–key dot product depends only on *relative* position. RoPE has no learnable parameters, so the sin and cos values can be precomputed once and stored as a non-trainable buffer.
- **Ask the student:**
  - Why rotate queries and keys but not values?
  - Why does rotating both vectors make their dot product depend only on the distance between positions?
  - Which dimensions are paired together in *this* assignment's convention? Check the handout, since implementations differ.
  - What role does `token_positions` play?
- **Sanity checks:**
  - At position 0, RoPE leaves the input unchanged.
  - Vector norms are preserved.
  - The q·k score for positions (m, n) equals the score for (m+c, n+c).

### Softmax

- **Concept:** for numerical stability, subtract the maximum before exponentiating.
- **Ask the student:** why doesn't subtracting a constant change the result?
- **Sanity checks:**
  - Inputs like [1000, 1001] give no NaN or inf.
  - The output sums to 1 along the chosen dimension.

### Scaled dot-product attention

- **Concept:** scores are the dot products between queries and keys, scaled by the square root of the key dimension, then softmaxed and used to take a weighted average of the values. Masked positions must get zero probability. Check the adapter docstring and the handout for what True and False mean in the mask.
- **Ask the student:**
  - Why scale by the square root of `d_k`?
  - What shape is the score tensor?
  - Do you apply the mask before or after softmax, and what value do masked positions get?
- **Sanity checks:**
  - On a length-3 toy example, print the scores before and after masking.
  - Each row of attention probabilities sums to 1.
  - With a causal mask, the output at position 0 depends only on the first value.

### Multi-head self-attention

- **Concept:** compute Q, K and V for all heads in one matrix multiply each, split into heads, run attention on all heads at once as a batch, merge the heads and apply the output projection. Use a causal mask. With RoPE, rotate Q and K per head, using the head dimension.
- **Ask the student:**
  - The adapter docstring says how the projection weight rows are ordered by head. What does that tell you about how to split the heads?
  - Why is batching the heads faster than looping over them?
- **Sanity checks:**
  - **Causality:** changing a *future* token must not change any earlier output.
  - **Head batching:** in a test, compare your batched version against a slow loop over heads.

### Transformer block and full LM

- **Block concept (pre-norm):** the block has two sublayers, attention and then the feed-forward layer. Each sublayer normalizes its input and adds its result back onto a residual stream. The handout gives the exact equations.
- **LM concept:** token embeddings → N blocks → final norm → output projection → logits (not probabilities).
- **Ask the student:**
  - What is the residual stream, and why does pre-norm train more stably than post-norm?
  - Why return logits instead of softmax probabilities?
- **Sanity checks:**
  - Name your modules so their `state_dict` keys match the adapter docstrings (`ln1`, `attn.q_proj`, `ffn.w1`, `ln_final`, `lm_head`, …). Then weights load directly.
  - The output shape is (batch, seq, vocab).
  - The parameter count matches your own accounting.

### Transformer accounting (written problem)

- **Concept:** multiplying an (m×n) matrix by an (n×p) matrix costs about 2·m·n·p FLOPs. Count parameters and FLOPs for each part of the model.
- **Ask the student:**
  - Which parts dominate the FLOPs as `d_model` grows?
  - Which parts dominate as context length grows?

---

## 5. Loss and optimization

### Cross-entropy (`run_cross_entropy`)

- **Concept:** the loss is the negative log-probability of the correct next token, averaged over the batch. Compute it in a numerically stable way: cancel the log against the exp and subtract the max, rather than computing softmax and then taking its log. Perplexity is the exponential of the average loss.
- **Ask the student:** why can "softmax, then log" produce `-inf` or NaN?
- **Sanity checks:**
  - With all-equal logits, the loss equals log(vocab_size).
  - Very large logits don't produce NaN.

### Optimizer API and SGD example

- **Concept:** the handout walks through a simple SGD with a decaying learning rate to teach how the `Optimizer` base class works: `param_groups`, the per-parameter `state`, and `step`.
- **Ask the student:** what goes into `self.state[p]`, and why is it stored per parameter?
- **Learning-rate tuning problem:** try the handout's learning rates and watch for divergence.

### AdamW (`get_adamw_cls`)

- **Concept:** AdamW keeps running estimates of the gradient's mean and squared magnitude, uses a bias-corrected step size, and applies weight decay *directly to the parameters* (decoupled weight decay) rather than through the gradient. Follow the handout's algorithm exactly, in its order.
- **Ask the student:**
  - How is decoupled weight decay different from adding an L2 penalty to the loss when using Adam?
  - Does the step counter start at 0 or at 1, and where does that matter?
  - How much memory does AdamW add per parameter?
- **Sanity checks:**
  - In a test, compare a few steps on a tiny model against the built-in optimizer with the same hyperparameters.
- **AdamW accounting problem:** separately count memory for parameters, gradients, optimizer state and activations, then count FLOPs per training step.

### Cosine learning-rate schedule (`run_get_lr_cosine_schedule`)

- **Concept:**
  1. Linear warmup up to the maximum learning rate.
  2. Cosine decay down to the minimum.
  3. Constant at the minimum after that.
  The handout defines the three regions exactly.
- **Ask the student:**
  - Why use warmup at all?
  - What is the learning rate exactly at `it == warmup_iters` and at `it == cosine_cycle_iters`?
- **Sanity checks:** plot the learning rate against the iteration number and check the endpoints of each region.

### Gradient clipping (`run_gradient_clipping`)

- **Concept:** compute a single L2 norm over *all* parameters' gradients together. If it is above the limit, scale every gradient down by the same factor. The handout gives the small epsilon to use.
- **Ask the student:**
  - Why use one global norm instead of clipping each parameter separately?
  - What should happen to parameters whose gradient is `None`?
- **Sanity checks:** in a test, compare against PyTorch's built-in clipping utility.

---

## 6. Data loading, checkpointing and the training loop

### Data loader (`run_get_batch`)

- **Concept:** pick random starting positions in the long 1-D array of token ids. Each example is a window of `context_length` tokens, and its labels are the same window shifted one position to the right. For large datasets, memory-map the file instead of loading it all into RAM.
- **Ask the student:**
  - What is the largest valid starting index?
  - Why sample randomly instead of looping through the data in epochs?
  - Does the dtype you memory-map with match the dtype you saved?
- **Sanity checks:**
  - Labels equal the inputs shifted by one.
  - Every id is less than the vocabulary size.
  - The tensors are on the requested device.

### Checkpointing

- **Concept:** a checkpoint must hold everything needed to resume *exactly*: the model weights, the optimizer state and the iteration count.
- **Ask the student:** why do you need to save the optimizer state too?
- **Sanity checks:** save, load into a fresh model and optimizer, and take one step. The result should match continuing without reloading.

### Training loop

- **Concept:** a single script with configurable hyperparameters that:
  - loads the memory-mapped data
  - trains, checking validation loss periodically
  - logs results
  - saves checkpoints
- **Ask the student:**
  - How often will you evaluate, and on how many validation batches?
  - What will you log: loss against steps, against wall-clock time, or both?
- **Sanity checks:**
  - **Overfit one batch:** loss should drop close to zero. If it doesn't, the bug is in the model, the loss or the optimizer, not in the data.

---

## 7. Decoding and text generation

- **Concept:** generate one token at a time, sampling from the model's next-token distribution. **Temperature** sharpens or flattens that distribution. **Top-p (nucleus) sampling** keeps the smallest set of most-likely tokens whose total probability reaches p, then renormalizes. Stop at the end-of-text token or at a maximum length.
- **Ask the student:**
  - What happens as temperature approaches 0, and as it gets very large?
  - Why might top-p work better than always keeping a fixed top-k?
- **Sanity checks:**
  - A tiny temperature behaves like argmax.
  - p = 1 leaves the distribution unchanged.

---

## 8. Experiments

- **Concept:** the handout gives a baseline configuration for TinyStories and a validation-loss target, including a low-resource variant for CPU and Apple MPS. The experiments are:
  - a learning-rate sweep, including finding where training diverges
  - batch-size experiments
  - generating samples
  - ablations: removing RMSNorm, pre-norm vs. post-norm, no position embeddings vs. RoPE, and SiLU vs. SwiGLU with matched parameter counts
  - a run on OpenWebText
  - the leaderboard (check the handout for the current time limit)
- **Ask the student:**
  - What is your hypothesis before each run?
  - Are you changing only one variable at a time?
  - Will you compare runs by step or by wall-clock time?
- **Sanity checks:**
  - Log every run's config along with its results.
  - Do a quick short run before any long one.

---

## General debugging playbook

- Assert shapes at every module boundary.
- Use tiny inputs you can work out by hand.
- Run a single test at a time: `uv run pytest -k <test_name>`.
- Compare against PyTorch built-ins in your *tests*, never inside your implementation.
- Check dtype and device whenever results look strange.
- Overfit one batch before training for real.
- Profile before you optimize.
