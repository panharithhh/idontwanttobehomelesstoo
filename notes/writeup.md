# Assignment 1 writeup

My answers to the written problems in `cs336_assignment1_basics.pdf`. Deliverables are short: usually one or two sentences each.

How I use this file: copy the question in, run whatever the handout suggests in `cs336_basics/lol.ipynb`, write down what I actually observed, then answer in my own words.

---

## unicode1 — Understanding Unicode (1 point) — §2.1

**(a) What Unicode character does `chr(0)` return?**

Run: `chr(0)`

Observed: "x/00' 

Answer: It will return null

**(b) How does this character's `__repr__()` differ from its printed representation?**

Run: `chr(0)` on its own line, then `print(chr(0))`. Compare the two outputs.

Observed: chr(0) will return "\x00' as for print()  it will return nth 

Answer: the Reason is that it return as x/00 it's a way of python showing a charactre it cna't show you , as for the print statement since chr(0) represent null so the print will show in a str way which is nothing  

a.it returns a one-character string containing the null character (U+0000).  
b.  Since chr(0) represent null so print will show in a str way which is nothing Because it's a control character with no glyph. 

keep in mind 
**Control characters: 0–31, plus 127 (delete).**
**glyph isn';t a character it is a drawn shape of one**. 


**(c) What happens when this character occurs in text?**

Run: `"this is a test" + chr(0) + "string"`, then `print("this is a test" + chr(0) + "string")`.

Observed: so a have len of 21 and b have len of 20 

Answer: the reason is that print consider it as getting convered already as a empty value as for A since we just len() including the null which b donest have 

---

## unicode2 — Unicode encodings — §2.2

Copy the parts in from the handout when I get there.

Observed:

Answer:

---

## Later written problems

Add a section for each as I reach it. From the handout's problem boxes:

- [ ] `train_bpe_tinystories` — results and timing from training on TinyStories
- [ ] `train_bpe_expts_owt` — results on OpenWebText, and how the vocabularies differ
- [ ] `tokenizer_experiments` — compression ratio, throughput, choice of dtype
- [ ] `transformer_accounting` — parameter and FLOP counts
- [ ] `learning_rate_tuning` — what happens at each learning rate
- [ ] `adamwAccounting` — memory and FLOPs per training step
- [ ] Experiments and ablations — one section per run, with the config and the hypothesis written down *before* the run

---

## Notes to self

- Every number I report should come with how I measured it (which file, which machine, how many iterations).
- If an answer takes more than three sentences, I probably haven't understood it yet.
