# Glossary

Plain-language meanings for the terms that come up in Assignment 1. Add to it whenever a new word trips you up.

## Text and characters

**Character** — one symbol of text: `a`, `牛`, `é`, a space. Not the same as one byte.

**Unicode** — the standard that assigns a number to every character in every script.

**Code point** — the number Unicode assigns to a character. `s` is 115, written `U+0073` (hex). `chr(n)` goes number → character, `ord(c)` goes character → number.

**Hex (hexadecimal)** — counting in base 16, digits `0`–`9` then `a`–`f`. Two hex digits cover 0–255, exactly one byte. `U+0073` and `\x00` are both hex.

**Glyph** — the shape drawn on screen for a character. The character is the idea, the glyph is the picture.

**Printable character** — one that has a glyph, like `A` or `?`.

**Control character** — a code point meant to *do* something rather than be drawn. Code points 0–31 and 127. Examples: newline (10) moves to the next line, tab (9) advances, null (0) does nothing at all.

**Null character** — code point 0, `chr(0)`, repr `'\x00'`. A control character with no glyph. Not the same as Python's `None` and not an empty string.

**ASCII** — the original 128 characters (code points 0–127): English letters, digits, punctuation and the control characters. Unicode's first 128 code points match it.

## Bytes and encodings

**Bit** — a single 0 or 1.

**Byte** — 8 bits, so 2^8 = 256 possible values, 0 through 255. Files on disk are sequences of bytes.

**Encoding** — a rule for turning characters into bytes. Decoding is the reverse.

**UTF-8** — the encoding this course uses. Variable width: ASCII characters take 1 byte, others take 2–4. `é` becomes the two bytes 195, 169.

**UTF-16 / UTF-32** — other encodings of the same Unicode characters, using wider fixed-ish units. They make text bigger and are not what we train on.

**`str`** — Python's type for text, a sequence of characters. Written `"hi"`.

**`bytes`** — Python's type for raw bytes, written `b"hi"`. Indexing one gives an **integer** 0–255, not a character.

**`.encode()` / `.decode()`** — `str.encode("utf-8")` gives bytes; `bytes.decode("utf-8")` gives text back. Decoding can fail if the bytes aren't valid UTF-8.

**Escape sequence** — how Python writes a character it can't display, starting with a backslash. `\x00` means "the character with hex code 00". `\n` is newline, `\t` is tab.

## Python mechanics

**`repr`** — the unambiguous text form of an object, meant for programmers, ideally valid Python. A notebook shows the repr of a cell's last expression, which is why strings appear with quotes.

**`str()` / printed form** — the friendly form, what `print()` writes out. No quotes, escapes resolved.

**Context manager** — an object that sets something up and cleans it up afterwards. The `with` statement drives it. `with open(...) as f:` guarantees the file is closed, even if an error is raised.

**Working directory** — the folder a process treats as "here". Relative paths are resolved from it. A notebook kernel and a terminal each have their own, and they are often different (`%pwd` in a notebook, `pwd` in a shell).

**Absolute vs relative path** — absolute starts from `/` and always means the same file. Relative is interpreted from the working directory.

**Memory-mapped file (`np.memmap`)** — a file that acts like an array without being fully loaded into RAM. The OS pages in pieces as you touch them.

## Tokenizer terms (coming next)

**Token** — one unit in the model's vocabulary. Here, a sequence of bytes.

**Token id** — the integer standing for a token. Models consume ids, not text.

**Vocabulary** — the mapping from id to token bytes. Byte-level BPE starts with 256 entries, one per byte value, plus special tokens.

**Special token** — a token added by hand, never produced by merging, such as `<|endoftext|>` marking a document boundary.

**Pre-token** — a chunk of text (roughly a word with its leading space) produced by splitting with a regex before any merging. Merges never cross pre-token boundaries.

**Merge** — one step of BPE training: the most frequent adjacent pair of tokens becomes a single new token. The ordered list of merges is part of the trained tokenizer.

**BPE (byte-pair encoding)** — the algorithm: start from bytes, repeatedly merge the most frequent adjacent pair until the vocabulary is the size you asked for.

**Compression ratio** — bytes of input divided by tokens produced. Higher means each token carries more text.
