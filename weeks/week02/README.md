# Week 2 — Tokenisation from Scratch

> Part of the **[Build an SLM from Scratch](../README.md)** series.
> Companion LinkedIn post: *[link to your post]*

---

## What we built

A **character-level tokeniser** in pure Python — the complete data pipeline that converts raw Shakespeare text into a stream of integers ready for training.

Two core functions:
- `encode("hello")` → `[46, 43, 50, 50, 53]`
- `decode([46, 43, 50, 50, 53])` → `"hello"`

Plus a train/validation split and a batch sampler that generates `(input, target)` pairs for training.

---

## The core idea

Before a model can learn anything, it needs to answer a deceptively simple question:

> *What is a character, in computational terms?*

The answer: a number. Every unique character in Shakespeare gets a unique integer ID. The model never sees letters — it only ever sees integers.

This mapping is called the **vocabulary**. Shakespeare uses 65 unique characters — letters, punctuation, spaces, newlines. So our vocabulary has 65 entries.

```
' ' → 0
'!' → 1
'$' → 2
...
'z' → 64
```

Everything the model learns, everything it predicts, everything it generates — it all happens in this integer space. `encode()` and `decode()` are the translators between the human world and the model's world.

---

## What the pipeline produces

```
Original text  : "To be, or not to be"
Encoded        : [58, 53, 1, 40, 43, 6, 1, 53, 56, 1, 52, 53, 58, 1, 58, 53, 1, 40, 43]

Total dataset  : 1,115,394 integers
Train split    : 1,003,854 integers  (90%)
Val split      :   111,540 integers  (10%)
```

---

## How batching works

A batch is a set of `(input, target)` pairs sampled randomly from the training data.

For a sequence starting at position `i` with `block_size = 4`:

```
input  = data[i   : i+4]   →  "hell"  →  [46, 43, 50, 50]
target = data[i+1 : i+5]   →  "ello"  →  [43, 50, 50, 53]
```

One sequence of length `block_size` gives the model `block_size` training examples — at each position, it learns to predict the next character given everything before it.

---

## How to run it

```bash
python3 week02/tokeniser.py
```

It will:
1. Load (or download) Shakespeare
2. Build the vocabulary
3. Encode the full text
4. Verify the round-trip: `decode(encode(text)) == text`
5. Show a sample batch with inputs and targets

Requires Python 3.10+. No external dependencies.

---

## File structure

```
week02/
├── tokeniser.py    ← vocabulary, encode/decode, batch sampler
└── README.md       ← you are here
```

---

## Key concepts introduced

| Concept | What it means |
|---------|---------------|
| **Vocabulary** | The complete set of unique tokens the model can see and produce |
| **Encoding** | Converting a string into a list of integer token IDs |
| **Decoding** | Converting a list of integer token IDs back into a string |
| **Train/val split** | Holding back 10% of data to measure whether the model genuinely learned |
| **Block size** | The number of characters the model sees at once (context window) |
| **Batch** | A set of independent sequences processed together in one training step |

---

## Why we don't shuffle the data

Order matters in language. If we shuffled the dataset before training, the model would see `"To be"` in one batch and `", or not to be"` in a completely different batch — with no connection between them. The sequential structure of the text is part of what the model learns from.

We shuffle *which position we start from* (random batch sampling), but we never break up the natural order of the text itself.

---

## What's next

**Week 3 — The Bigram Model (Trainable)**

Now that we have a proper data pipeline, we replace the fixed probability table from week 1 with a **trainable weight matrix**. Instead of counting pairs and normalising, we store learnable parameters and optimise them. The model gets better over time — not by counting, but by adjusting.

→ [Week 3](../week03/README.md)

---

*Built with ❤️ and pure Python by [Kiran](https://linkedin.com/in/kirankumargosu) | [gosulab](https://kirangosu.com)*