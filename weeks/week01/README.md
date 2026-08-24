# Week 1 — What Is a Language Model, Really?

> Part of the **[Build an SLM from Scratch](../README.md)** series.
> Companion LinkedIn post: *[link to your post]*

---

## What we built

A **character-level bigram language model** in pure Python. No ML frameworks. No neural networks. Just counting and probability.

It reads 1.1 million characters of Shakespeare, learns which characters tend to follow which, and uses that to generate new text — one character at a time.

The output is terrible. That's the point.

---

## The core idea

A language model answers one question:

> *Given what came before, what comes next?*

The bigram model answers the simplest possible version of that question:

> *Given the **one** character I just saw, what character is most likely to come next?*

It does this by building a **65 × 65 probability table** — one row and one column for every unique character in Shakespeare. Each cell holds the probability of transitioning from character `i` to character `j`.

```
counts['t']['h'] = 23,481   →   prob['t']['h'] = 0.31
counts['q']['u'] = 1,832    →   prob['q']['u'] = 0.90
```

To generate text: start with any character, look up its row, sample the next character from those probabilities, repeat.

---

## What the model actually learned

```
'q'  →  'u'    90.5%   ████████████████████████████████████
':'  →  '\n'   84.4%   █████████████████████████████████
'v'  →  'e'    76.2%   ██████████████████████████████
'Q'  →  'U'    73.3%   █████████████████████████████
','  →  ' '    70.8%   ████████████████████████████
```

Nobody told it that `q` is almost always followed by `u`. Nobody explained punctuation rules. It learned this purely from counting. That's the key insight — **structure emerges from data**.

---

## Sample output

```
O e sit iced bor no the we athiseveereren ce calld cethe s hy.
Prof f ctominthalilite y a aron hy'to sinar intoin wolll thin
BAstom ts l th:
KETharmamy.
```

Gibberish — but structured gibberish. Notice:
- Spaces appear with realistic frequency
- Punctuation follows roughly correct patterns (`:` tends to end a line)
- `qu` appears together consistently
- Capital letters tend to start new lines

The model doesn't know what words mean. But it knows the shape of the language.

---

## How to run it

```bash
# Clone the repo
git clone https://github.com/kirankumargosu/gosu-slm
cd gosu-slm

# Run week 1 — downloads Shakespeare automatically
python3 week01/bigram.py
```

Requires Python 3.10+. No external dependencies.

---

## File structure

```
week01/
├── bigram.py    ← the full model (~130 lines, heavily commented)
└── README.md    ← you are here

data/
└── shakespeare.txt    ← auto-downloaded on first run
```

---

## Key concepts introduced

| Concept | What it means |
|---------|---------------|
| **Bigram** | A pair of consecutive characters |
| **Vocabulary** | The set of all unique characters in the training text |
| **Probability table** | A matrix of transition probabilities between characters |
| **Laplace smoothing** | Adding 1 to all counts to avoid zero probabilities |
| **Sampling** | Picking the next character according to a probability distribution |

---

## The limitation that drives everything forward

The bigram model only looks **one character back**.

Every prediction is made in complete isolation from context. It has no memory. It doesn't know that it just wrote `"To be, or not"` — the next character depends only on the last one.

This is why the output is incoherent. Real language has **long-range dependencies**. What you write now depends on what you wrote five, ten, fifty characters ago.

Fixing this is what the rest of the series is about.

---

## What's next

**Week 2 — Tokenisation from Scratch**

Before we can give the model more context, we need to understand how text becomes numbers — properly. We'll build a full character tokeniser: `encode()`, `decode()`, the vocabulary, and a data preparation pipeline that turns Shakespeare into a stream of integers ready for training.

→ [Week 2](../week02/README.md)

---

*Built with ❤️ and pure Python by [Kiran](https://linkedin.com/in/kirankumargosu) | [gosulab](https://kirangosu.com)*
