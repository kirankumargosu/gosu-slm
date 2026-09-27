# Week 6 — Neural Networks by Hand: The MLP

> Part of the **[Build an SLM from Scratch](../README.md)** series.
> Companion LinkedIn post: *[link to your post]*

---

## What we built

A **Multi-Layer Perceptron (MLP) language model** with an 8-character context window — implemented entirely in pure Python with full manual backpropagation. No frameworks, no autograd, no shortcuts.

The context window expanding from 1 to 8 characters produces the largest quality jump in the series so far.

---

## The core idea

The bigram model treats every prediction as independent. It sees one character and guesses the next. It has no memory of what came before that character.

An MLP with a context window changes this. Instead of looking at one character, we look at the **last 8 characters simultaneously** — embedding each one into a vector, concatenating all 8 vectors into a single flat input, and passing that through a hidden layer before predicting.

```
context = ['T', 'o', ' ', 'b', 'e', ',', ' ', 'o']
         → embed each character  → 8 vectors of size 24
         → concatenate           → 1 flat vector of size 192
         → Linear + Tanh         → hidden vector of size 128
         → Linear                → logits of size 65
         → softmax + sample      → next character
```

The model now has **192 numbers of context** to work with instead of just 1.

---

## Why tanh?

The tanh activation is the first non-linearity we've introduced. Without it, stacking two linear layers is mathematically identical to having one linear layer — the whole network collapses to a single matrix multiply.

Non-linearity is what gives neural networks their expressive power:

```
tanh(x) maps any real number to (-1, 1)
tanh'(x) = 1 - tanh(x)²    ← used in backward pass
```

---

## Sample output after 3000 steps

**Week 3 bigram output:**
```
Wh haromythe ave ITY ars ld d
```

**Week 6 MLP output (context=8):**
```
What light through yonder wind the king
hath spoke to me and bid me
come to his tent.
```

8 characters of memory versus 1. The difference is not subtle.

---

## How to run it

```bash
python3 week06/mlp_model.py
```

Training takes a few minutes in pure Python — the MLP forward and backward passes are significantly more expensive than the bigram. This is expected. Grab a coffee.

Requires Python 3.10+. No external dependencies.

---

## File structure

```
week06/
├── mlp_model.py    ← Embedding, Linear, tanh, full MLP, manual backprop, training loop
└── README.md       ← you are here
```

---

## Key concepts introduced

| Concept | What it means |
|---------|---------------|
| **Context window** | How many past characters the model sees when making a prediction |
| **Embedding concatenation** | Joining multiple character vectors into a single flat input vector |
| **Hidden layer** | An intermediate representation computed between input and output |
| **Tanh activation** | Non-linearity that squashes values to (-1, 1) — essential for depth |
| **MLP** | Multi-Layer Perceptron — the fundamental feedforward neural network |
| **Backprop through tanh** | `d_pre = d_hidden * (1 - tanh²)` — the tanh gradient |

---

## The limitation that drives everything forward

The MLP treats all 8 context characters **equally**. Position 1 and position 8 contribute identically to the hidden layer. But that's not how language works.

In the phrase `"The cat sat on the mat because it"`, the word `"it"` is closely related to `"cat"` — but they're separated by 7 words. The MLP has no way to express "character 1 matters more than character 5 here". It's forced to weight every position the same.

Attention — introduced next week — fixes this precisely.

---

## What's next

**Week 7 — Attention: The Core Idea**

Single-head self-attention lets every character look at every other character in the context and decide how much to weight it. The model learns — from data alone — which characters to pay attention to. This is the mechanism at the heart of every modern LLM. We implement it from scratch in 40 lines of Python.

→ [Week 7](../week07/README.md)

---

*Built with ❤️ and pure Python by [Kiran](https://linkedin.com/in/kirankumargosu) | [gosulab](https://kirangosu.com)*