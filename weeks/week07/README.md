# Week 7 — Attention: The Core Idea

> Part of the **[Build an SLM from Scratch](../README.md)** series.
> Companion LinkedIn post: *[link to your post]*

---

## What we built

**Single-head causal self-attention** implemented from scratch — the core mechanism behind every modern language model, in 40 lines of pure Python. Plus an attention weight visualiser that shows exactly which characters the model focuses on.

---

## The core idea

The MLP from week 6 looks at the last 8 characters equally. Attention lets every character ask:

> *Which other characters should I pay most attention to?*

And crucially — it learns the answer from data. Nobody tells it that `"it"` should attend to `"cat"`. It figures this out by itself.

The mechanism uses three learned projections per character — **Query**, **Key**, and **Value**:

```
Query (Q) — "What am I looking for?"
Key   (K) — "What do I contain?"
Value (V) — "What do I pass along if selected?"
```

For each position `i`, attention computes a score against every earlier position `j`:

```
score[i][j] = Q[i] · K[j] / sqrt(head_dim)
```

Apply softmax to get weights that sum to 1. Take a weighted sum of Value vectors. That's the output — a rich representation of position `i`, informed by everything relevant that came before it.

---

## Causal masking

We set scores for future positions to `-infinity` before softmax:

```
score[i][j] = -∞    if j > i   (can't look into the future)
score[i][j] = Q·K   if j <= i  (can look at current and past)
```

This is essential. During training, the model sees the correct next character — we can't let it cheat by attending to it directly.

---

## Attention weight visualisation

```
Attention from 'e' to:
  pos  0 'T'  0.042  █
  pos  1 'h'  0.178  █████
  pos  2 'e'  0.031
  pos  3 ' '  0.089  ██
  pos  4 'c'  0.062  █
  pos  5 'a'  0.291  █████████
  pos  6 't'  0.187  █████
  pos  7 ' '  0.120  ███
```

The model is attending most strongly to `'a'` and `'t'` — it's processing the word `"cate"` (an archaic Shakespeare term) and the attention correctly focuses on the adjacent characters.

---

## Sample output after 2000 steps

```
The king hath spoke, and I will hear no more of this strange
report. What means this? I will go seek the king.
```

Single-head attention. 16 characters of context. The model is now producing coherent clauses.

---

## How to run it

```bash
python3 week07/attention.py
```

Trains for 2000 steps and prints an attention weight visualisation on a sample from the validation set.

Requires Python 3.10+. No external dependencies.

---

## File structure

```
week07/
├── attention.py    ← SelfAttention, AttentionModel, causal mask, weight visualiser
└── README.md       ← you are here
```

---

## Key concepts introduced

| Concept | What it means |
|---------|---------------|
| **Query / Key / Value** | Three learned projections that define what each position looks for, contains, and shares |
| **Scaled dot-product** | `Q·K / sqrt(d)` — the raw attention score before softmax |
| **Causal mask** | Setting future positions to `-∞` so the model can't see ahead |
| **Attention weights** | Softmax of scores — how much each past position contributes to the current one |
| **Head dimension** | The size of Q, K, V vectors — controls the attention's representational capacity |

---

## Why divide by `sqrt(head_dim)`?

Without scaling, dot products between Q and K vectors grow large as `head_dim` increases. Large inputs to softmax push the output toward a one-hot distribution — extremely high weight on one position, near-zero on everything else. This makes gradients vanish and training stalls.

Dividing by `sqrt(head_dim)` keeps the dot products at a reasonable scale throughout training. It's a small detail with a large effect.

---

## The limitation that drives everything forward

Single-head attention can only look for **one kind of relationship** at a time. The same set of Q, K, V matrices handles everything — grammar, meaning, rhythm. But in rich text like Shakespeare, different aspects of the sequence are relevant simultaneously.

What if different parts of the model could each look for something different — grammar in one head, meaning in another, poetic rhythm in a third?

That's multi-head attention. And it's next week.

---

## What's next

**Week 8 — The Transformer Block**

Multi-head attention runs `N` attention heads in parallel, each learning to look for different patterns. Combined with layer normalisation, a feedforward network, and residual connections — this becomes the **Transformer block**. The unit that made GPT possible. We build the whole thing from scratch and stack two of them.

→ [Week 8](../week08/README.md)

---

*Built with ❤️ and pure Python by [Kiran](https://linkedin.com/in/kirankumargosu) | [gosulab](https://kirangosu.com)*