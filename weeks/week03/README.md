# Week 3 — The Bigram Model (Trainable)

> Part of the **[Build an SLM from Scratch](../README.md)** series.
> Companion LinkedIn post: *[link to your post]*

---

## What we built

A **trainable bigram language model** — our first model with actual learnable parameters. Instead of a fixed probability lookup table, we have a weight matrix that improves through training.

It still only looks one character back. But now the model learns from data rather than counting from it. That distinction changes everything.

---

## The core idea

In week 1, we built a probability table by counting character pairs and normalising. It worked — but the table was fixed. You couldn't make it better without rerunning the whole count.

This week we replace the table with a **weight matrix `W`** of shape `(vocab_size × vocab_size)`. Each row `W[i]` contains the raw scores — called **logits** — for "what comes after character `i`".

```
W[i]  →  logits  →  softmax  →  probabilities  →  sample next character
```

The logits start as small random numbers. Training adjusts them so the model assigns high probability to the characters that actually follow in Shakespeare, and low probability to the ones that don't.

---

## The loss function

We measure how wrong the model is using **cross-entropy loss**:

```
loss = -log(probability assigned to the correct character)
```

If the model assigns probability `1.0` to the correct character → `loss = 0.0` (perfect).
If the model assigns probability `0.01` to the correct character → `loss = 4.6` (bad).

The random baseline loss for a 65-character vocabulary is `log(65) ≈ 4.17`. If our model can't beat that, it hasn't learned anything.

---

## Sample output before and after training

**Before training** (random weights):
```
wJCkTgm-EHBqrxPWzsYoVe$f'IZynULRi!MNadpF
```

**After 1000 steps**:
```
Wh haromythe ave ITY ars ld d wend heas
tofof hed I d s tit he
```

Still gibberish — but you can see words beginning to form. The model has learned that `'h'` tends to follow `'t'`, that spaces and newlines appear with the right frequency, that `'I'` is a common standalone character.

---

## How to run it

```bash
python3 week03/bigram_model.py
```

It will train for 1000 steps, print loss at each checkpoint, then generate 300 characters. Watch the loss fall from ~4.17 toward ~2.5.

Requires Python 3.10+. No external dependencies.

---

## File structure

```
week03/
├── bigram_model.py    ← trainable weight matrix, softmax, cross-entropy, training loop
└── README.md          ← you are here
```

---

## Key concepts introduced

| Concept | What it means |
|---------|---------------|
| **Logits** | Raw unnormalised scores output by the model before softmax |
| **Softmax** | Converts logits into a valid probability distribution (sums to 1) |
| **Cross-entropy loss** | Measures how wrong the model's probability for the correct class is |
| **Weight matrix** | The model's learnable parameters — adjusted during training |
| **Gradient** | The direction and magnitude to adjust each weight to reduce loss |
| **Random baseline** | `log(vocab_size)` — the loss of a completely random model |

---

## The gradient formula

The gradient of softmax cross-entropy with respect to the logits has a beautiful closed form:

```
d_loss / d_logit[j] = prob[j] - 1    if j == target
d_loss / d_logit[j] = prob[j]        otherwise
```

In plain English: the gradient is just `probabilities - one_hot(target)`. For the correct class, we subtract 1 from its probability. For all others, we leave the probability as-is. This is one of the most elegant formulas in all of deep learning.

---

## The limitation that drives everything forward

We made the model trainable — but it still only looks **one character back**.

The weight matrix `W` has no way to represent longer context. Row `W['h']` says "given that the last character was `h`, here are my predictions" — but it has no idea whether that `h` is part of `"the"`, `"she"`, `"his"`, or `"Hamlet"`. Context matters enormously. One character of context is not enough.

---

## What's next

**Week 4 — The Training Loop from Scratch**

We rebuild the training infrastructure properly — a `Linear` layer with full manual backpropagation, gradient accumulation across batches, and a live loss curve in the terminal. The loop we build this week will carry through the rest of the series unchanged.

→ [Week 4](../week04/README.md)

---

*Built with ❤️ and pure Python by [Kiran](https://linkedin.com/in/kirankumargosu) | [gosulab](https://kirangosu.com)*