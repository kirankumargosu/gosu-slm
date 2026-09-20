# Week 5 — Making It Generate: Sampling Strategies

> Part of the **[Build an SLM from Scratch](../README.md)** series.
> Companion LinkedIn post: *[link to your post]*

---

## What we built

Three **sampling strategies** — greedy, temperature, and top-k — applied to the same trained model to show how generation behaviour changes without touching a single weight.

The model doesn't change. The output does. Dramatically.

---

## The core idea

After training, the model produces a **probability distribution** over all 65 characters for every next position. The question is: how do you pick one?

This decision — called the **decoding strategy** — shapes the output just as much as training does.

---

## Strategy 1 — Greedy sampling

```python
next_char = argmax(logits)   # always pick the highest probability character
```

Deterministic. Safe. Boring. The model gets stuck in loops because it always makes the most conservative choice — once it finds a high-probability sequence it repeats it forever.

**Output:**
```
the the the and the the the and the the the and the
```

---

## Strategy 2 — Temperature sampling

```python
scaled_logits = logits / temperature
probs = softmax(scaled_logits)
next_char = random.choices(range(vocab_size), weights=probs)
```

`temperature` controls how peaked or flat the distribution is:

| Temperature | Effect | Output character |
|-------------|--------|-----------------|
| 0.2 | Very peaked — conservative | Safe, repetitive |
| 1.0 | Raw model distribution | Balanced |
| 1.8 | Very flat — uniform | Creative, incoherent |

**Output at temperature 0.8:**
```
What the king hath to me the lord
the court and I will be
```

---

## Strategy 3 — Top-k sampling

```python
top_k_indices = sorted(range(vocab_size), key=lambda i: logits[i])[-k:]
# zero out everything except top k, then sample
```

Restricts sampling to the `k` most likely characters before applying temperature. Prevents the model from ever picking a very unlikely character (e.g., `'z'` after `'qu'`).

The sweet spot for our Shakespeare model: `k=20, temperature=0.8`.

**Output at top-k=20, temperature=0.8:**
```
What light through yonder wind the king
hath spoke to me and I will
```

---

## Side-by-side comparison

Same model. Same weights. Same seed. Different strategy.

```
GREEDY         : the the the and the the the and the the
TEMP = 0.5     : What the king and the lord to me
TEMP = 1.0     : What hath the king to sayd and speake
TEMP = 1.5     : Wher hoath quing sproke yon wHat
TOP-K = 5      : What the king hath to the court
TOP-K = 20     : What light through yonder wind the king
```

---

## How to run it

```bash
python3 week05/generator.py
```

Trains the model for 1500 steps then runs all six generation experiments back-to-back with the same random seed, so outputs are directly comparable.

Requires Python 3.10+. No external dependencies.

---

## File structure

```
week05/
├── generator.py    ← greedy_sample(), temperature_sample(), topk_sample(), side-by-side demo
└── README.md       ← you are here
```

---

## Key concepts introduced

| Concept | What it means |
|---------|---------------|
| **Greedy decoding** | Always pick the highest-probability token — deterministic but repetitive |
| **Temperature** | Scalar applied to logits before softmax — controls distribution sharpness |
| **Top-k sampling** | Restrict candidates to the k most likely tokens before sampling |
| **Logits** | Raw model scores before softmax — the input to every sampling strategy |
| **Decoding strategy** | The algorithm used to pick tokens at generation time |

---

## The insight

The model's *behaviour* is not fixed by training alone. A well-trained model with bad sampling produces bad text. A well-trained model with thoughtful sampling produces surprisingly good text.

Temperature and top-k are not tricks or hacks — they're principled ways of controlling the trade-off between coherence (low temperature, small k) and creativity (high temperature, large k).

Every major LLM — GPT-4, Claude, Gemini — uses variants of these strategies. The principles are identical to what we built here.

---

## What's next

**Week 6 — Neural Networks by Hand: The MLP**

The bigram model looks one character back. This week we extend the context window to **8 characters** using a Multi-Layer Perceptron. The quality jump is significant — the model can now see phrases, not just pairs. And we implement every operation — the embedding concatenation, the hidden layer, the tanh activation, the full backward pass — entirely by hand.

→ [Week 6](../week06/README.md)

---

*Built with ❤️ and pure Python by [Kiran](https://linkedin.com/in/kirankumargosu) | [gosulab](https://kirangosu.com)*