# ⚡ gosu-slm — Build an SLM from Scratch

> A 10-week series building a Small Language Model in **pure Python**.
> No PyTorch. No TensorFlow. No magic.
> Trained on Shakespeare. Built from first principles.

Companion LinkedIn series by [Kiran](https://linkedin.com/in/kirankumargosu)

---

## The idea

Most people learn *about* language models. This series teaches you to *build* one.

Every week: one concept, one Python file, one step closer to a working transformer.
By week 9 we have a model that generates Shakespeare. By week 10 we understand
exactly why it works — and what it would take to scale it to GPT.

---

## Progress

| Week | Topic | Status | Code |
|------|-------|--------|------|
| 1 | What Is a Language Model, Really? | ✅ Live | [weeks/week01/](week01/) |
| 2 | Tokenisation from Scratch | 🔜 Coming | — |
| 3 | The Bigram Model | 🔜 Coming | — |
| 4 | The Training Loop from Scratch | 🔜 Coming | — |
| 5 | Making It Generate — Sampling Strategies | 🔜 Coming | — |
| 6 | Neural Networks by Hand | 🔜 Coming | — |
| 7 | Attention — the Core Idea | 🔜 Coming | — |
| 8 | Building the Transformer Block | 🔜 Coming | — |
| 9 | The Full SLM — Putting It All Together | 🔜 Coming | — |
| 10 | Scaling & Beyond — What's Next | 🔜 Coming | — |

*Updated every Monday when the LinkedIn post goes live.*

---

## Stack

- **Language**: Python 3.10+ (pure — no ML frameworks)
- **Dependencies**: none (stdlib only)
- **Training data**: [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) (~1MB, auto-downloaded)
- **Hardware**: runs on a laptop — no GPU needed

---

## Getting started

```bash
git clone https://github.com/kirankumargosu/gosu-slm
cd gosu-slm

# Run week 1 — downloads Shakespeare automatically on first run
python3 week01/bigram.py
```

Each week's folder is self-contained. You can run any week independently.

---

## Final model (Week 9)

By week 9, `slm/slm.py` will be a complete character-level transformer:

```
Vocabulary     : 65 characters
Context window : 256 characters
Embedding dim  : 384
Attention heads: 6
Blocks         : 3
Parameters     : ~10 million
Training data  : 1.1M characters of Shakespeare
```

Sample output from the trained model:

```
What light through yonder window breaks?
It is the east, and Juliet is the sun.
Arise, fair sun, and kill the envious moon,
Who is already sick and pale with grief
```

---

## Series philosophy

Every line of code is written to be **read**, not just run.

- Teaching comments throughout — no assumed ML knowledge
- No framework magic — every operation is visible
- Real output at every stage — you see the model improve week by week
- Pure Python — if you know Python, you can follow this

---

*⚡ Part of the [AI Unlocked](https://linkedin.com/in/kirankumargosu) series on LinkedIn*