# Week 4 — The Training Loop from Scratch

> Part of the **[Build an SLM from Scratch](../README.md)** series.
> Companion LinkedIn post: *[link to your post]*

---

## What we built

A **complete training infrastructure** in pure Python — a `Linear` layer, an `Embedding` layer, manual backpropagation through both, gradient accumulation across batches, and a terminal loss curve. Plus a small 2-layer network that puts it all together.

This is the engine that powers every remaining week. The architecture will grow. The training loop will not change.

---

## The core idea

Last week we computed gradients for the bigram model's single weight matrix using a closed-form formula. That worked because the model was so simple.

Real networks have many layers. Gradients need to flow *backward* through all of them — each layer computing how much its inputs contributed to the final error, and passing that signal further back.

This is **backpropagation** — and we implement it entirely by hand this week.

The chain rule says:

```
d_loss/d_W1 = d_loss/d_output × d_output/d_hidden × d_hidden/d_W1
```

Each layer computes its piece of this chain during the backward pass and passes the gradient signal to the layer before it.

---

## The Linear layer

```python
class Linear:
    def forward(self, x):   # y = xW + b
    def backward(self, d_out):  # compute dW, db, dx
    def update(self, lr):   # W -= lr * dW
```

The `backward()` method computes three things:
- `dW` — how to adjust the weights
- `db` — how to adjust the bias
- `dx` — gradient to pass to the previous layer

`dx` is what allows gradients to flow back through multiple layers. Without it, only the last layer would ever learn.

---

## Loss curve — watching the model learn

```
step    0 | loss 4.1823 |
step  200 | loss 3.8441 | ████
step  400 | loss 3.5209 | ████████
step  600 | loss 3.2874 | ████████████
step  800 | loss 3.1102 | ████████████████
step 1000 | loss 2.9847 | ████████████████████
step 2000 | loss 2.8103 | ████████████████████████
```

Loss falling in real time. The model is genuinely getting better — not because we told it anything, but because the gradient is nudging thousands of numbers in the right direction, step by step.

---

## Sample output after 2000 steps

```
Thir anto the wand shat I the wis
he ther and the hat hin me
KING:
What hat the kin
```

Words are forming. "the", "and", "what", "king" — all learned purely from seeing which integers tend to follow which. Still one character of context. Still limited. But unmistakably more coherent than week 3.

---

## How to run it

```bash
python3 week04/trainer.py
```

Trains for 2000 steps, prints loss every 200 steps with an ASCII bar chart, then generates 400 characters. Expect it to take 30–60 seconds.

Requires Python 3.10+. No external dependencies.

---

## File structure

```
week04/
├── trainer.py    ← Linear layer, Embedding layer, backprop, training loop, loss curve
└── README.md     ← you are here
```

---

## Key concepts introduced

| Concept | What it means |
|---------|---------------|
| **Backpropagation** | Computing gradients layer by layer using the chain rule |
| **Linear layer** | A learnable transformation: `y = xW + b` |
| **Embedding** | A lookup table mapping integer token IDs to dense vectors |
| **Gradient accumulation** | Summing gradients across a batch before updating weights |
| **He initialisation** | Scaling initial weights by `sqrt(2/fan_in)` for stable training |
| **Learning rate** | How large each weight update step is |

---

## Why batching matters

We don't update the weights after every single example — we accumulate gradients across a full batch first. This gives us a better estimate of the true gradient direction (averaging out noise from individual examples) and allows us to take a more confident step.

```
batch_size = 64 examples
↓
accumulate 64 gradients
↓
average them → one weight update
```

A single example's gradient is noisy. The average of 64 is much more reliable.

---

## What's next

**Week 5 — Making It Generate: Sampling Strategies**

The training loop is solid. But how we *sample* from the model at generation time matters just as much as how we train it. Three strategies — greedy, temperature, and top-k — produce completely different text from the exact same trained weights. We compare all three side by side.

→ [Week 5](../week05/README.md)

---

*Built with ❤️ and pure Python by [Kiran](https://linkedin.com/in/kirankumargosu) | [gosulab](https://kirangosu.com)*