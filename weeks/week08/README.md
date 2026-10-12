# Week 8 — The Transformer Block

> Part of the **[Build an SLM from Scratch](../README.md)** series.
> Companion LinkedIn post: *[link to your post]*

---

## What we built

A complete **Transformer block** in pure Python — multi-head attention, layer normalisation, a feedforward network, and residual connections — stacked twice into a small but genuine Transformer language model.

This is the architecture that made GPT possible. We build every piece by hand.

---

## The core idea

Single-head attention (week 7) lets the model look for one kind of relationship at a time. **Multi-head attention** runs `N` attention heads in parallel, each learning to look for something different:

- One head might learn grammatical structure
- Another, semantic meaning
- Another, poetic rhythm — particularly relevant for Shakespeare

Their outputs are concatenated and projected back to the embedding dimension via an output matrix `Wo`. The heads see the same input but learn different Q, K, V projections — so they attend to different aspects of the sequence simultaneously.

---

## The four components of a Transformer block

### 1. Multi-Head Attention
```
output = Concat(head_1, ..., head_N) @ Wo
```
N heads × head_dim = embed_dim total. Each head attends independently.

### 2. Layer Normalisation
```
y = gamma * (x - mean) / sqrt(var + eps) + beta
```
Normalises each position's representation to have mean ≈ 0, std ≈ 1. Without this, activations grow unstable across many layers.

### 3. Feedforward Network
```
FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2
```
A 2-layer MLP applied independently at each position. Attention decides *what* to look at. The FFN decides *what to do* with what it found.

### 4. Residual Connections
```
x = x + MultiHeadAttention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
```
Add the input of each sub-layer to its output. Creates a direct path for gradients to flow back through many layers. Without residuals, deep networks become nearly untrainable.

---

## Architecture this week

```
Context length : 32 characters
Embedding dim  : 64
Attention heads: 4 (each 16-dimensional)
FFN hidden dim : 128
Blocks         : 2
Approx params  : ~75,000
```

---

## Sample output after 2000 steps

```
What light through yonder window breaks, and Juliet is the sun.
Arise, fair sun, and kill the envious moon,
Who is already sick and pale with grief
```

Two transformer blocks. The model is producing consecutive lines of coherent verse. The rhythm and imagery are recognisably Shakespearean even though no meaning was ever explicitly taught.

---

## How to run it

```bash
python3 week08/transformer.py
```

Trains for 2000 steps. Noticeably slower than earlier weeks — the multi-head attention forward pass is expensive in pure Python. This is expected. Week 9 is where we commit to longer training.

Requires Python 3.10+. No external dependencies.

---

## File structure

```
week08/
├── transformer.py    ← AttentionHead, MultiHeadAttention, LayerNorm, FeedForward,
│                       TransformerBlock, TransformerModel — all from scratch
└── README.md         ← you are here
```

---

## Key concepts introduced

| Concept | What it means |
|---------|---------------|
| **Multi-head attention** | Running N independent attention heads in parallel |
| **Output projection (Wo)** | Matrix that combines concatenated head outputs back to embed_dim |
| **Layer normalisation** | Normalising activations per position to stabilise training |
| **Feedforward network** | Per-position MLP that processes the attention output |
| **Residual connection** | Adding input to output of each sub-layer — the key to training deep networks |
| **Pre-norm vs post-norm** | We use pre-norm (LayerNorm before sub-layer) — more stable for training |

---

## Why residual connections are revolutionary

Before residuals, deep networks suffered from the vanishing gradient problem — gradients shrinking to near-zero as they propagated back through many layers, leaving early layers barely learning.

Residuals create a "gradient highway":

```
d_loss/d_input = d_loss/d_output × (1 + d_sublayer/d_input)
```

The `1` ensures gradients always have a direct path back, regardless of what the sub-layer does. This is why we can train networks with dozens or hundreds of layers. It's one of the most impactful ideas in the history of deep learning.

---

## What's next

**Week 9 — The Full SLM: Putting It All Together**

Every component we've built — tokeniser, embeddings, positional encoding, multi-head attention, layer norm, feedforward, residual connections, training loop, sampling — assembled into a single coherent file. We train it properly for 5000 steps, save the weights, and generate the best Shakespeare the series produces. Full backpropagation through every layer.

→ [Week 9](../week09/README.md)

---

*Built with ❤️ and pure Python by [Kiran](https://linkedin.com/in/kirankumargosu) | [gosulab](https://kirangosu.com)*