"""
gosu-slm | Week 8 — The Transformer Block
==========================================
Single-head attention lets the model look for ONE kind of relationship.
Multi-head attention runs several attention heads in parallel, each
learning to look for different patterns simultaneously.

This week we build the complete Transformer block:
  Multi-Head Attention  ← N heads running in parallel, outputs concatenated
  Layer Normalisation   ← stabilise activations before each sub-layer
  Feedforward Network   ← 2-layer MLP applied independently at each position
  Residual Connections  ← add input to output of each sub-layer

We stack 2 of these blocks to form a small but genuine Transformer.

Run:
  python transformer.py

Author : Kiran Kumar Gosu
Series : Build an SLM from Scratch
Week   : 8 / 10
Repo   : github.com/kirankumargosu/gosu-slm
"""

import urllib.request, os, random, math

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../", "data", "shakespeare.txt"))

# Hyperparameters — note embed_dim must be divisible by num_heads
CONTEXT_LEN = 32
EMBED_DIM   = 64    # total embedding dimension
NUM_HEADS   = 4     # number of attention heads
HEAD_DIM    = EMBED_DIM // NUM_HEADS   # 16 per head
FF_DIM      = 128   # feedforward hidden size
NUM_BLOCKS  = 2     # number of transformer blocks
BATCH_SIZE  = 16
LR          = 0.02
NUM_STEPS   = 2000


# ── shared utils ─────────────────────────────────────────────────────────────

def load_and_encode():
    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        urllib.request.urlretrieve(SHAKESPEARE_URL, DATA_PATH)
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    vocab_size = len(chars)
    ch2idx = {ch: i for i, ch in enumerate(chars)}
    idx2ch = {i: ch for i, ch in enumerate(chars)}
    data = [ch2idx[ch] for ch in text]
    split = int(0.9 * len(data))
    return data[:split], data[split:], vocab_size, idx2ch


def softmax(logits):
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def dot(a, b):
    return sum(ai*bi for ai, bi in zip(a, b))


def mat_vec(M, v):
    return [dot(row, v) for row in M]


def rand_mat(r, c, scale=None):
    s = scale or math.sqrt(2.0/r)
    return [[random.gauss(0, s) for _ in range(c)] for _ in range(r)]


# ── 1. LAYER NORMALISATION ───────────────────────────────────────────────────

class LayerNorm:
    """
    Normalise the values in a vector to have mean ≈ 0 and std ≈ 1,
    then apply learned scale (gamma) and shift (beta).

    Why? Neural networks train much more stably when activations stay
    in a consistent range. Without normalisation, values can explode
    or vanish as they pass through many layers.

    Formula: y = gamma * (x - mean) / sqrt(var + eps) + beta
    """

    def __init__(self, dim, eps=1e-5):
        self.gamma = [1.0] * dim
        self.beta  = [0.0] * dim
        self.eps   = eps

    def forward(self, x: list[float]) -> list[float]:
        n    = len(x)
        mean = sum(x) / n
        var  = sum((xi - mean)**2 for xi in x) / n
        std  = math.sqrt(var + self.eps)
        return [self.gamma[i] * (x[i] - mean) / std + self.beta[i]
                for i in range(n)]


# ── 2. SINGLE ATTENTION HEAD ─────────────────────────────────────────────────

class AttentionHead:
    """One head of scaled dot-product causal self-attention."""

    def __init__(self, embed_dim, head_dim):
        self.head_dim = head_dim
        self.scale    = math.sqrt(head_dim)
        self.Wq = rand_mat(head_dim, embed_dim)
        self.Wk = rand_mat(head_dim, embed_dim)
        self.Wv = rand_mat(head_dim, embed_dim)

    def forward(self, xs: list[list[float]]) -> list[list[float]]:
        T = len(xs)
        Q = [mat_vec(self.Wq, x) for x in xs]
        K = [mat_vec(self.Wk, x) for x in xs]
        V = [mat_vec(self.Wv, x) for x in xs]

        # Scaled dot-product scores with causal mask
        scores = []
        for i in range(T):
            row = []
            for j in range(T):
                s = dot(Q[i], K[j]) / self.scale
                row.append(s if j <= i else -1e9)
            scores.append(softmax(row))

        # Weighted sum of values
        out = []
        for i in range(T):
            v = [0.0] * self.head_dim
            for j in range(T):
                for d in range(self.head_dim):
                    v[d] += scores[i][j] * V[j][d]
            out.append(v)

        return out


# ── 3. MULTI-HEAD ATTENTION ──────────────────────────────────────────────────

class MultiHeadAttention:
    """
    Run NUM_HEADS attention heads in parallel, concatenate outputs,
    project back to embed_dim.

    Each head learns to attend to different kinds of relationships:
    one might focus on grammatical structure, another on word meaning,
    another on rhythm (important for Shakespeare).

    The concatenated output is projected back to embed_dim via Wo,
    allowing the heads to 'communicate' through the output projection.
    """

    def __init__(self, embed_dim, num_heads, head_dim):
        self.heads    = [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        self.num_heads = num_heads
        self.head_dim  = head_dim
        concat_dim    = num_heads * head_dim
        # Output projection: concatenated heads → embed_dim
        self.Wo = rand_mat(embed_dim, concat_dim)

    def forward(self, xs: list[list[float]]) -> list[list[float]]:
        T = len(xs)

        # Run all heads
        head_outputs = [head.forward(xs) for head in self.heads]

        # Concatenate: for each position, join all head outputs
        concatenated = []
        for t in range(T):
            concat_t = []
            for head_out in head_outputs:
                concat_t.extend(head_out[t])
            concatenated.append(concat_t)

        # Project back to embed_dim
        output = [mat_vec(self.Wo, c) for c in concatenated]
        return output


# ── 4. FEEDFORWARD NETWORK ───────────────────────────────────────────────────

class FeedForward:
    """
    A 2-layer MLP applied independently to each position.

    After attention collects information from other positions,
    the feedforward network processes that information within
    each position — like a per-token 'thinking' step.

    Architecture: Linear(embed_dim → ff_dim) → ReLU → Linear(ff_dim → embed_dim)

    We use ReLU here (max(0,x)) instead of tanh for speed.
    ReLU is the standard choice in modern transformers.
    """

    def __init__(self, embed_dim, ff_dim):
        self.W1 = rand_mat(ff_dim, embed_dim)
        self.b1 = [0.0] * ff_dim
        self.W2 = rand_mat(embed_dim, ff_dim)
        self.b2 = [0.0] * embed_dim

    def _relu(self, x):
        return max(0.0, x)

    def forward(self, x: list[float]) -> list[float]:
        # First layer: Linear + ReLU
        h = [self._relu(dot(row, x) + b) for row, b in zip(self.W1, self.b1)]
        # Second layer: Linear
        out = [dot(row, h) + b for row, b in zip(self.W2, self.b2)]
        return out


# ── 5. TRANSFORMER BLOCK ─────────────────────────────────────────────────────

class TransformerBlock:
    """
    One full Transformer block with residual connections.

    Structure:
      x = x + MultiHeadAttention(LayerNorm(x))   ← attention sub-layer
      x = x + FeedForward(LayerNorm(x))           ← feedforward sub-layer

    Residual connections (the '+') are the key insight here.
    They create a 'highway' for gradients to flow directly back through
    many layers during training, solving the vanishing gradient problem.
    Without them, deep networks become nearly untrainable.
    """

    def __init__(self, embed_dim, num_heads, head_dim, ff_dim):
        self.ln1  = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, head_dim)
        self.ln2  = LayerNorm(embed_dim)
        self.ff   = FeedForward(embed_dim, ff_dim)

    def forward(self, xs: list[list[float]]) -> list[list[float]]:
        T = len(xs)
        embed_dim = len(xs[0])

        # Attention sub-layer with residual
        normed   = [self.ln1.forward(xs[t]) for t in range(T)]
        attn_out = self.attn.forward(normed)
        xs = [[xs[t][d] + attn_out[t][d] for d in range(embed_dim)]
              for t in range(T)]

        # Feedforward sub-layer with residual
        normed = [self.ln2.forward(xs[t]) for t in range(T)]
        ff_out = [self.ff.forward(normed[t]) for t in range(T)]
        xs = [[xs[t][d] + ff_out[t][d] for d in range(embed_dim)]
              for t in range(T)]

        return xs


# ── 6. FULL TRANSFORMER MODEL ────────────────────────────────────────────────

class TransformerModel:
    """
    A small but genuine Transformer language model.

    Architecture:
      TokenEmbedding + PositionalEmbedding
      → N × TransformerBlock
      → LayerNorm
      → Linear(embed_dim → vocab_size)
    """

    def __init__(self, vocab_size, context_len=CONTEXT_LEN,
                 embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
                 head_dim=HEAD_DIM, ff_dim=FF_DIM, num_blocks=NUM_BLOCKS):
        self.vocab_size  = vocab_size
        self.context_len = context_len
        self.embed_dim   = embed_dim

        # Token and position embeddings
        self.tok_emb = [[random.gauss(0, 0.02) for _ in range(embed_dim)]
                        for _ in range(vocab_size)]
        self.pos_emb = [[random.gauss(0, 0.01) for _ in range(embed_dim)]
                        for _ in range(context_len)]

        # Transformer blocks
        self.blocks = [TransformerBlock(embed_dim, num_heads, head_dim, ff_dim)
                       for _ in range(num_blocks)]

        # Final layer norm + output head
        self.ln_f  = LayerNorm(embed_dim)
        self.Wout  = rand_mat(vocab_size, embed_dim, scale=0.02)
        self.bout  = [0.0] * vocab_size

    def forward(self, context: list[int]) -> list[float]:
        """Forward pass — returns logits for the next character."""
        # Embed tokens + positions
        xs = []
        for pos, idx in enumerate(context):
            e = [self.tok_emb[idx][d] + self.pos_emb[pos][d]
                 for d in range(self.embed_dim)]
            xs.append(e)

        # Pass through transformer blocks
        for block in self.blocks:
            xs = block.forward(xs)

        # Apply final layer norm to last position
        last = self.ln_f.forward(xs[-1])

        # Project to vocab logits
        logits = [self.bout[j] + dot(self.Wout[j], last)
                  for j in range(self.vocab_size)]
        return logits

    def train_step(self, context: list[int], target: int, lr: float) -> float:
        """Forward pass + loss + simplified parameter nudge."""
        logits = self.forward(context)
        probs  = softmax(logits)
        loss   = -math.log(max(probs[target], 1e-9))

        # Gradient at output layer
        d_logits = probs[:]
        d_logits[target] -= 1.0

        # Update output projection (simplified — full backprop is in week09)
        for j in range(self.vocab_size):
            self.bout[j] -= lr * d_logits[j]

        # Lightly perturb embeddings to encourage exploration
        for idx in context:
            for d in range(self.embed_dim):
                self.tok_emb[idx][d] -= lr * 0.0005 * random.gauss(0, 1)

        return loss

    def generate(self, idx2ch, num_chars=500, temperature=0.8, k=20):
        newline = list(idx2ch.keys())[list(idx2ch.values()).index('\n')]
        context = [newline] * self.context_len
        out = []
        for _ in range(num_chars):
            logits = self.forward(context)
            sorted_i = sorted(range(self.vocab_size),
                              key=lambda i: logits[i], reverse=True)
            top_k = sorted_i[:k]
            restr = [-1e9] * self.vocab_size
            for i in top_k:
                restr[i] = logits[i] / temperature
            m = max(r for r in restr if r > -1e8)
            exps = [math.exp(r - m) if r > -1e8 else 0.0 for r in restr]
            total = sum(exps)
            probs = [e / total for e in exps]
            nxt = random.choices(range(self.vocab_size), weights=probs, k=1)[0]
            out.append(idx2ch[nxt])
            context = context[1:] + [nxt]
        return "".join(out)

    def count_params(self):
        n  = self.vocab_size * self.embed_dim          # token emb
        n += self.context_len * self.embed_dim         # pos emb
        n += self.vocab_size * self.embed_dim          # Wout
        per_block = (
            3 * EMBED_DIM * HEAD_DIM * NUM_HEADS +     # Q,K,V per head × heads
            EMBED_DIM * (NUM_HEADS * HEAD_DIM) +       # Wo
            EMBED_DIM * FF_DIM + FF_DIM * EMBED_DIM    # FFN
        )
        n += NUM_BLOCKS * per_block
        return n


# ── 7. MAIN ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  gosu-slm | Week 8 — Transformer Block")
    print("=" * 60)

    random.seed(42)
    train_data, val_data, vocab_size, idx2ch = load_and_encode()

    model = TransformerModel(vocab_size)
    print(f"\nArchitecture: {NUM_BLOCKS} blocks × ({NUM_HEADS} heads, dim={EMBED_DIM}, ff={FF_DIM})")
    print(f"Context: {CONTEXT_LEN} chars | Approx params: {model.count_params():,}")

    print(f"\nTraining for {NUM_STEPS} steps...")
    print(f"{'─'*60}")

    for step in range(NUM_STEPS):
        max_start = len(train_data) - CONTEXT_LEN - 1
        starts = [random.randint(0, max_start) for _ in range(BATCH_SIZE)]
        total_loss = 0.0
        for s in starts:
            ctx = train_data[s : s + CONTEXT_LEN]
            tgt = train_data[s + CONTEXT_LEN]
            total_loss += model.train_step(ctx, tgt, LR / BATCH_SIZE)

        if step % 400 == 0 or step == NUM_STEPS - 1:
            avg = total_loss / BATCH_SIZE
            bar = "█" * max(0, int((4.5 - avg) * 7))
            print(f"  step {step:>4d} | loss {avg:.4f} | {bar}")

    print(f"\n{'─'*60}")
    print("GENERATED TEXT:")
    print(f"{'─'*60}")
    print(model.generate(idx2ch, num_chars=450))

    print(f"\n{'─'*60}")
    print(f"{NUM_HEADS} attention heads. {NUM_BLOCKS} transformer blocks. Residual connections.")
    print("Next week: assemble the complete SLM with full backprop.")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
