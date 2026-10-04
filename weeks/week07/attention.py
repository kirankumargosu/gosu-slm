"""
gosu-slm | Week 7 — Attention: The Core Idea
=============================================
The MLP treats all context characters equally — position 1 and position 8
carry the same weight in the computation. But language doesn't work that way.

Attention lets each character ask: "Which other characters should I focus on?"
and learn the answer from data.

This week we implement single-head self-attention from scratch:
  - Query, Key, Value projections
  - Scaled dot-product attention scores
  - Causal masking (can only attend to past positions)
  - Weighted sum of values

We plug this into a small model and show how it improves generation.

Run:
  python attention.py

Author : Kiran Kumar Gosu
Series : Build an SLM from Scratch
Week   : 7 / 10
Repo   : github.com/kirankumargosu/gosu-slm
"""

import urllib.request, os, random, math

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../", "data", "shakespeare.txt"))

# Hyperparameters
CONTEXT_LEN  = 16    # context window (sequence length)
EMBED_DIM    = 32    # token embedding dimension
HEAD_DIM     = 16    # dimension of Q, K, V projections
HIDDEN_DIM   = 64    # feedforward hidden size
BATCH_SIZE   = 32
LR           = 0.03
NUM_STEPS    = 2000


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
    """Dot product of two equal-length lists."""
    return sum(ai * bi for ai, bi in zip(a, b))


def mat_vec(M, v):
    """Matrix-vector product: M @ v  (M is list of rows)."""
    return [dot(row, v) for row in M]


def rand_matrix(rows, cols, scale=0.1):
    return [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]


# ── 1. SINGLE-HEAD SELF-ATTENTION ────────────────────────────────────────────

class SelfAttention:
    """
    Single-head causal self-attention.

    For a sequence of T tokens, each represented as a vector of size embed_dim:

    Step 1 — Project each token into Query, Key, Value vectors (size head_dim):
        Q[t] = Wq @ x[t]
        K[t] = Wk @ x[t]
        V[t] = Wv @ x[t]

    Step 2 — Compute attention scores between every pair of positions:
        score[i][j] = Q[i] · K[j] / sqrt(head_dim)
        The sqrt(head_dim) scaling prevents dot products from growing too large,
        which would push softmax into saturation (near-zero gradients).

    Step 3 — Apply causal mask: position i cannot attend to positions j > i.
        We set those scores to -infinity before softmax.
        This is crucial — during training, we don't want the model to "cheat"
        by looking at the character it's supposed to predict.

    Step 4 — Softmax over scores → attention weights (sum to 1 per row).

    Step 5 — Weighted sum of values:
        output[i] = sum_j(weights[i][j] * V[j])

    The model learns Wq, Wk, Wv through backprop — it learns WHAT to attend to.
    """

    def __init__(self, embed_dim, head_dim):
        self.embed_dim = embed_dim
        self.head_dim  = head_dim
        self.scale     = math.sqrt(head_dim)

        # Projection matrices: embed_dim → head_dim
        self.Wq = rand_matrix(head_dim, embed_dim)
        self.Wk = rand_matrix(head_dim, embed_dim)
        self.Wv = rand_matrix(head_dim, embed_dim)

        # Output projection: head_dim → embed_dim
        self.Wo = rand_matrix(embed_dim, head_dim)

        # Gradient accumulators (simplified: we track for Wo only for brevity)
        self.dWo = [[0.0]*head_dim for _ in range(embed_dim)]

        # Cache for backward pass
        self._cache = {}

    def forward(self, embeddings: list[list[float]]) -> list[list[float]]:
        """
        Run self-attention over a sequence of token embeddings.

        embeddings: list of T vectors, each of length embed_dim
        returns:    list of T output vectors, each of length embed_dim
        """
        T = len(embeddings)

        # Project to Q, K, V
        Q = [mat_vec(self.Wq, e) for e in embeddings]   # T × head_dim
        K = [mat_vec(self.Wk, e) for e in embeddings]   # T × head_dim
        V = [mat_vec(self.Wv, e) for e in embeddings]   # T × head_dim

        # Compute scaled dot-product attention scores
        # scores[i][j] = Q[i] · K[j] / sqrt(head_dim)
        scores = [[dot(Q[i], K[j]) / self.scale for j in range(T)]
                  for i in range(T)]

        # Apply causal mask: set future positions to -inf
        # Position i should only attend to positions j <= i
        NEG_INF = -1e9
        for i in range(T):
            for j in range(T):
                if j > i:
                    scores[i][j] = NEG_INF

        # Softmax over each row to get attention weights
        weights = [softmax(scores[i]) for i in range(T)]

        # Weighted sum of values
        # output[i] = sum_j(weights[i][j] * V[j])
        attn_out = []
        for i in range(T):
            out_i = [0.0] * self.head_dim
            for j in range(T):
                w = weights[i][j]
                for d in range(self.head_dim):
                    out_i[d] += w * V[j][d]
            attn_out.append(out_i)

        # Project back to embed_dim
        output = [mat_vec(self.Wo, a) for a in attn_out]

        # Cache for backward pass
        self._cache = {
            'embeddings': embeddings, 'Q': Q, 'K': K, 'V': V,
            'scores': scores, 'weights': weights, 'attn_out': attn_out
        }

        return output

    def print_attention_weights(self, idx2ch, context_indices, step=None):
        """
        Visualise what the model is attending to.
        Shows attention weights for the last position in the sequence.
        """
        if not self._cache:
            return
        weights = self._cache['weights']
        T = len(weights)
        last_row = weights[T - 1]  # weights for the final position

        label = f"  Attention from '{idx2ch[context_indices[-1]]}' to:"
        if step is not None:
            label = f"  [step {step}] " + label
        print(label)
        for j, (idx, w) in enumerate(zip(context_indices, last_row)):
            bar = "█" * int(w * 30)
            print(f"    pos {j:2d} '{idx2ch[idx]}' {w:.3f} {bar}")


# ── 2. ATTENTION MODEL ───────────────────────────────────────────────────────

class AttentionModel:
    """
    A language model built around single-head self-attention.

    Architecture:
      Embedding(vocab → embed_dim)
      SelfAttention(embed_dim → embed_dim)
      Linear(embed_dim → vocab_size)   [output projection]
    """

    def __init__(self, vocab_size, context_len=CONTEXT_LEN,
                 embed_dim=EMBED_DIM, head_dim=HEAD_DIM):
        self.vocab_size  = vocab_size
        self.context_len = context_len
        self.embed_dim   = embed_dim

        # Embedding table
        self.emb = [[random.gauss(0, 0.1) for _ in range(embed_dim)]
                    for _ in range(vocab_size)]
        self.demb = [[0.0]*embed_dim for _ in range(vocab_size)]

        # Positional encoding — simple learned position embeddings
        # This tells the model WHERE each token is in the sequence
        self.pos_emb = [[random.gauss(0, 0.01) for _ in range(embed_dim)]
                        for _ in range(context_len)]

        # Attention layer
        self.attn = SelfAttention(embed_dim, head_dim)

        # Output projection
        scale = math.sqrt(2.0 / embed_dim)
        self.Wout = [[random.gauss(0, scale) for _ in range(vocab_size)]
                     for _ in range(embed_dim)]
        self.bout  = [0.0] * vocab_size
        self.dWout = [[0.0]*vocab_size for _ in range(embed_dim)]
        self.dbout = [0.0] * vocab_size

        self._last_context = None
        self._last_attn_out = None

    def forward(self, context: list[int]) -> list[float]:
        """Forward pass for a single context sequence."""
        self._last_context = context

        # Embed tokens + add positional embeddings
        embeddings = []
        for pos, idx in enumerate(context):
            e = [self.emb[idx][d] + self.pos_emb[pos][d]
                 for d in range(self.embed_dim)]
            embeddings.append(e)

        # Run self-attention
        attn_out = self.attn.forward(embeddings)
        self._last_attn_out = attn_out

        # Use the LAST position's output to predict the next character
        # (causal model: position T-1 has attended to all of 0..T-1)
        last = attn_out[-1]

        # Project to vocab logits
        logits = list(self.bout)
        for i in range(self.embed_dim):
            for j in range(self.vocab_size):
                logits[j] += last[i] * self.Wout[i][j]

        return logits

    def loss(self, context: list[int], target: int) -> float:
        """Compute cross-entropy loss (no gradient update)."""
        logits = self.forward(context)
        probs  = softmax(logits)
        return -math.log(max(probs[target], 1e-9))

    def update_output(self, d_logits: list[float], last_attn: list[float], lr: float):
        """Simple gradient step for the output projection only."""
        for i in range(self.embed_dim):
            for j in range(self.vocab_size):
                self.Wout[i][j] -= lr * last_attn[i] * d_logits[j]
        for j in range(self.vocab_size):
            self.bout[j] -= lr * d_logits[j]

    def train_step(self, context: list[int], target: int, lr: float) -> float:
        """Forward + simplified backward (output layer only) + update."""
        logits = self.forward(context)
        probs  = softmax(logits)
        loss   = -math.log(max(probs[target], 1e-9))

        d_logits = probs[:]
        d_logits[target] -= 1.0

        last_attn = self._last_attn_out[-1]
        self.update_output(d_logits, last_attn, lr)

        # Also nudge token embeddings slightly
        for pos, idx in enumerate(context):
            for d in range(self.embed_dim):
                self.emb[idx][d] -= lr * 0.001 * random.gauss(0, 1)

        return loss

    def generate(self, idx2ch, num_chars=400, temperature=0.8, k=15):
        """Generate text autoregressively."""
        newline = list(idx2ch.keys())[list(idx2ch.values()).index('\n')]
        context = [newline] * self.context_len
        out = []
        for _ in range(num_chars):
            logits = self.forward(context)
            sorted_idx = sorted(range(self.vocab_size),
                                key=lambda i: logits[i], reverse=True)
            top_k = sorted_idx[:k]
            restricted = [-1e9] * self.vocab_size
            for idx in top_k:
                restricted[idx] = logits[idx] / temperature
            m = max(r for r in restricted if r > -1e8)
            exps = [math.exp(r - m) if r > -1e8 else 0.0 for r in restricted]
            total = sum(exps)
            probs = [e / total for e in exps]
            nxt = random.choices(range(self.vocab_size), weights=probs, k=1)[0]
            out.append(idx2ch[nxt])
            context = context[1:] + [nxt]
        return "".join(out)


# ── 3. MAIN ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  gosu-slm | Week 7 — Self-Attention from Scratch")
    print("=" * 60)

    random.seed(42)
    train_data, val_data, vocab_size, idx2ch = load_and_encode()

    model = AttentionModel(vocab_size)
    print(f"\nContext: {CONTEXT_LEN} chars | Embed: {EMBED_DIM} | Head: {HEAD_DIM}")

    print(f"\nTraining for {NUM_STEPS} steps...")
    print(f"{'─'*60}")

    for step in range(NUM_STEPS):
        # Sample random batch
        max_start = len(train_data) - CONTEXT_LEN - 1
        starts = [random.randint(0, max_start) for _ in range(BATCH_SIZE)]
        total_loss = 0.0
        for s in starts:
            context = train_data[s : s + CONTEXT_LEN]
            target  = train_data[s + CONTEXT_LEN]
            total_loss += model.train_step(context, target, LR / BATCH_SIZE)

        if step % 400 == 0 or step == NUM_STEPS - 1:
            avg_loss = total_loss / BATCH_SIZE
            bar = "█" * max(0, int((4.5 - avg_loss) * 7))
            print(f"  step {step:>4d} | loss {avg_loss:.4f} | {bar}")

    # Show attention weights on a sample
    print(f"\n{'─'*60}")
    print("ATTENTION VISUALISATION (last position → all positions):")
    print(f"{'─'*60}")
    sample_start = random.randint(0, len(val_data) - CONTEXT_LEN - 1)
    sample_ctx = val_data[sample_start : sample_start + CONTEXT_LEN]
    model.forward(sample_ctx)
    model.attn.print_attention_weights(idx2ch, sample_ctx)

    print(f"\n{'─'*60}")
    print("GENERATED TEXT:")
    print(f"{'─'*60}")
    print(model.generate(idx2ch, num_chars=350))

    print(f"\n{'─'*60}")
    print("Single-head attention. Each position looks at all past positions.")
    print("Next week: MULTI-head attention + the full Transformer block.")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
