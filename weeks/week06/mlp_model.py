"""
gosu-slm | Week 6 — Neural Networks by Hand: The MLP
=====================================================
The bigram model only looks at the last 1 character.
This week we extend the context window to 8 characters
using a Multi-Layer Perceptron (MLP).

Architecture:
  Embedding(vocab_size → embed_dim)        ← one per context character
  Concatenate 8 embeddings → flat vector   ← the full context representation
  Linear(context_len * embed_dim → hidden) ← compress to hidden state
  Tanh activation                          ← introduce non-linearity
  Linear(hidden → vocab_size)              ← project to logits

The tanh activation is crucial — without it, stacking linear layers
is mathematically equivalent to a single linear layer (useless).
Non-linearity is what gives neural networks their expressive power.

Run:
  python mlp_model.py

Author : Kiran Kumar Gosu
Series : Build an SLM from Scratch
Week   : 6 / 10
Repo   : github.com/kirankumargosu/gosu-slm
"""

import urllib.request, os, random, math

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../", "data", "shakespeare.txt"))

# Hyperparameters
CONTEXT_LEN = 8      # how many characters the model looks back
EMBED_DIM   = 24     # size of each character's embedding vector
HIDDEN_DIM  = 128    # size of the hidden layer
BATCH_SIZE  = 64
LEARNING_RATE = 0.05
NUM_STEPS   = 3000


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


def cross_entropy(probs, target):
    return -math.log(max(probs[target], 1e-9))


# ── 1. ACTIVATION: TANH ──────────────────────────────────────────────────────

def tanh(x: float) -> float:
    """
    Hyperbolic tangent activation function.
    Maps any real number to the range (-1, 1).

    This 'squashes' the hidden layer values, preventing them from
    growing unboundedly and introducing the non-linearity that makes
    multi-layer networks powerful.

    Derivative: tanh'(x) = 1 - tanh(x)²
    We'll use this in the backward pass.
    """
    return math.tanh(x)


def tanh_backward(tanh_output: float) -> float:
    """Derivative of tanh given its output value (not input)."""
    return 1.0 - tanh_output ** 2


# ── 2. MLP MODEL ─────────────────────────────────────────────────────────────

class MLPModel:
    """
    Character-level language model with an 8-character context window.

    The key difference from the bigram: we concatenate the embeddings
    of the last CONTEXT_LEN characters into a single flat vector,
    then pass that through a hidden layer before projecting to logits.
    """

    def __init__(self, vocab_size, context_len=CONTEXT_LEN,
                 embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM):
        self.vocab_size  = vocab_size
        self.context_len = context_len
        self.embed_dim   = embed_dim
        self.hidden_dim  = hidden_dim

        # Input to hidden layer: (context_len * embed_dim) → hidden_dim
        flat_dim = context_len * embed_dim
        scale1 = math.sqrt(2.0 / flat_dim)
        self.W1 = [[random.gauss(0, scale1) for _ in range(hidden_dim)]
                   for _ in range(flat_dim)]
        self.b1 = [0.0] * hidden_dim

        # Hidden to output: hidden_dim → vocab_size
        scale2 = math.sqrt(2.0 / hidden_dim)
        self.W2 = [[random.gauss(0, scale2) for _ in range(vocab_size)]
                   for _ in range(hidden_dim)]
        self.b2 = [0.0] * vocab_size

        # Embedding table: vocab_size × embed_dim
        self.emb = [[random.gauss(0, 0.1) for _ in range(embed_dim)]
                    for _ in range(vocab_size)]

        # Gradient accumulators
        self.dW1  = [[0.0]*hidden_dim  for _ in range(flat_dim)]
        self.db1  = [0.0]*hidden_dim
        self.dW2  = [[0.0]*vocab_size  for _ in range(hidden_dim)]
        self.db2  = [0.0]*vocab_size
        self.demb = [[0.0]*embed_dim   for _ in range(vocab_size)]

        # Forward pass cache
        self._context  = None
        self._flat     = None
        self._hidden   = None
        self._hidden_pre = None  # pre-tanh values (needed for backprop)

    def _get_context_embedding(self, context: list[int]) -> list[float]:
        """
        Look up the embedding for each character in the context window
        and concatenate them into a single flat vector.

        context: list of CONTEXT_LEN integer token indices
        returns: flat vector of length CONTEXT_LEN * embed_dim
        """
        flat = []
        for idx in context:
            flat.extend(self.emb[idx])
        return flat

    def _linear(self, x, W, b):
        """Matrix multiply: y = xW + b"""
        out_dim = len(W[0])
        out = list(b)
        for i, xi in enumerate(x):
            for j in range(out_dim):
                out[j] += xi * W[i][j]
        return out

    def forward(self, context: list[int]) -> list[float]:
        """
        Forward pass through the full MLP.

        1. Embed each context character
        2. Concatenate embeddings into flat vector
        3. Linear + tanh → hidden layer
        4. Linear → logits
        """
        self._context = context

        # Step 1+2: embed and concatenate
        flat = self._get_context_embedding(context)
        self._flat = flat

        # Step 3: first linear layer + tanh
        pre_tanh = self._linear(flat, self.W1, self.b1)
        self._hidden_pre = pre_tanh
        hidden = [tanh(v) for v in pre_tanh]
        self._hidden = hidden

        # Step 4: output linear layer → logits
        logits = self._linear(hidden, self.W2, self.b2)
        return logits

    def backward(self, logits: list[float], target: int) -> float:
        """
        Backward pass: compute gradients via chain rule.

        Returns the loss for this example.
        """
        probs = softmax(logits)
        loss  = cross_entropy(probs, target)

        # Gradient of loss w.r.t. logits: probs - one_hot(target)
        d_logits = probs[:]
        d_logits[target] -= 1.0

        # Backprop through W2, b2
        for i in range(self.hidden_dim):
            for j in range(self.vocab_size):
                self.dW2[i][j] += self._hidden[i] * d_logits[j]
        for j in range(self.vocab_size):
            self.db2[j] += d_logits[j]

        # Gradient flowing back into hidden layer
        d_hidden = [0.0] * self.hidden_dim
        for i in range(self.hidden_dim):
            for j in range(self.vocab_size):
                d_hidden[i] += self.W2[i][j] * d_logits[j]

        # Backprop through tanh: d_pre = d_hidden * (1 - tanh²)
        d_pre = [d_hidden[i] * tanh_backward(self._hidden[i])
                 for i in range(self.hidden_dim)]

        # Backprop through W1, b1
        flat_dim = self.context_len * self.embed_dim
        for i in range(flat_dim):
            for j in range(self.hidden_dim):
                self.dW1[i][j] += self._flat[i] * d_pre[j]
        for j in range(self.hidden_dim):
            self.db1[j] += d_pre[j]

        # Gradient w.r.t. flat embedding vector
        d_flat = [0.0] * flat_dim
        for i in range(flat_dim):
            for j in range(self.hidden_dim):
                d_flat[i] += self.W1[i][j] * d_pre[j]

        # Distribute gradients back to each embedding in the context
        for pos, idx in enumerate(self._context):
            start = pos * self.embed_dim
            for k in range(self.embed_dim):
                self.demb[idx][k] += d_flat[start + k]

        return loss

    def update(self, lr: float, batch_size: int):
        """Apply gradients and zero them out."""
        scale = lr / batch_size

        for i in range(len(self.W1)):
            for j in range(len(self.W1[0])):
                self.W1[i][j] -= scale * self.dW1[i][j]
                self.dW1[i][j] = 0.0
        for j in range(len(self.b1)):
            self.b1[j] -= scale * self.db1[j]
            self.db1[j] = 0.0

        for i in range(len(self.W2)):
            for j in range(len(self.W2[0])):
                self.W2[i][j] -= scale * self.dW2[i][j]
                self.dW2[i][j] = 0.0
        for j in range(len(self.b2)):
            self.b2[j] -= scale * self.db2[j]
            self.db2[j] = 0.0

        for i in range(self.vocab_size):
            for k in range(self.embed_dim):
                self.emb[i][k] -= scale * self.demb[i][k]
                self.demb[i][k] = 0.0

    def generate(self, idx2ch, num_chars=400, temperature=0.8, k=20, seed_idx=0):
        """Generate text using top-k sampling with temperature."""
        context = [seed_idx] * self.context_len
        out = []
        for _ in range(num_chars):
            logits = self.forward(context)
            # Top-k sampling
            vocab_size = self.vocab_size
            sorted_idx = sorted(range(vocab_size), key=lambda i: logits[i], reverse=True)
            top_k = sorted_idx[:k]
            restricted = [-float('inf')] * vocab_size
            for idx in top_k:
                restricted[idx] = logits[idx] / temperature
            m = max(r for r in restricted if r != -float('inf'))
            exps = [math.exp(r - m) if r != -float('inf') else 0.0 for r in restricted]
            total = sum(exps)
            probs = [e / total for e in exps]
            nxt = random.choices(range(vocab_size), weights=probs, k=1)[0]
            out.append(idx2ch[nxt])
            # Slide context window forward
            context = context[1:] + [nxt]
        return "".join(out)


# ── 3. TRAINING ──────────────────────────────────────────────────────────────

def get_batch(data, context_len, batch_size):
    max_start = len(data) - context_len - 1
    starts = [random.randint(0, max_start) for _ in range(batch_size)]
    inputs  = [data[i : i + context_len] for i in starts]
    targets = [data[i + context_len]      for i in starts]
    return inputs, targets


def train(model, train_data, val_data, num_steps=NUM_STEPS):
    print(f"\nTraining MLP | context={model.context_len} | hidden={model.hidden_dim} | steps={num_steps}")
    print(f"{'─'*60}")

    for step in range(num_steps):
        inputs, targets = get_batch(train_data, model.context_len, BATCH_SIZE)
        total_loss = 0.0
        for inp, tgt in zip(inputs, targets):
            logits = model.forward(inp)
            total_loss += model.backward(logits, tgt)
        model.update(LEARNING_RATE, BATCH_SIZE)

        if step % 500 == 0 or step == num_steps - 1:
            val_inputs, val_targets = get_batch(val_data, model.context_len, 256)
            val_loss = 0.0
            for inp, tgt in zip(val_inputs, val_targets):
                logits = model.forward(inp)
                probs  = softmax(logits)
                val_loss += cross_entropy(probs, tgt)
            val_loss /= 256
            train_loss = total_loss / BATCH_SIZE
            bar = "█" * max(0, int((4.5 - train_loss) * 8))
            print(f"  step {step:>4d} | train {train_loss:.4f} | val {val_loss:.4f} | {bar}")


# ── 4. MAIN ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  gosu-slm | Week 6 — MLP Language Model")
    print("=" * 60)

    random.seed(42)
    train_data, val_data, vocab_size, idx2ch = load_and_encode()

    flat_dim = CONTEXT_LEN * EMBED_DIM
    total_params = (vocab_size * EMBED_DIM +
                    flat_dim * HIDDEN_DIM + HIDDEN_DIM +
                    HIDDEN_DIM * vocab_size + vocab_size)
    print(f"\nArchitecture: Embed({vocab_size}x{EMBED_DIM}) → Linear({flat_dim}→{HIDDEN_DIM}) → Tanh → Linear({HIDDEN_DIM}→{vocab_size})")
    print(f"Context window: {CONTEXT_LEN} characters")
    print(f"Total parameters: {total_params:,}")

    model = MLPModel(vocab_size)
    train(model, train_data, val_data)

    newline_idx = list(idx2ch.keys())[list(idx2ch.values()).index('\n')]

    print(f"\n{'─'*60}")
    print("GENERATED TEXT (temp=0.8, top-k=20):")
    print(f"{'─'*60}")
    print(model.generate(idx2ch, num_chars=400,
                         temperature=0.8, k=20, seed_idx=newline_idx))

    print(f"\n{'─'*60}")
    print(f"Context window: {CONTEXT_LEN} chars → dramatically better output.")
    print("The model can now see phrases, not just individual characters.")
    print("Next week: attention — letting EVERY character see EVERY other.")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
