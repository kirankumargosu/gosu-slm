"""
gosu-slm | Week 4 — The Training Loop from Scratch
====================================================
This week we build a proper training infrastructure and introduce
the first real neural network layer: a linear transformation.

We implement everything by hand:
  - Linear layer (matrix multiply + bias)
  - Forward pass through a 2-layer network
  - Backward pass — computing gradients manually
  - Gradient descent weight update
  - Training + validation loss tracking
  - Loss curve visualisation in the terminal

The model is still bigram (1 character of context), but the training
machinery here will carry through every remaining week.

Run:
  python trainer.py

Author : Kiran Kumar Gosu
Series : Build an SLM from Scratch
Week   : 4 / 10
Repo   : github.com/kirankumargosu/gosu-slm
"""

import urllib.request
import os
import random
import math

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../", "data", "shakespeare.txt"))


# ── shared utilities ─────────────────────────────────────────────────────────

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


def get_batch(data, block_size=1, batch_size=64):
    max_start = len(data) - block_size - 1
    starts = [random.randint(0, max_start) for _ in range(batch_size)]
    inputs  = [data[i] for i in starts]            # single character input
    targets = [data[i + 1] for i in starts]        # single character target
    return inputs, targets


def softmax(logits):
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def cross_entropy(probs, target):
    return -math.log(max(probs[target], 1e-9))


# ── 1. LINEAR LAYER ──────────────────────────────────────────────────────────

class Linear:
    """
    A single fully-connected (linear) layer: y = xW + b

    This is the fundamental building block of every neural network.
    Given an input vector x of size in_features, it produces an output
    vector y of size out_features.

    Parameters:
      W  — weight matrix, shape (in_features, out_features)
      b  — bias vector, shape (out_features,)

    During backprop we compute and store gradients:
      dW — gradient w.r.t. W
      db — gradient w.r.t. b
      dx — gradient w.r.t. input (needed to propagate further back)
    """

    def __init__(self, in_features: int, out_features: int):
        # He initialisation: scale by sqrt(2/in_features)
        # This keeps activations at a reasonable scale during forward pass
        scale = math.sqrt(2.0 / in_features)
        self.W = [[random.gauss(0, scale) for _ in range(out_features)]
                  for _ in range(in_features)]
        self.b = [0.0] * out_features

        # Gradients — populated during backward()
        self.dW = [[0.0] * out_features for _ in range(in_features)]
        self.db = [0.0] * out_features

        # Cache input for use in backward()
        self._input = None

    def forward(self, x: list[float]) -> list[float]:
        """
        Compute y = xW + b.

        x is a row vector of length in_features.
        We compute the dot product of x with each column of W, then add bias.
        """
        self._input = x  # cache for backprop
        out_size = len(self.W[0])
        out = [self.b[j] for j in range(out_size)]
        for i, xi in enumerate(x):
            for j in range(out_size):
                out[j] += xi * self.W[i][j]
        return out

    def backward(self, d_out: list[float]) -> list[float]:
        """
        Compute gradients given the upstream gradient d_out.

        d_out is the gradient of the loss w.r.t. this layer's output.

        Chain rule gives us:
          dW[i][j] = input[i] * d_out[j]
          db[j]    = d_out[j]
          dx[i]    = sum_j(W[i][j] * d_out[j])
        """
        in_size  = len(self.W)
        out_size = len(self.W[0])
        x = self._input

        # Gradient w.r.t. weights and bias
        for i in range(in_size):
            for j in range(out_size):
                self.dW[i][j] += x[i] * d_out[j]
        for j in range(out_size):
            self.db[j] += d_out[j]

        # Gradient w.r.t. input (to pass back to previous layer)
        dx = [0.0] * in_size
        for i in range(in_size):
            for j in range(out_size):
                dx[i] += self.W[i][j] * d_out[j]
        return dx

    def update(self, lr: float):
        """Apply gradient descent and reset gradients to zero."""
        for i in range(len(self.W)):
            for j in range(len(self.W[0])):
                self.W[i][j] -= lr * self.dW[i][j]
                self.dW[i][j] = 0.0
        for j in range(len(self.b)):
            self.b[j] -= lr * self.db[j]
            self.db[j] = 0.0


# ── 2. EMBEDDING ─────────────────────────────────────────────────────────────

class Embedding:
    """
    Map integer token indices to dense vectors (embeddings).

    An embedding table is just a matrix where row i is the vector
    representation of token i. Instead of one-hot encoding (a vector
    of all zeros except a single 1), we use a dense learned vector.

    This is more efficient and lets the model learn that similar
    characters have similar representations.
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        scale = 0.1
        self.table = [[random.gauss(0, scale) for _ in range(embed_dim)]
                      for _ in range(vocab_size)]
        # Gradients for each embedding row
        self.d_table = [[0.0] * embed_dim for _ in range(vocab_size)]
        self._last_idx = None

    def forward(self, idx: int) -> list[float]:
        self._last_idx = idx
        return self.table[idx][:]  # return a copy

    def backward(self, d_out: list[float]):
        """Accumulate gradient into the embedding row that was used."""
        for j, g in enumerate(d_out):
            self.d_table[self._last_idx][j] += g

    def update(self, lr: float):
        for i in range(len(self.table)):
            for j in range(len(self.table[0])):
                self.table[i][j] -= lr * self.d_table[i][j]
                self.d_table[i][j] = 0.0


# ── 3. NETWORK ───────────────────────────────────────────────────────────────

class Network:
    """
    A 2-layer network: Embedding → Linear → Logits

    embed_dim controls the size of each character's representation.
    The linear layer projects from embed_dim back to vocab_size (logits).
    """

    def __init__(self, vocab_size: int, embed_dim: int = 32):
        self.embed   = Embedding(vocab_size, embed_dim)
        self.linear  = Linear(embed_dim, vocab_size)
        self.vocab_size = vocab_size
        self._last_probs = None
        self._last_target = None

    def forward(self, idx: int) -> list[float]:
        """Embed the input token, project to logits."""
        x = self.embed.forward(idx)
        logits = self.linear.forward(x)
        return logits

    def loss_and_backward(self, idx: int, target: int) -> float:
        """
        Forward pass + compute loss + backward pass in one step.

        The gradient of softmax cross-entropy w.r.t. logits is:
          d_logits[j] = probs[j] - 1  if j == target
          d_logits[j] = probs[j]      otherwise

        This is the cleanest gradient formula in deep learning.
        """
        logits = self.forward(idx)
        probs  = softmax(logits)
        loss   = cross_entropy(probs, target)

        # Gradient of loss w.r.t. logits
        d_logits = probs[:]
        d_logits[target] -= 1.0

        # Backprop through linear layer
        d_embed = self.linear.backward(d_logits)

        # Backprop through embedding
        self.embed.backward(d_embed)

        return loss

    def update(self, lr: float):
        self.embed.update(lr)
        self.linear.update(lr)

    def generate(self, idx2ch: dict, num_chars: int = 400, seed: int = 0) -> str:
        current = seed
        out = []
        for _ in range(num_chars):
            logits = self.forward(current)
            probs  = softmax(logits)
            nxt    = random.choices(range(self.vocab_size), weights=probs, k=1)[0]
            out.append(idx2ch[nxt])
            current = nxt
        return "".join(out)


# ── 4. TRAINING LOOP ─────────────────────────────────────────────────────────

def train(model, train_data, val_data, lr=0.05, num_steps=2000, batch_size=64, eval_every=200):
    """
    Full training loop with periodic validation.

    Every eval_every steps we pause training and compute validation loss.
    If validation loss is rising while training loss falls — that's overfitting.
    """
    print(f"\nTraining | lr={lr} | steps={num_steps} | batch={batch_size}")
    print(f"{'─'*60}")

    train_losses = []
    val_losses   = []

    for step in range(num_steps):
        # Sample a batch and run forward + backward for each example
        inputs, targets = get_batch(train_data, batch_size=batch_size)
        batch_loss = 0.0
        for inp, tgt in zip(inputs, targets):
            batch_loss += model.loss_and_backward(inp, tgt)

        # Average the gradients across the batch before updating
        # (our backward accumulates, so we scale the update by 1/batch_size)
        model.update(lr / batch_size)

        train_losses.append(batch_loss / batch_size)

        if step % eval_every == 0 or step == num_steps - 1:
            # Evaluate on validation data (no gradient updates)
            val_inputs, val_targets = get_batch(val_data, batch_size=256)
            val_loss = 0.0
            for inp, tgt in zip(val_inputs, val_targets):
                logits = model.forward(inp)
                probs  = softmax(logits)
                val_loss += cross_entropy(probs, tgt)
            val_loss /= 256
            val_losses.append(val_loss)

            avg_train = sum(train_losses[-eval_every:]) / len(train_losses[-eval_every:])
            bar = "█" * max(0, int((4.5 - avg_train) * 8))
            print(f"  step {step:>4d} | train {avg_train:.4f} | val {val_loss:.4f} | {bar}")

    return train_losses, val_losses


def plot_loss_curve(losses, label="loss", width=50):
    """ASCII loss curve in the terminal."""
    if len(losses) < 2:
        return
    mn, mx = min(losses), max(losses)
    rng = mx - mn or 1.0
    print(f"\n  {label} curve (each point = avg of recent steps):")
    # Sample 50 evenly-spaced points
    step = max(1, len(losses) // width)
    sampled = losses[::step][:width]
    for val in sampled:
        bar_len = int((val - mn) / rng * 20)
        print(f"  {val:.3f} {'▓' * bar_len}")


# ── 5. MAIN ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  gosu-slm | Week 4 — Training Loop from Scratch")
    print("=" * 60)

    random.seed(42)
    train_data, val_data, vocab_size, idx2ch = load_and_encode()

    model = Network(vocab_size, embed_dim=32)
    print(f"\nModel: Embedding({vocab_size}→32) + Linear(32→{vocab_size})")
    total_params = vocab_size * 32 + 32 * vocab_size + vocab_size
    print(f"Total parameters: {total_params:,}")

    train_losses, val_losses = train(
        model, train_data, val_data,
        lr=0.05, num_steps=2000, batch_size=64, eval_every=200
    )

    plot_loss_curve(train_losses, label="train")

    print(f"\n{'─'*60}")
    print("GENERATED TEXT:")
    print(f"{'─'*60}")
    print(model.generate(idx2ch, num_chars=400))

    print(f"\n{'─'*60}")
    print("The training loop is solid. Gradients flow. Loss falls.")
    print("But we're still only looking at 1 character of context.")
    print("Next week: sampling strategies — how we generate matters as much as how we train.")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
