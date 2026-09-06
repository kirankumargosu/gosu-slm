"""
gosu-slm | Week 3 — The Bigram Model (Trainable)
=================================================
Last week we counted character pairs. This week we replace the count table
with a weight matrix — and make it trainable.

The key shift: instead of looking up probabilities from a fixed table,
we store a matrix of raw scores (called logits). Training adjusts these
scores so the model gets better at predicting the next character.

This week introduces:
  - The embedding table as a weight matrix
  - Cross-entropy loss — measuring how wrong the model is
  - The forward pass — computing logits from input tokens
  - Manual softmax — converting logits to probabilities

Run:
  python bigram_model.py

Author : Kiran Kumar Gosu
Series : Build an SLM from Scratch
Week   : 3 / 10
Repo   : github.com/kirankumargosu/gosu-slm
"""

import urllib.request
import os
import random
import math

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../", "data", "shakespeare.txt"))


# ── shared utilities (copied from week02) ────────────────────────────────────

def load_and_encode():
    """Load Shakespeare and return encoded data + vocabulary mappings."""
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


def get_batch(data, block_size=8, batch_size=32):
    max_start = len(data) - block_size - 1
    starts = [random.randint(0, max_start) for _ in range(batch_size)]
    inputs  = [data[i : i + block_size]     for i in starts]
    targets = [data[i + 1 : i + block_size + 1] for i in starts]
    return inputs, targets


# ── 1. MATH UTILITIES ────────────────────────────────────────────────────────

def softmax(logits: list[float]) -> list[float]:
    """
    Convert a list of raw scores (logits) into probabilities.

    Softmax does two things:
      1. Exponentiates each value (making everything positive)
      2. Divides by the sum (making everything sum to 1.0)

    The result is a valid probability distribution.

    We subtract the maximum value first (max trick) to prevent overflow
    when exponentiating large numbers — a standard numerical stability trick.
    """
    max_val = max(logits)
    exps = [math.exp(x - max_val) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def cross_entropy_loss(logits: list[float], target_idx: int) -> float:
    """
    Measure how wrong the model's prediction is.

    Cross-entropy loss for a single prediction:
      loss = -log(probability assigned to the correct class)

    If the model assigns probability 1.0 to the correct class → loss = 0.0 (perfect)
    If the model assigns probability 0.01 to the correct class → loss = 4.6 (bad)

    We want to minimise this value during training.
    """
    probs = softmax(logits)
    # Clamp to avoid log(0) which is -infinity
    correct_prob = max(probs[target_idx], 1e-9)
    return -math.log(correct_prob)


# ── 2. BIGRAM MODEL ──────────────────────────────────────────────────────────

class BigramModel:
    """
    A trainable bigram language model using an embedding (weight) matrix.

    Architecture:
      - One weight matrix of shape (vocab_size, vocab_size)
      - Each row is the "embedding" for one character
      - To predict the next character: look up the current character's row
        → that row contains the logits (raw scores) for all possible next chars
      - Apply softmax to get probabilities, sample to generate text

    This is identical in spirit to last week's probability table —
    but now the values are learnable parameters instead of fixed counts.
    """

    def __init__(self, vocab_size: int, learning_rate: float = 0.1):
        self.vocab_size = vocab_size
        self.lr = learning_rate

        # The weight matrix: shape (vocab_size × vocab_size)
        # Each W[i] is the logit vector for "what comes after character i"
        # Initialised with small random values (not zeros — zeros give no gradient signal)
        self.W = [
            [random.gauss(0, 0.01) for _ in range(vocab_size)]
            for _ in range(vocab_size)
        ]

    def forward(self, input_idx: int) -> list[float]:
        """
        Given the current character index, return logits for the next character.

        This is the 'forward pass' — computing the model's output from its input.
        For a bigram model it's just a row lookup.
        """
        return self.W[input_idx]

    def loss(self, inputs: list[list[int]], targets: list[list[int]]) -> float:
        """
        Compute average cross-entropy loss over a batch of sequences.

        For each position in each sequence:
          1. Get logits by looking up the input character's row
          2. Compare against the target character using cross-entropy
        Average the losses across all positions and sequences.
        """
        total_loss = 0.0
        total_count = 0

        for inp_seq, tgt_seq in zip(inputs, targets):
            for inp_char, tgt_char in zip(inp_seq, tgt_seq):
                logits = self.forward(inp_char)
                total_loss += cross_entropy_loss(logits, tgt_char)
                total_count += 1

        return total_loss / total_count

    def update(self, inputs: list[list[int]], targets: list[list[int]]) -> float:
        """
        Compute gradients and update weights using gradient descent.

        For the bigram model, the gradient of the cross-entropy loss
        with respect to the logits has a beautiful closed form:

          dL/d(logit_j) = prob_j - 1  if j == target
          dL/d(logit_j) = prob_j      if j != target

        In other words: the gradient is just (probabilities - one_hot_target).
        This is the softmax + cross-entropy gradient — one of the most important
        formulas in all of deep learning.

        Returns the average loss for this batch.
        """
        total_loss = 0.0
        total_count = 0

        # Accumulate gradients for each row of W that was used in this batch
        # grads[i] holds the sum of gradients for row i
        grads = [[0.0] * self.vocab_size for _ in range(self.vocab_size)]
        counts = [0] * self.vocab_size  # how many times each row was used

        for inp_seq, tgt_seq in zip(inputs, targets):
            for inp_char, tgt_char in zip(inp_seq, tgt_seq):
                logits = self.forward(inp_char)
                total_loss += cross_entropy_loss(logits, tgt_char)
                total_count += 1

                # Compute softmax probabilities
                probs = softmax(logits)

                # Gradient = probs - one_hot(target)
                # For the target index: grad = prob - 1
                # For all other indices: grad = prob
                for j in range(self.vocab_size):
                    grad = probs[j] - (1.0 if j == tgt_char else 0.0)
                    grads[inp_char][j] += grad

                counts[inp_char] += 1

        # Apply gradient descent: W = W - lr * gradient
        for i in range(self.vocab_size):
            if counts[i] > 0:
                for j in range(self.vocab_size):
                    # Average the gradient (divide by count) before stepping
                    self.W[i][j] -= self.lr * (grads[i][j] / counts[i])

        return total_loss / total_count

    def generate(self, idx2ch: dict, num_chars: int = 300, seed_idx: int = 0) -> str:
        """
        Generate text by sampling from the model one character at a time.

        At each step:
          1. Forward pass → logits for the current character
          2. Softmax → probability distribution
          3. Sample → next character index
          4. Repeat
        """
        current = seed_idx
        result = []

        for _ in range(num_chars):
            logits = self.forward(current)
            probs = softmax(logits)
            # random.choices samples according to weights
            next_idx = random.choices(range(self.vocab_size), weights=probs, k=1)[0]
            result.append(idx2ch[next_idx])
            current = next_idx

        return "".join(result)


# ── 3. TRAINING LOOP ─────────────────────────────────────────────────────────

def train(model, train_data, num_steps=1000, block_size=8, batch_size=32):
    """
    Run the training loop for a given number of steps.

    Each step:
      1. Sample a random batch from training data
      2. Compute gradients and update model weights
      3. Print loss periodically
    """
    print(f"\nTraining for {num_steps} steps...")
    losses = []

    for step in range(num_steps):
        inputs, targets = get_batch(train_data, block_size, batch_size)
        loss = model.update(inputs, targets)
        losses.append(loss)

        if step % 100 == 0 or step == num_steps - 1:
            avg = sum(losses[-100:]) / len(losses[-100:])
            bar = "█" * int((4.5 - min(avg, 4.5)) * 10)
            print(f"  Step {step:>4d} | loss {avg:.4f} | {bar}")

    return losses


# ── 4. MAIN ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  gosu-slm | Week 3 — Trainable Bigram Model")
    print("=" * 55)

    random.seed(42)
    train_data, val_data, vocab_size, idx2ch = load_and_encode()
    print(f"\nVocabulary size: {vocab_size}")
    print(f"Training tokens: {len(train_data):,}")

    # Create and train the model
    model = BigramModel(vocab_size, learning_rate=0.1)

    # Show loss before training
    inputs, targets = get_batch(train_data)
    initial_loss = model.loss(inputs, targets)
    print(f"\nInitial loss: {initial_loss:.4f}")
    print(f"(Random baseline ≈ {math.log(vocab_size):.4f} — log of vocab size)")

    # Train
    train(model, train_data, num_steps=1000)

    # Show loss after training
    inputs, targets = get_batch(train_data)
    final_loss = model.loss(inputs, targets)
    print(f"\nFinal loss: {final_loss:.4f}")

    # Validate on unseen data
    val_inputs, val_targets = get_batch(val_data)
    val_loss = model.loss(val_inputs, val_targets)
    print(f"Validation loss: {val_loss:.4f}")
    print(f"(Similar to train loss = good. Much higher = overfitting)")

    # Generate some text
    print(f"\n{'─'*55}")
    print("GENERATED TEXT (300 characters):")
    print(f"{'─'*55}")
    print(model.generate(idx2ch, num_chars=300))

    print(f"\n{'─'*55}")
    print("The model is learning. Loss is falling.")
    print("But it only looks ONE character back.")
    print("Next week: the training loop — built properly, from scratch.")
    print(f"{'─'*55}\n")


if __name__ == "__main__":
    main()
