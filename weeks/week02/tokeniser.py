"""
gosu-slm | Week 2 — Tokenisation from Scratch
==============================================
Before a model can learn anything, it needs to convert text into numbers.
That process is called tokenisation.

We're using character-level tokenisation — the simplest kind.
Every unique character gets a unique integer ID. That's it.

The two core operations:
  encode("hello") → [46, 43, 50, 50, 53]   (string → list of ints)
  decode([46, 43, 50, 50, 53]) → "hello"    (list of ints → string)

We also build the full data pipeline here:
  - Load Shakespeare
  - Build vocabulary
  - Encode the entire text into a flat list of integers
  - Split into train (90%) and validation (10%) sets
  - Create batches of (input, target) pairs for training

Run:
  python tokeniser.py

Author : Kiran Kumar Gosu
Series : Build an SLM from Scratch
Week   : 2 / 10
Repo   : github.com/kirankumargosu/gosu-slm
"""

import urllib.request
import os
import random

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../", "data", "shakespeare.txt"))


# ── 1. DATA LOADER ───────────────────────────────────────────────────────────

def load_shakespeare() -> str:
    if not os.path.exists(DATA_PATH):
        print("Downloading Shakespeare...")
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        urllib.request.urlretrieve(SHAKESPEARE_URL, DATA_PATH)
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return f.read()


# ── 2. VOCABULARY ────────────────────────────────────────────────────────────

class Vocabulary:
    """
    Holds the character ↔ integer mappings for our dataset.

    Every model needs a fixed vocabulary — the set of all possible
    tokens it can see and produce. Ours has 65 entries (Shakespeare
    uses 65 unique characters including letters, punctuation, spaces).
    """

    def __init__(self, text: str):
        # Collect every unique character, sort for consistency
        self.chars = sorted(set(text))
        self.size = len(self.chars)

        # Forward mapping: character → integer
        self.ch2idx = {ch: idx for idx, ch in enumerate(self.chars)}

        # Reverse mapping: integer → character
        self.idx2ch = {idx: ch for idx, ch in enumerate(self.chars)}

    def __repr__(self):
        return f"Vocabulary(size={self.size}, chars={repr(''.join(self.chars))})"


# ── 3. ENCODER / DECODER ─────────────────────────────────────────────────────

def encode(text: str, vocab: Vocabulary) -> list[int]:
    """
    Convert a string into a list of integers.

    Each character is replaced by its index in the vocabulary.
    This is the representation the model actually works with —
    it never sees characters, only numbers.

    Example:
      encode("Hi", vocab) → [20, 47]
    """
    return [vocab.ch2idx[ch] for ch in text]


def decode(indices: list[int], vocab: Vocabulary) -> str:
    """
    Convert a list of integers back into a string.

    The inverse of encode(). Used to turn the model's output
    (a sequence of integer predictions) back into readable text.

    Example:
      decode([20, 47], vocab) → "Hi"
    """
    return "".join(vocab.idx2ch[idx] for idx in indices)


# ── 4. TRAIN / VALIDATION SPLIT ──────────────────────────────────────────────

def train_val_split(data: list[int], val_fraction: float = 0.1) -> tuple[list[int], list[int]]:
    """
    Split the encoded dataset into training and validation sets.

    We use the first 90% for training and hold back 10% for validation.
    The validation set lets us check whether the model is genuinely
    learning or just memorising — it never sees validation data during training.

    We do NOT shuffle here. Order matters in language — the model learns
    that certain sequences follow others. Shuffling would break that signal.
    """
    split_idx = int(len(data) * (1 - val_fraction))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    return train_data, val_data


# ── 5. BATCH GENERATION ──────────────────────────────────────────────────────

def get_batch(
    data: list[int],
    block_size: int = 8,
    batch_size: int = 4,
) -> tuple[list[list[int]], list[list[int]]]:
    """
    Sample a random batch of (input, target) pairs from the data.

    This is how training data is fed to the model — in small random chunks.

    block_size: how many characters the model sees at once (context window)
    batch_size: how many independent sequences to process in parallel

    For each sequence:
      input  = data[i : i + block_size]       ← the context
      target = data[i + 1 : i + block_size+1]  ← what comes next at each position

    Example with block_size=4, one sequence starting at index 10:
      data[10:14] = [46, 43, 50, 50]  (input:  "hell")
      data[11:15] = [43, 50, 50, 53]  (target: "ello")

    The model learns: given [46], predict 43.
                      given [46, 43], predict 50.
                      given [46, 43, 50], predict 50.
                      given [46, 43, 50, 50], predict 53.

    One sequence of length block_size gives us block_size training examples.
    """
    # Pick batch_size random starting positions
    # We stop block_size before the end so there's always a full target sequence
    max_start = len(data) - block_size - 1
    start_indices = [random.randint(0, max_start) for _ in range(batch_size)]

    inputs  = [data[i : i + block_size]     for i in start_indices]
    targets = [data[i + 1 : i + block_size + 1] for i in start_indices]

    return inputs, targets


# ── 6. MAIN ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  gosu-slm | Week 2 — Tokenisation from Scratch")
    print("=" * 55)

    # Load text
    text = load_shakespeare()
    print(f"\nLoaded {len(text):,} characters")

    # Build vocabulary
    vocab = Vocabulary(text)
    print(f"\n{vocab}")
    print(f"Characters: {repr(''.join(vocab.chars))}")

    # Encode the full text
    data = encode(text, vocab)
    print(f"\nEncoded text: {len(data):,} integers")
    print(f"First 20 integers: {data[:20]}")
    print(f"First 20 characters: {repr(text[:20])}")

    # Verify round-trip: encode then decode should give back the original
    sample = "To be, or not to be"
    encoded = encode(sample, vocab)
    decoded = decode(encoded, vocab)
    print(f"\nRound-trip check:")
    print(f"  Original : {repr(sample)}")
    print(f"  Encoded  : {encoded}")
    print(f"  Decoded  : {repr(decoded)}")
    assert sample == decoded, "Round-trip failed!"
    print(f"  ✓ Round-trip passed")

    # Train / validation split
    train_data, val_data = train_val_split(data, val_fraction=0.1)
    print(f"\nTrain / validation split:")
    print(f"  Train : {len(train_data):,} tokens ({len(train_data)/len(data)*100:.1f}%)")
    print(f"  Val   : {len(val_data):,} tokens ({len(val_data)/len(data)*100:.1f}%)")

    # Show a sample batch
    print(f"\nSample batch (block_size=8, batch_size=2):")
    inputs, targets = get_batch(train_data, block_size=8, batch_size=2)
    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        print(f"\n  Sequence {i+1}:")
        print(f"    Input  integers : {inp}")
        print(f"    Target integers : {tgt}")
        print(f"    Input  text     : {repr(decode(inp, vocab))}")
        print(f"    Target text     : {repr(decode(tgt, vocab))}")

    print(f"\n{'─'*55}")
    print("The entire training pipeline is in place.")
    print("Next week: we build the first trainable model.")
    print(f"{'─'*55}\n")

    return vocab, train_data, val_data


if __name__ == "__main__":
    random.seed(42)
    main()
