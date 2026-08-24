"""
gosu-slm | Week 1 — What Is a Language Model, Really?
======================================================
The simplest possible language model: a character-level bigram.

A bigram model answers one question:
  "Given the character I just saw, what character is most likely to come next?"

It does this by counting. We read through all of Shakespeare and tally up
every time character X is followed by character Y. After reading everything,
we convert those tallies into probabilities. That's our model.

No neural networks. No training loop. Just counting and probability.
It produces terrible text — but it *works*, and it teaches us everything
that comes after.

Run this file directly:
  python bigram.py

It will:
  1. Download Shakespeare if not already present
  2. Build the bigram probability table
  3. Generate 500 characters of "Shakespeare"
  4. Print some stats about what it learned

Author : Kiran Kumar Gosu
Series : Build an SLM from Scratch
Week   : 1 / 10
Repo   : github.com/kirankumargosu/gosu-slm
"""

import urllib.request
import os
import random


# ── 1. DATA LOADER ───────────────────────────────────────────────────────────

# The URL of the complete works of Shakespeare — a classic ML dataset.
# It's about 1MB of plain text, completely free and public domain.
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# We store the file in the parent data/ folder so all weeks share it.
DATA_PATH = os.path.join(os.path.dirname(__file__), "../../", "data", "shakespeare.txt")


def load_shakespeare() -> str:
    """
    Load Shakespeare from disk. If not found, download it automatically.
    Returns the entire text as a single Python string.
    """
    # Normalise the path so ".." gets resolved cleanly on all operating systems
    path = os.path.normpath(DATA_PATH)

    if not os.path.exists(path):
        print("Shakespeare not found locally. Downloading (~1MB)...")
        # Make sure the data/ directory exists before writing
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Try the simple download first
        try:
            urllib.request.urlretrieve(SHAKESPEARE_URL, path)
            print(f"Saved to: {path}")
        except Exception as e:
            # Some macOS/Python installs fail SSL verification — try a safer alternative
            print(f"Standard download failed: {e}")
            print("Attempting download with certifi-backed SSL context...")
            try:
                import ssl
                import certifi

                ctx = ssl.create_default_context(cafile=certifi.where())
                with urllib.request.urlopen(SHAKESPEARE_URL, context=ctx) as resp, open(path, "wb") as out:
                    out.write(resp.read())
                print(f"Saved to: {path}")
            except Exception as e2:
                # As a last resort, fall back to an unverified context (insecure)
                print(f"certifi download failed: {e2}")
                print("Retrying with unverified SSL (insecure) — consider installing the 'certifi' package or running the Python Install Certificates.command on macOS.")
                import ssl

                ctx = ssl._create_unverified_context()
                with urllib.request.urlopen(SHAKESPEARE_URL, context=ctx) as resp, open(path, "wb") as out:
                    out.write(resp.read())
                print(f"Saved to: {path} (insecure HTTPS)")
    else:
        print(f"Loading Shakespeare from: {path}")

    # Read the entire file as a single string
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"First 100 characters:\n{text[:100]!r}, ... total {len(text):,} characters")
    return text


# ── 2. VOCABULARY ────────────────────────────────────────────────────────────

def build_vocab(text: str) -> tuple[list, dict, dict]:
    """
    Build the character vocabulary from the text.

    Returns three things:
      chars   — sorted list of every unique character in the text
      ch2idx  — dict mapping character → integer  (e.g. 'a' → 0)
      idx2ch  — dict mapping integer → character  (e.g. 0 → 'a')

    This vocabulary is the foundation of everything. The model never sees
    actual characters — it only ever sees their integer IDs.
    """
    # set() gives us unique characters; sorted() makes the ordering consistent
    # so that 'a' always maps to the same integer no matter when we run this
    chars = sorted(set(text))

    # Dictionary comprehension: {char: index for each (index, char) pair}
    ch2idx = {ch: idx for idx, ch in enumerate(chars)}
    print(f"Top 10 characters to index : {dict(list(ch2idx.items())[:10])}")

    # Reverse mapping: {index: char}
    idx2ch = {idx: ch for idx, ch in enumerate(chars)}
    print(f"Top 10 index to character  : {dict(list(idx2ch.items())[:10])}")

    return chars, ch2idx, idx2ch


# ── 3. BIGRAM TABLE ──────────────────────────────────────────────────────────

def build_bigram_table(text: str, ch2idx: dict, vocab_size: int) -> list[list[float]]:
    """
    Build the bigram probability table.

    This is the core of the model. We create a vocab_size x vocab_size matrix.
    Each cell [i][j] will store:
      "given that the current character is i, the probability that the next character is j"

    Step 1 — Count: scan through the text, increment counts[i][j] every time character i is followed by character j.

    Step 2 — Normalise: divide each row by its sum so the values become probabilities (they sum to 1.0 across each row).
    """

    # Initialise the count matrix with zeros
    # We use a small smoothing value (1) instead of 0 to avoid zero probabilities.
    # This is called Laplace smoothing — without it, any unseen bigram would
    # have probability 0, which can cause problems during generation.
    
    counts = [[1] * vocab_size for _ in range(vocab_size)]

    # Scan through every consecutive pair of characters in the text.
    # zip(text, text[1:]) pairs each character with the one that follows it.
    # Example: "hello" → (h,e), (e,l), (l,l), (l,o)
    for ch1, ch2 in zip(text, text[1:]):
        i = ch2idx[ch1]  # index of the current character
        j = ch2idx[ch2]  # index of the next character
        counts[i][j] += 1

    print(f"First 5 rows of counts:\n{counts[:5]}")  # print the first 5 rows of counts for debugging
    # Normalise each row into probabilities.
    # After this step, every row sums to exactly 1.0.
    probs = []
    for row in counts:
        total = sum(row)
        # Divide each count by the row total to get a probability
        probs.append([count / total for count in row])

    print(f"Bigram table built: {vocab_size}x{vocab_size} = {vocab_size**2:,} probabilities")
    return probs


# ── 4. PRINT STATS ───────────────────────────────────────────────────────

def print_stats(text: str, chars: list, probs: list[list[float]]) -> None:
    """Print a summary of what the model learned."""
    vocab_size = len(chars)
    print(f"\n{'─' * 50}")
    print("MODEL STATS")
    print(f"{'─' * 50}")
    print(f"  Training text   : {len(text):>10,} characters")
    print(f"  Vocabulary size : {vocab_size:>10,} unique characters")
    print(f"  Bigram table    : {vocab_size}x{vocab_size} = {vocab_size**2:,} probabilities")
    print(f"  Model 'params'  : {vocab_size**2:,} (just a lookup table)")
    print(f"  Characters      : {repr(''.join(chars))}")

# ── 5. PRINT TOP BIGRAMS ──────────────────────────────────────────────────────

def print_top_bigrams(
    probs: list[list[float]],
    idx2ch: dict,
    vocab_size: int,
    top_n: int = 10
) -> None:
    """
    Print the most probable bigrams the model learned.

    This gives us a window into what the model knows. We'd expect to see
    things like 'th' → 'e' dominating, since 'the' is the most common
    word in English.
    """
    print(f"\n{'─' * 50}")
    print(f"TOP {top_n} MOST LIKELY CHARACTER PAIRS")
    print(f"{'─' * 50}")

    # Collect all bigram probabilities into a flat list of (prob, i, j) tuples
    all_bigrams = []
    for i in range(vocab_size):
        for j in range(vocab_size):
            all_bigrams.append((probs[i][j], i, j))

    # Sort by probability descending, take top_n
    all_bigrams.sort(reverse=True)

    for prob, i, j in all_bigrams[:top_n]:
        # repr() shows special characters like \n as \n instead of a newline
        ch1 = repr(idx2ch[i])
        ch2 = repr(idx2ch[j])
        bar = "█" * int(prob * 40)  # visual bar proportional to probability
        print(f"  {ch1:6} → {ch2:6}  {prob:.3f}  {bar}")


# ── 6. TEXT GENERATION ───────────────────────────────────────────────────────

def generate(
    probs: list[list[float]],
    idx2ch: dict,
    vocab_size: int,
    num_chars: int = 500,
    seed_char: str = "\n"
) -> str:
    """
    Generate text by sampling from the bigram probability table.

    Starting from seed_char, we:
      1. Look up the current character's row in the probability table
      2. Sample the next character according to those probabilities
      3. That sampled character becomes our new current character
      4. Repeat until we've generated num_chars characters

    The key function here is random.choices() — it samples from a list
    according to weights (probabilities). Higher weight = more likely to pick.
    """
    # We need the character-to-index mapping to look up our seed character
    # Build it from idx2ch by reversing it
    ch2idx = {ch: idx for idx, ch in idx2ch.items()}

    # Start from our seed character
    current_idx = ch2idx.get(seed_char, 0)
    result = []

    for _ in range(num_chars):
        # Get the probability distribution for the current character
        # probs[current_idx] is a list of vocab_size probabilities
        row = probs[current_idx]

        # Sample the next character index using the probabilities as weights.
        # random.choices returns a list, so we take [0] to get the single item.
        next_idx = random.choices(range(vocab_size), weights=row, k=1)[0]

        # Convert the index back to a character and add it to our result
        result.append(idx2ch[next_idx])

        # The next character becomes our new current character
        current_idx = next_idx

    return "".join(result)


def main():
    print("=" * 50)
    print("  gosu-slm | Week 1 — Bigram Language Model")
    print("=" * 50)

    # Step 1: Load the data
    text = load_shakespeare()

    # Step 2: Build the vocabulary
    chars, ch2idx, idx2ch = build_vocab(text)
    vocab_size = len(chars)

    # Step 3: Build the bigram probability table
    print("\nBuilding bigram table...")
    probs = build_bigram_table(text, ch2idx, vocab_size)
    print("Done.")

    # Step 4: Print stats
    print_stats(text, chars, probs)

    # Step 5: Show what the model learned
    print_top_bigrams(probs, idx2ch, vocab_size, top_n=10)

    # Step 6: Generate some text
    print(f"\n{'─' * 50}")
    print("GENERATED TEXT (500 characters)")
    print(f"{'─' * 50}")
    generated = generate(probs, idx2ch, vocab_size, num_chars=500)
    print(generated)

    print(f"\n{'─' * 50}")
    print("Notice: the output looks like noise — but it's not random.")
    print("It respects character frequencies learned from Shakespeare.")
    print("Next week: tokenisation. The foundation everything else builds on.")
    print(f"{'─' * 50}\n")


if __name__ == "__main__":
    # random.seed ensures reproducible output — same seed = same generated text
    # Remove this line if you want different output each run
    random.seed(42)
    main()
