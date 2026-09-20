"""
gosu-slm | Week 5 — Making It Generate: Sampling Strategies
============================================================
The model's output is a probability distribution over the vocabulary.
How we sample from that distribution changes everything.

Three strategies:
  1. Greedy      — always pick the highest-probability character
  2. Temperature — scale logits before softmax to control randomness
  3. Top-k       — restrict sampling to the k most likely characters

We reuse the trained Network from week04 and show how the same
model produces radically different text under each strategy.

Run:
  python generator.py

Author : Kiran Kumar Gosu
Series : Build an SLM from Scratch
Week   : 5 / 10
Repo   : github.com/kirankumargosu/gosu-slm
"""

import urllib.request, os, random, math, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week04"))
from trainer import load_and_encode, Network, train

SEED = 42


# ── 1. SAMPLING STRATEGIES ───────────────────────────────────────────────────

def softmax(logits):
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def greedy_sample(logits: list[float]) -> int:
    """
    Always pick the highest-probability token.

    Deterministic — same input always gives the same output.
    Produces safe, predictable text that quickly becomes repetitive.
    The model gets 'stuck' in loops because it always makes the
    most conservative choice.
    """
    return logits.index(max(logits))


def temperature_sample(logits: list[float], temperature: float = 1.0) -> int:
    """
    Scale logits by temperature before applying softmax, then sample.

    temperature < 1.0  → sharper distribution → more conservative/repetitive
    temperature = 1.0  → raw model distribution (no change)
    temperature > 1.0  → flatter distribution → more random/creative

    Intuition: temperature is like confidence.
      Low temp  = model is very sure, sticks to safe choices
      High temp = model is uncertain, explores more unusual combinations

    We divide logits by temperature (not multiply) so that:
      - Low temp makes differences between logits larger → more peaked
      - High temp makes differences smaller → more uniform
    """
    if temperature <= 0:
        return greedy_sample(logits)
    scaled = [l / temperature for l in logits]
    probs  = softmax(scaled)
    return random.choices(range(len(probs)), weights=probs, k=1)[0]


def topk_sample(logits: list[float], k: int = 10, temperature: float = 1.0) -> int:
    """
    Keep only the top-k logits, zero out the rest, then sample.

    This prevents the model from ever picking very unlikely characters
    (e.g., 'z' after 'qu' — technically possible but almost never correct).

    The k parameter controls the size of the 'shortlist':
      k = 1         → greedy (only pick the top choice)
      k = vocab_size → unrestricted (same as temperature sampling)
      k = 10-20     → sweet spot for most language models

    Steps:
      1. Find the top-k indices by logit value
      2. Build a new distribution with only those k values
      3. Sample from the restricted distribution
    """
    vocab_size = len(logits)

    # Get indices sorted by logit value, descending
    sorted_indices = sorted(range(vocab_size), key=lambda i: logits[i], reverse=True)
    top_k_indices  = sorted_indices[:k]

    # Build restricted logits — keep top-k, set rest to -infinity
    restricted = [-float('inf')] * vocab_size
    for idx in top_k_indices:
        restricted[idx] = logits[idx] / max(temperature, 1e-8)

    # Softmax over restricted logits (the -inf entries → probability 0)
    m = max(r for r in restricted if r != -float('inf'))
    exps = [math.exp(r - m) if r != -float('inf') else 0.0 for r in restricted]
    total = sum(exps)
    probs = [e / total for e in exps]

    return random.choices(range(vocab_size), weights=probs, k=1)[0]


# ── 2. GENERATOR ─────────────────────────────────────────────────────────────

def generate(model, idx2ch, num_chars=300, strategy="temperature",
             temperature=1.0, k=10, seed_idx=0):
    """
    Generate text using a given sampling strategy.

    strategy options: "greedy", "temperature", "topk"
    """
    vocab_size = model.vocab_size
    current = seed_idx
    out = []

    for _ in range(num_chars):
        logits = model.forward(current)

        if strategy == "greedy":
            nxt = greedy_sample(logits)
        elif strategy == "topk":
            nxt = topk_sample(logits, k=k, temperature=temperature)
        else:  # temperature
            nxt = temperature_sample(logits, temperature=temperature)

        out.append(idx2ch[nxt])
        current = nxt

    return "".join(out)


# ── 3. MAIN ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  gosu-slm | Week 5 — Sampling Strategies")
    print("=" * 60)

    random.seed(SEED)
    train_data, val_data, vocab_size, idx2ch = load_and_encode()

    # Train a model to use for generation experiments
    print("\nTraining model (1500 steps)...")
    model = Network(vocab_size, embed_dim=32)
    train(model, train_data, val_data, lr=0.05, num_steps=1500,
          batch_size=64, eval_every=500)

    newline_idx = list(idx2ch.keys())[list(idx2ch.values()).index('\n')]

    experiments = [
        ("GREEDY (temperature=0)",
         dict(strategy="greedy")),
        ("TEMPERATURE = 0.5 (conservative)",
         dict(strategy="temperature", temperature=0.5)),
        ("TEMPERATURE = 1.0 (balanced)",
         dict(strategy="temperature", temperature=1.0)),
        ("TEMPERATURE = 1.5 (creative/chaotic)",
         dict(strategy="temperature", temperature=1.5)),
        ("TOP-K = 5, temp = 0.8 (focused)",
         dict(strategy="topk", k=5, temperature=0.8)),
        ("TOP-K = 20, temp = 1.0 (sweet spot)",
         dict(strategy="topk", k=20, temperature=1.0)),
    ]

    for label, kwargs in experiments:
        random.seed(SEED)  # same seed so outputs are comparable
        print(f"\n{'─'*60}")
        print(f"  {label}")
        print(f"{'─'*60}")
        print(generate(model, idx2ch, num_chars=250,
                       seed_idx=newline_idx, **kwargs))

    print(f"\n{'─'*60}")
    print("Same weights. Same training. Completely different text.")
    print("The generation strategy shapes output as much as training does.")
    print("Next week: context. The model finally looks back more than 1 step.")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
