# WordPiece Transformer — Architecture

## Hyperparameters

| Parameter | Value |
|---|---|
| Vocabulary size | 10 000 |
| Max sequence length | 128 tokens |
| Embedding dimension $d$ | 128 |
| Transformer layers | 2 |
| Attention heads | 4 |
| Positional encoding | RoPE |
| Aggregation | mean pooling |

## Learned parameters

| Parameter | Shape | Role |
|---|---|---|
| $E$ | $10000 \times 128$ | embedding table |
| $W_Q, W_K, W_V, W_O$ | $128 \times 128$ (×2 layers) | attention projections |
| $W_1, W_2$ | $128 \times 512$, $512 \times 128$ (×2 layers) | MLP |
| $W$ | $128 \times C$ | classification head |
| $b$ | $C$ | bias |

## Tokenizer: WordPiece

Words are split into sub-words learned statistically on the training corpus.
Unknown words are always decomposable into known sub-words.

```
"apprentissage"  →  ["apprent", "##iss", "##age"]   (3 tokens)
"chat"           →  ["chat"]                          (1 token)
"chatbot"        →  ["chat", "##bot"]                 (2 tokens)
```

**Key difference with NGram:** ~1-3 tokens per word vs ~20 with NGram.
A 50-word sentence → ~80 tokens, well within the 128-token limit.

## Pipeline

```
Input sentence
  │
  ▼  WordPiece tokenisation
  │  split into sub-words  →  integer IDs  →  padded/truncated to 128 tokens
  │
  ▼  Embedding lookup   E ∈ ℝ^{10000 × 128}
  │
  │  each token n  →  e_n ∈ ℝ^{128}
  │
  │  sequence: (128, 128)
  │
  ▼  Transformer Block × 2
  │
  │  ┌──────────────────────────────────────────────────┐
  │  │                                                  │
  │  │   x  ──► RMSNorm ──► Self-Attention ──► + ──► x │
  │  │                                                  │
  │  │   x  ──► RMSNorm ──► MLP            ──► + ──► x │
  │  │                                                  │
  │  └──────────────────────────────────────────────────┘
  │
  ▼  Mean pooling  (masked — ignores padding tokens)
  │
  │        1    N
  │   s =  ─   Σ  e_n   ∈  ℝ^{128}
  │        N   n=1
  │
  ▼  Classification head
  │
  │   z = Wᵀs + b   ∈  ℝ^C
  │
  ▼  Prediction
  │
  │   ŷ = argmax(z)
```

## Self-Attention (inside each Transformer block)

4 attention heads, each working on $128 / 4 = 32$ dimensions.

For each head, tokens exchange information via:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{32}}\right) V$$

- $Q, K, V \in \mathbb{R}^{128 \times 32}$ — projections of the token sequence
- RoPE encodes **relative positions** directly into $Q$ and $K$


## MLP (inside each Transformer block)

Applied token-by-token after attention:

$$\text{MLP}(x) = W_2 \cdot \text{ReLU}^2(W_1 x)$$

with $W_1 \in \mathbb{R}^{128 \times 512}$, $W_2 \in \mathbb{R}^{512 \times 128}$ — a 4× expansion then compression.
