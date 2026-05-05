# FastText — Architecture

## Hyperparameters

| Parameter | Value |
|---|---|
| Dimension $d$ | 128 |
| Character n-grams | 3 to 6 |
| Hash buckets | 100 000 |
| Aggregation | mean pooling |

## Vocabulary

$$V = \underbrace{3}_{\text{PAD, UNK, EOS}} + \underbrace{n_{\text{words}}}_{\text{known words}} + \underbrace{100\,000}_{\text{n-gram buckets}}$$

## Learned parameters

| Parameter | Shape | Role |
|---|---|---|
| $E$ | $V \times 128$ | embedding table |
| $W$ | $128 \times C$ | projection to classes |
| $b$ | $C$ | bias |

## Pipeline

```
Input sentence
  │
  ▼  Tokenisation
  │  each word  ──►  word ID  +  IDs of its character n-grams
  │             ──►  sequence of N indices
  │
  ▼  Embedding lookup
  │  each index n  ──►  e_n ∈ ℝ¹²⁸   (row n of E)
  │
  ▼  Mean pooling
  │
  │        1   N
  │   s =  ─  Σ  e_n   ∈  ℝ¹²⁸
  │        N  n=1
  │
  ▼  Classification head
  │
  │   z = Wᵀs + b   ∈  ℝᶜ
  │
  ▼  Prediction
  │
  │   ŷ = argmax(z)
```