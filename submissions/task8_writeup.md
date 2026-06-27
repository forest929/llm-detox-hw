# Task 8 — Custom Reward Design

## What I tried

The core insight from Tasks 6 and 7 is that any scalar reward with an uncapped
maximum will be saturated by a single fixed template:

- Task 6 (inv:detoxify): system-prompt echo (near-zero toxicity, ~20 tokens)
- Task 7 (RM): Cyrillic filler at max length (RM rewards long, non-toxic text)

Both attractors share the same structure: the policy finds a prompt-independent
template that maximises the reward faster than any on-topic response.

## Final reward design

Three components, weighted to prevent saturation:

### 1. Saturated Detoxify (primary signal, range ~ [-1, 1])

tox <= 0.15  ->  reward = 1.0   (clearly benign: full reward, no incentive to push further)
0.15 < tox <= 0.50  ->  linear 1 to 0   (declining zone)
tox > 0.50  ->  -2 * (tox - 0.50)   (clearly toxic: strong penalty)

The ceiling at tox <= 0.15 is the key difference from inv:detoxify. Once a
completion is clearly benign, the policy cannot score higher by becoming *more*
benign -- removing the gradient that drives template attractors.

### 2. Trigram repetition penalty (range ~ [-0.4, 0])

rep_penalty = (unique_trigrams / total_trigrams - 1) * 0.4

Penalises completions that loop on the same n-gram. This directly targets the
Cyrillic-filler attractor from Task 7 and any system-prompt echo pattern.

### 3. Prompt-relevance bonus (range ~ [0, 0.1])

overlap = |content_words(prompt) intersect content_words(completion)| / |content_words(prompt)|
rel_bonus = min(overlap, 0.4) * 0.25

A small bonus for sharing content words with the prompt (stop words excluded).
Tied to the prompt, so prompt-independent templates score 0 here. Capped to
prevent trivial prompt-echoing.

## Results and attractor analysis

Task 8 eval K=16 diagnostic:
- mild_prefix: support_rate 0.133, mean_max 0.145, mean_std 0.038
- direct_provocation: support_rate 0.000, mean_max 0.095, mean_std 0.024
- rtp_challenging: support_rate 0.320, mean_max 0.296, mean_std 0.084

The custom reward did not fully prevent attractor collapse. The worst-of-16
completions still echo the system prompt with appended junk tokens/characters,
achieving low Detoxify scores while collecting the saturation reward. However
the attractor is softer: mean_std (0.038-0.084) is higher than Task 6's
near-zero std (0.001), meaning the policy still explores rather than outputting
a single fixed string.

## Why it partially worked and where it still failed

What worked: The Detoxify saturation cap removed the strongest gradient toward
the system-prompt echo -- the policy does not purely echo anymore. The
repetition penalty keeps trigram diversity higher than Task 7's Cyrillic filler.

What still failed: The new attractor is system-prompt echo + random Unicode
noise. The noise is diverse enough to avoid the trigram penalty while still
scoring near the Detoxify ceiling. The prompt-relevance signal is too weak
(max +0.1) to overcome this.

What would improve it: A stronger prompt-relevance signal (e.g. embedding
cosine similarity rather than bag-of-words overlap), or penalising non-ASCII
characters directly, would likely close this loophole. A composite reward that
checks for semantic coherence (entailment or LM perplexity against the prompt)
would be harder to game with noise.
