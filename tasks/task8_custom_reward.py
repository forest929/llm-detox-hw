"""Task 8 — Design and ship a custom reward [15 points].

Tasks 6 and 7 showed you two attractors: ``inv:detoxify`` collapses
the policy onto an OpenAI-style refusal template; ``rm:<your-RM>``
collapses onto a different template (in our runs, a Mandarin-greeting
or "I'm sorry could you provide more context" attractor). Design a
reward function that *can't be saturated by a single template*.

The function below is loaded by the verl reward worker when you launch
PPO with ``TOXIC_REWARD=custom:tasks.task8_custom_reward``. It runs in
the same docker container as the rollout. Detoxify, your trained RM,
and any other reward source are importable here.

Some hints (pick any combination, or invent your own):

  * **Saturating Detoxify above a threshold.** Once a completion is
    "clearly benign", uniform reward removes the incentive to push
    toward template attractors.
  * **Penalising repetition.** Trigram-repetition penalty bites where
    the policy starts looping on a phrase.
  * **Penalising length-cap hits.** If the policy learns to always
    run to the token cap, penalise that signal.
  * **Prompt-relevance signal.** A response that ignores the prompt
    can still score high on Detoxify by accident. Bag-of-words
    overlap or embedding similarity ties the reward to the prompt.
    Beware trivial echoing — bake a check against that.
  * **Blending or gating with your RM.** Detoxify and your RM
    disagree in interesting ways; their disagreement is signal.

The score function returns a list of floats — one reward per
completion, in the same order as the input ``texts`` list. Higher =
better.

Submit your final reward design + writeup in:

  * this file (the implementation)
  * ``submissions/task8_writeup.md`` (what you tried, what collapsed
    into what, what your final design looks like, why)
"""
from __future__ import annotations

from typing import Sequence


def reward_score(
    texts: Sequence[str],
    prompts: Sequence[str] | None = None,
) -> list[float]:
    """Return one reward per completion. Higher = better.

    The verl reward worker calls this once per training step with the
    flattened list of K-rollouts across the prompt batch.

    Args:
        texts: completions to score, one entry per completion.
        prompts: same-length list of the originating prompts (the verl
            dispatcher uses ``reward_score.prompt_conditioned`` below
            to decide whether to pass these). Set the attribute to
            ``False`` if your design is purely response-side.

    Returns:
        ``list[float]`` of the same length as ``texts``. Higher = better.

    See the top-of-file docstring for design hints (saturating Detoxify,
    repetition penalty, length-cap penalty, prompt-relevance, blending
    with your RM).
    """
    import re
    from detoxify import Detoxify

    dx = Detoxify("original")
    raw = dx.predict(list(texts))
    tox = raw["toxicity"]
    if not isinstance(tox, (list, tuple)):
        tox = [float(tox)]
    else:
        tox = [float(t) for t in tox]

    _STOPWORDS = {
        "the", "a", "an", "is", "it", "in", "on", "at", "to", "for",
        "of", "and", "or", "but", "you", "i", "me", "my", "your",
        "that", "this", "with", "as", "be", "was", "are", "were", "by",
        "we", "they", "he", "she", "do", "did", "not", "can", "will",
    }

    rewards = []
    for idx, (text, t) in enumerate(zip(texts, tox)):
        # --- Detoxify component (saturated): max reward once clearly benign ---
        if t <= 0.15:
            det_r = 1.0
        elif t <= 0.50:
            det_r = 1.0 - (t - 0.15) / 0.35
        else:
            det_r = -2.0 * (t - 0.50)  # -1 at t=1.0

        # --- Trigram repetition penalty (within the completion) ---
        tokens = text.lower().split()
        if len(tokens) >= 3:
            tris = [tuple(tokens[j:j + 3]) for j in range(len(tokens) - 2)]
            uniqueness = len(set(tris)) / len(tris)
            rep_penalty = (uniqueness - 1.0) * 0.4  # in [-0.4, 0]
        else:
            rep_penalty = 0.0

        # --- Prompt-relevance bonus (if prompts provided) ---
        if prompts is not None:
            p_words = {
                w for w in re.findall(r"\w+", prompts[idx].lower())
                if w not in _STOPWORDS and len(w) > 2
            }
            t_words = {
                w for w in re.findall(r"\w+", text.lower())
                if w not in _STOPWORDS and len(w) > 2
            }
            if p_words:
                overlap = len(p_words & t_words) / len(p_words)
                rel_bonus = min(overlap, 0.4) * 0.25  # max +0.1
            else:
                rel_bonus = 0.0
        else:
            rel_bonus = 0.0

        rewards.append(det_r + rep_penalty + rel_bonus)

    return rewards


# Tag the function so the verl dispatcher knows whether to pass prompts.
# Set to ``False`` if your reward is purely response-side.
reward_score.prompt_conditioned = True
