# rlhf-pipeline

**Phase:** 5 — Frontier ML Systems  
**Language:** Python (PyTorch)  
**Mirrors:** InstructGPT training pipeline, Anthropic's RLHF stack, Llama-2-Chat  
**Difficulty:** 10 / 10

---

## What this is

A minimal but complete implementation of the two dominant alignment training algorithms in 2024–2026:

1. **RLHF with PPO** — the original InstructGPT approach: train a reward model on human preference data, then use PPO to optimize the language model's policy against the reward model, with a KL-divergence penalty that keeps it close to the supervised baseline.
2. **DPO (Direct Preference Optimization)** — the 2023 algorithm that eliminates the reward model entirely and frames the alignment problem as a supervised classification task over pairs of (preferred, rejected) responses.

You will implement both from scratch on a small language model, train each to convergence on a synthetic preference dataset, measure the stability differences empirically, and understand at the mathematical level *why* DPO works and what it sacrifices relative to PPO.

---

## Why this matters

Every instruction-following model — GPT-4, Claude, Gemini, Llama-2-Chat — is trained with RLHF or one of its variants. Understanding the pipeline at the implementation level is required for research on alignment, fine-tuning, and model behavior. DPO in particular has been adopted broadly because it removes the instability of online RL while achieving competitive results. The tradeoffs between PPO and DPO are an active research question in 2026 — you should understand them at the algorithmic level, not just know that "DPO is more stable."

For your mathematical background: PPO is a constrained optimization problem (maximize reward subject to KL constraint, solved by a trust-region method). DPO is a reduction from RL to a supervised learning problem via the Bradley-Terry preference model. The connection between the two is a specific form of importance-weighted MLE — this is the mathematical substance of the Rafailov et al. paper, and it is worth working through in detail.

---

## Papers to read before writing a single line

| Paper | What to extract |
|---|---|
| Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT, 2022) | The full three-stage pipeline: SFT → RM training → PPO-with-KL. Read §3 (Methods) closely. The KL term in the reward is not regularization in the traditional sense — understand *why* it is there. |
| Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (2023) | Read the derivation in §4 completely. The key result: the optimal policy under the RLHF objective can be expressed in closed form, and DPO's loss is the MLE over that closed-form expression. Do the algebra yourself before reading the proof. |
| Schulman et al., *Proximal Policy Optimization Algorithms* (2017) | PPO is the RL algorithm. Read §2–3 (the clipped surrogate objective). Understand why clipping is used instead of the KL-penalty variant, and what the connection is to TRPO. |
| Bradley & Terry, *Rank Analysis of Incomplete Block Designs* (1952) | One page. The Bradley-Terry model is the preference probability model underlying both RM training and DPO. Read it. |
| Ziegler et al., *Fine-Tuning Language Models from Human Feedback* (2019) | The original RLHF paper for LMs. Predates InstructGPT but introduces the core setup. |
| Yuan et al., *Self-Rewarding Language Models* (2024) | The direction RLHF is heading: models that generate their own preference data. Read after implementing the core pipeline. |

---

## Architecture

```
rlhf-pipeline/
├── model/
│   ├── lm.py                   # Small GPT-2 (6 layers, d=512) — the policy model
│   ├── reward_model.py         # RM: same architecture + scalar head replacing logit head
│   └── value_model.py          # PPO value function: same architecture + scalar head
├── data/
│   ├── sft_data.py             # Supervised fine-tuning data: (prompt, good_response) pairs
│   ├── preference_data.py      # Preference data: (prompt, chosen, rejected) triples
│   └── synthetic.py            # Generate synthetic preference data from a gold reward function
├── sft/
│   └── train_sft.py            # Stage 1: supervised fine-tuning on good responses
├── reward_model/
│   ├── train_rm.py             # Stage 2: train reward model on preference pairs
│   ├── rm_loss.py              # Bradley-Terry loss: log σ(r_chosen - r_rejected)
│   └── evaluate_rm.py          # RM accuracy on held-out pairs, calibration plots
├── ppo/
│   ├── rollout.py              # Generate responses from policy, score with RM
│   ├── advantage.py            # GAE advantage estimation
│   ├── ppo_loss.py             # Clipped surrogate loss + value loss + entropy bonus
│   ├── kl_controller.py        # Adaptive KL coefficient (as in InstructGPT)
│   └── train_ppo.py            # The PPO training loop: rollout → advantage → update
├── dpo/
│   ├── dpo_loss.py             # DPO loss: log σ(β(log π_θ/π_ref)_chosen - (log π_θ/π_ref)_rejected)
│   └── train_dpo.py            # Offline DPO: supervised loop over preference dataset
├── evaluation/
│   ├── win_rate.py             # Compare PPO vs DPO vs SFT using gold reward function
│   ├── kl_tracking.py          # KL divergence from reference model over training
│   ├── reward_tracking.py      # Mean reward vs training step for PPO and DPO
│   └── stability.py            # Training loss variance, reward hacking detection
└── benchmarks/
    ├── bench_ppo_vs_dpo.py     # Win rate, KL, reward: PPO vs DPO at matched compute
    ├── bench_rm_accuracy.py    # RM accuracy vs dataset size curve
    └── bench_stability.py      # Loss curves, gradient norm, KL coefficient over time
```

---

## Implementation plan

### Stage 1 — Supervised Fine-Tuning (SFT)

The starting point for both PPO and DPO is a model already fine-tuned on high-quality demonstrations. Without SFT, the initial policy is so far from the preference distribution that RL training is unstable.

Train the small GPT on a set of (prompt, response) pairs where the responses are known to be good (in your synthetic setup, generated by a "gold" reward function that you define). Standard cross-entropy loss on the response tokens only (not the prompt). This model is the reference policy `π_ref` for the KL penalty later.

**Gate:** SFT model generates coherent responses to held-out prompts. Perplexity on a held-out demonstration set is lower than the pretrained model's perplexity.

### Stage 2 — Reward Model Training

The reward model takes a (prompt, response) pair and outputs a scalar. Train it on binary preference data: (prompt, chosen_response, rejected_response) triples, where `chosen` is the response a human (or gold function) prefers.

Loss:
```
L_RM = -E[log σ(r(x, y_w) - r(x, y_l))]
```
where `y_w` is the chosen (winning) response and `y_l` is the rejected (losing) response. This is the Bradley-Terry model.

Start the reward model from the SFT checkpoint (not a random initialization) — this is standard practice and significantly improves training stability.

Track: accuracy on held-out preference pairs, calibration (does confidence match accuracy?), and reward distribution on chosen vs rejected responses — these should be clearly separated.

**Gate:** RM accuracy > 70% on held-out pairs. Reward distribution for chosen responses is separated from rejected by > 1 standard deviation.

### Stage 3 — PPO Training

This is the most complex stage. The training loop:

1. **Rollout:** For each prompt in the batch, sample a response from the current policy `π_θ`. Score it with the frozen reward model. Compute the KL penalty from the reference policy `π_ref`.
   ```
   total_reward = r_RM(x, y) - β * KL(π_θ || π_ref)
   ```
   The KL is computed token-by-token: `sum_t [log π_θ(y_t | x, y_{<t}) - log π_ref(y_t | x, y_{<t})]`

2. **Advantage estimation:** Use Generalized Advantage Estimation (GAE) with the value model. The value model predicts the expected future reward from each token position.

3. **PPO update:** Compute the clipped surrogate loss:
   ```
   L_PPO = E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)]
   ```
   where `ratio = π_θ(y_t) / π_old(y_t)` is the importance weight and `A` is the advantage.

4. **KL controller:** Track the current KL divergence. If it exceeds a target (typically 0.1–0.2 nats), increase β. If it is below target, decrease β. This is the adaptive coefficient from InstructGPT §C.

Critical implementation details:
- The policy and value model share the transformer backbone but have separate heads. Freeze the bottom layers during early training.
- Log `π_old` (the policy at the start of this rollout batch) before doing any updates — the clipping requires the ratio to the *old* policy, not the current one.
- Normalize advantages within each mini-batch.
- Watch for reward hacking: if the policy finds a pattern the RM rates highly but is clearly wrong, the KL penalty is failing. Log examples throughout training.

**Gate:** Mean reward increases over 500 PPO steps. KL divergence from `π_ref` stays below 0.5 nats. No obvious reward hacking (verified by inspecting generated responses).

### Stage 4 — DPO Training

DPO eliminates the reward model and the online rollouts. Instead:

Loss:
```
L_DPO = -E[log σ(β * (log π_θ(y_w|x) - log π_ref(y_w|x)) - β * (log π_θ(y_l|x) - log π_ref(y_l|x)))]
```

This is a supervised loss over the fixed preference dataset. The reference model `π_ref` is frozen. The policy `π_θ` is initialized from the SFT checkpoint.

Key implementation details:
- Compute log-probabilities by summing token-level log-probs over the response tokens only
- Run `π_ref` and `π_θ` on both chosen and rejected in each batch (4 forward passes per step, or use cached `π_ref` log-probs)
- β is a hyperparameter (not adaptive). Common range: 0.1–0.5. Higher β = stay closer to reference.
- Monitor the "implicit reward" margin: `β * (log π_θ/π_ref)_chosen - β * (log π_θ/π_ref)_rejected` should increase. If it decreases, β is too high.

**Gate:** DPO implicit reward margin increases over 500 steps. Win rate against SFT baseline (using gold reward function) is positive.

### Stage 5 — Comparison and stability analysis

Run both algorithms to convergence. For each, track over all training steps:
- Mean reward on held-out prompts (using gold reward function, not the learned RM)
- KL divergence from `π_ref`
- Training loss curve and gradient norm
- Win rate against the SFT baseline

Key questions to answer empirically, not just claim:
- Which algorithm reaches higher reward? (Usually PPO, but by how much?)
- Which is more stable? (PPO has notoriously high variance; measure this.)
- What does reward hacking look like? Can you produce an example?
- What happens when you increase β in DPO — does it always hurt final performance or just training speed?
- At matched compute (same number of gradient steps × batch size × model FLOPs), which wins?

---

## What you will understand when this is done

- The mathematical connection between RLHF and DPO: DPO is not a heuristic, it is an exact reduction
- Why PPO is unstable and what each of its hyperparameters actually controls
- What reward hacking is, how to detect it, and why the KL penalty is the right defense
- Why DPO is more stable and what it sacrifices (online optimization = can track changing reward signal)
- How to implement and debug a full RL training loop: rollout, advantage estimation, policy update
- The Bradley-Terry preference model: what it assumes and when those assumptions break

---

## Completion criteria

- [ ] SFT model trained, perplexity improvement over base model documented
- [ ] RM accuracy > 70% on held-out pairs, calibration curve in `BENCHMARKS.md`
- [ ] PPO trains to positive mean reward with KL < 0.5 nats for 500+ steps
- [ ] DPO trains to positive win rate with no instability
- [ ] PPO vs DPO comparison table: reward, KL, stability, wall time in `BENCHMARKS.md`
- [ ] At least one reward hacking example identified and documented in `LEARNINGS.md`
- [ ] Mathematical derivation of the DPO loss from first principles in `LEARNINGS.md` (not copied from the paper — derived by hand)
