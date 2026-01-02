# Q-Learning vs SARSA (Tabular RL) — Risk vs Reward

This project compares two classic **tabular reinforcement learning** methods—**Q-Learning** and **SARSA**—on a gridworld-style task where the agent must choose between a **safe route** and a **faster but risky route**.

The goal is not “highest score.” The point is to show how different update rules lead to different behaviors under risk—exactly the same kind of tradeoff you care about in robotics (aggressive vs conservative motion / navigation).

---

## What this shows

Q-Learning often learns **riskier, reward-maximizing** behavior, while SARSA often learns **safer** behavior because it updates based on the actions it actually takes during exploration.

---

## Algorithms (what’s different)

### Q-Learning (off-policy)
Q-Learning updates toward the best possible next action, even if the agent won’t take that action during exploration.

Update rule (in plain text):
Q(s,a) = Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

Interpretation:
Assume you will behave optimally next step.

This tends to produce more aggressive policies near hazards.

---

### SARSA (on-policy)
SARSA updates toward the next action the agent actually takes (including exploration).

Update rule (in plain text):
Q(s,a) = Q(s,a) + alpha * [ r + gamma * Q(s',a') - Q(s,a) ]

Interpretation:
Update based on what you actually do.

This tends to produce safer policies when exploration can cause failures.

---

## Environment / Task

A small gridworld-style environment where:
- There is a goal with positive reward
- There is a hazard region with a large negative penalty
- The shortest route passes near hazards (fast but risky)
- A longer route avoids hazards (slower but safer)

---

## Training setup

- Tabular state-action values (no neural networks)
- Epsilon-greedy exploration
- Hyperparameters:
  - alpha (learning rate)
  - gamma (discount)
  - epsilon (exploration)

Logged metrics:
- episodic return
- failures (hazard hits)
- learned policy visualization (arrows / heatmap)

---

## Results / Takeaways

Observed behavior:
- Q-Learning typically converges to a shorter, riskier path.
- SARSA typically converges to a safer path.

Why this matters for robotics:
Real robots explore imperfectly (noise, drift, control error). SARSA-like on-policy updates can be more conservative and robust when safety matters.

---

## How to run

python qlearn_SARSA_comparison.py
python qlearn_SARSA_difference.py
