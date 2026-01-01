import os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import matplotlib.pyplot as plt

def epsilon_linear(t, eps_start, eps_end, eps_decay_steps):
    """Linear epsilon schedule."""
    frac = min(t / eps_decay_steps, 1.0)
    return eps_start + frac * (eps_end - eps_start)


def choose_action_eps_greedy(Q, s, eps, action_space):
    """Epsilon-greedy action selection."""
    if np.random.rand() < eps:
        return action_space.sample()
    return int(np.argmax(Q[s]))


def moving_average(x, window=200):
    """Return moving average of `x` with given window."""
    x = np.asarray(x, dtype=np.float32)
    if len(x) < window:
        return x
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(x, kernel, mode="valid")


def plot_curves(returns_sarsa, returns_qlearn, falls_sarsa, falls_qlearn, window=200, out_path="cliff_curves.png"):
    """Save return learning curve (moving average)."""
    ma_s = moving_average(returns_sarsa, window=window)
    ma_q = moving_average(returns_qlearn, window=window)

    plt.figure(figsize=(10, 6))
    plt.plot(ma_s, label=f"SARSA return (MA {window})", linewidth=2)
    plt.plot(ma_q, label=f"Q-learn return (MA {window})", linewidth=2)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Return", fontsize=12)
    plt.title("CliffWalking-v1: Learning Curves", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# Video recording uses Gym's renderer and the RecordVideo wrapper.


# Training routines (keep epsilon nonzero for CliffWalking)
def train_sarsa(
    env_id="CliffWalking-v1",
    episodes=100_000,
    alpha=0.5,
    gamma=0.99,
    eps_start=1.0,      # full exploration initially
    eps_end=0.05,       # small epsilon in final phase
    eps_decay_steps=80_000,
    seed=0,
):
    env = gym.make(env_id)
    env.reset(seed=seed)
    np.random.seed(seed)

    nS = env.observation_space.n
    nA = env.action_space.n
    Q = np.zeros((nS, nA), dtype=np.float32)

    returns = []
    fell = []

    for ep in range(episodes):
        eps = epsilon_linear(ep, eps_start, eps_end, eps_decay_steps)

        s, _ = env.reset(seed=seed + ep)
        a = choose_action_eps_greedy(Q, s, eps, env.action_space)

        done = False
        total = 0.0
        fell_off = 0

        while not done:
            s2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            total += r

            # CliffWalking gives -100 on falling into the cliff
            if r <= -100:
                fell_off = 1

            if done:
                target = r
                Q[s, a] += alpha * (target - Q[s, a])
            else:
                a2 = choose_action_eps_greedy(Q, s2, eps, env.action_space)
                target = r + gamma * Q[s2, a2]
                Q[s, a] += alpha * (target - Q[s, a])
                s, a = s2, a2

        returns.append(total)
        fell.append(fell_off)

        if (ep + 1) % 5000 == 0:
            print(f"[SARSA] ep={ep+1} eps={eps:.2f} avgRet(last200)={np.mean(returns[-200:]):.2f} fallRate(last200)={np.mean(fell[-200:]):.2f}")

    env.close()
    return Q, np.array(returns, dtype=np.float32), np.array(fell, dtype=np.float32)


def train_q_learning(
    env_id="CliffWalking-v1",
    episodes=100_000,
    alpha=0.5,
    gamma=0.99,
    eps_start=1.0,      # full exploration initially
    eps_end=0.05,       # small epsilon in final phase
    eps_decay_steps=80_000,
    seed=0,
):
    env = gym.make(env_id)
    env.reset(seed=seed)
    np.random.seed(seed)

    nS = env.observation_space.n
    nA = env.action_space.n
    Q = np.zeros((nS, nA), dtype=np.float32)

    returns = []
    fell = []

    for ep in range(episodes):
        eps = epsilon_linear(ep, eps_start, eps_end, eps_decay_steps)

        s, _ = env.reset(seed=seed + ep)
        done = False
        total = 0.0
        fell_off = 0

        while not done:
            a = choose_action_eps_greedy(Q, s, eps, env.action_space)
            s2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            total += r

            if r <= -100:
                fell_off = 1

            best_next = 0.0 if done else float(np.max(Q[s2]))
            target = r + gamma * best_next
            Q[s, a] += alpha * (target - Q[s, a])

            s = s2

        returns.append(total)
        fell.append(fell_off)

        if (ep + 1) % 5000 == 0:
            print(f"[Q]     ep={ep+1} eps={eps:.2f} avgRet(last200)={np.mean(returns[-200:]):.2f} fallRate(last200)={np.mean(fell[-200:]):.2f}")

    env.close()
    return Q, np.array(returns, dtype=np.float32), np.array(fell, dtype=np.float32)


# Video demo: run ε-greedy rollouts (not fully greedy)
def record_video(env_id, Q, out_path, epsilon=0.15, num_episodes=5, seed=0, max_steps=200):
    """Record episodes using Gym's RecordVideo wrapper (uses env renderer).

    Saves video files into the directory of `out_path` using the basename
    of `out_path` as a prefix.
    """
    video_dir = os.path.dirname(out_path) or "."
    os.makedirs(video_dir, exist_ok=True)

    base = gym.make(env_id, render_mode="rgb_array")
    recorder = RecordVideo(
        base,
        video_folder=video_dir,
        name_prefix=os.path.splitext(os.path.basename(out_path))[0],
        episode_trigger=lambda ep: True,
        disable_logger=True,
    )

    try:
        for ep in range(num_episodes):
            obs, _ = recorder.reset(seed=seed + ep)
            done = False
            steps = 0
            total = 0.0

            while not done and steps < max_steps:
                a = choose_action_eps_greedy(Q, obs, epsilon, recorder.action_space)
                obs, r, terminated, truncated, _ = recorder.step(a)
                done = terminated or truncated
                total += r
                steps += 1

            print(f"  Episode {ep+1}: return={total:.1f}, steps={steps}")
    finally:
        recorder.close()

    print(f"Saved video(s) to: {video_dir} (prefix={os.path.splitext(os.path.basename(out_path))[0]})")


if __name__ == "__main__":
    env_id = "CliffWalking-v1"

    print("Training SARSA...")
    Qs, rs, fs = train_sarsa(env_id=env_id, seed=0)

    print("\nTraining Q-learning...")
    Qq, rq, fq = train_q_learning(env_id=env_id, seed=0)

    print("\nSaving plots...")
    plot_curves(rs, rq, fs, fq, window=500, out_path="cliff_curves.png")
    print("Saved: cliff_curves.png")

    print("\nRecording videos (final greedy policies)...")
    record_video(env_id, Qs, "videos_cliff/sarsa_greedy.mp4", epsilon=0.0, num_episodes=3, seed=42)
    record_video(env_id, Qq, "videos_cliff/qlearn_greedy.mp4", epsilon=0.0, num_episodes=3, seed=42)

    print("\nFinal stats (last500 episodes):")
    print("SARSA avg return:", float(rs[-500:].mean()), "fall rate:", float(fs[-500:].mean()))
    print("Q     avg return:", float(rq[-500:].mean()), "fall rate:", float(fq[-500:].mean()))
