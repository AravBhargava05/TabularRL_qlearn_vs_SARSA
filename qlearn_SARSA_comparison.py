import time
import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo

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
    """Moving average (windowed)."""
    x = np.asarray(x, dtype=np.float32)
    if len(x) < window:
        return x
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(x, kernel, mode="valid")

def plot_learning_curves(returns_sarsa, returns_qlearn, window=200, out_path="learning_curve_taxi.png"):
    """Save learning-curve plot to file."""
    ma_s = moving_average(returns_sarsa, window=window)
    ma_q = moving_average(returns_qlearn, window=window)

    plt.figure()
    plt.plot(ma_s, label=f"SARSA (moving avg {window})")
    plt.plot(ma_q, label=f"Q-learning (moving avg {window})")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Taxi-v3 Learning Curves (Often Similar)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def train_sarsa(
    env_id="Taxi-v3",
    episodes=50_000,
    alpha=0.1,
    gamma=0.99,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_steps=40_000,
    seed=0,
):
    """Train tabular SARSA on `env_id`. Returns Q and returns array."""
    env = gym.make(env_id)
    env.reset(seed=seed)
    np.random.seed(seed)

    nS = env.observation_space.n
    nA = env.action_space.n
    Q = np.zeros((nS, nA), dtype=np.float32)

    returns = []
    for ep in range(episodes):
        eps = epsilon_linear(ep, eps_start, eps_end, eps_decay_steps)

        s, _ = env.reset(seed=seed + ep)
        a = choose_action_eps_greedy(Q, s, eps, env.action_space)

        done = False
        total = 0.0

        while not done:
            s2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            total += r

            if done:
                target = r
                Q[s, a] += alpha * (target - Q[s, a])
            else:
                a2 = choose_action_eps_greedy(Q, s2, eps, env.action_space)
                target = r + gamma * Q[s2, a2]
                Q[s, a] += alpha * (target - Q[s, a])
                s, a = s2, a2

        returns.append(total)

        if (ep + 1) % 5000 == 0:
            print(f"[SARSA] ep={ep+1} eps={eps:.3f} avg(last200)={np.mean(returns[-200:]):.2f}")

    env.close()
    return Q, np.array(returns, dtype=np.float32)


def train_q_learning(
    env_id="Taxi-v3",
    episodes=50_000,
    alpha=0.1,
    gamma=0.99,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_steps=40_000,
    seed=0,
):
    """Train tabular Q-learning on `env_id`. Returns Q and returns array."""
    env = gym.make(env_id)
    env.reset(seed=seed)
    np.random.seed(seed)

    nS = env.observation_space.n
    nA = env.action_space.n
    Q = np.zeros((nS, nA), dtype=np.float32)

    returns = []
    for ep in range(episodes):
        eps = epsilon_linear(ep, eps_start, eps_end, eps_decay_steps)

        s, _ = env.reset(seed=seed + ep)
        done = False
        total = 0.0

        while not done:
            a = choose_action_eps_greedy(Q, s, eps, env.action_space)
            s2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            total += r

            best_next = 0.0 if done else float(np.max(Q[s2]))
            target = r + gamma * best_next
            Q[s, a] += alpha * (target - Q[s, a])

            s = s2

        returns.append(total)

        if (ep + 1) % 5000 == 0:
            print(f"[Q]     ep={ep+1} eps={eps:.3f} avg(last200)={np.mean(returns[-200:]):.2f}")

    env.close()
    return Q, np.array(returns, dtype=np.float32)

def record_video(env_id, Q, out_path, num_episodes=3, seed=0, max_steps=200, fps=12):
    """Record a few episodes to MP4 using Gym's RecordVideo wrapper.

    This uses the environment's renderer and lets Gym/ffmpeg handle encoding.
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
                a = int(np.argmax(Q[obs]))
                obs, r, terminated, truncated, _ = recorder.step(a)
                done = terminated or truncated
                total += r
                steps += 1

            print(f"  Episode {ep+1}: return={total:.1f}, steps={steps}")
    finally:
        recorder.close()

    print(f"Saved video(s) to: {video_dir} (prefix={os.path.splitext(os.path.basename(out_path))[0]})")

def play_once(env_id, Q, seed=0, delay=0.10, max_steps=200):
    """Play one episode with greedy policy."""
    env = gym.make(env_id, render_mode="human")
    try:
        s, _ = env.reset(seed=seed)
        done = False
        total = 0.0
        steps = 0
        while not done and steps < max_steps:
            a = int(np.argmax(Q[s]))
            s, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            total += r
            steps += 1
            time.sleep(delay)
        print(f"rollout return={total:.1f} steps={steps}")
    finally:
        env.close()

if __name__ == "__main__":
    episodes = 50_000
    eps_decay_steps = int(0.8 * episodes)

    print("Training SARSA...")
    Qs, rs = train_sarsa(episodes=episodes, eps_decay_steps=eps_decay_steps, seed=0)

    print("\nTraining Q-learning...")
    Qq, rq = train_q_learning(episodes=episodes, eps_decay_steps=eps_decay_steps, seed=0)

    print("\nSaving learning curve plot...")
    plot_learning_curves(rs, rq, window=200, out_path="learning_curve_taxi.png")
    print("Saved: learning_curve_taxi.png")

    print("\nRecording videos...")
    record_video("Taxi-v3", Qs, "videos/sarsa_trained.mp4", num_episodes=3, seed=42)
    record_video("Taxi-v3", Qq, "videos/qlearning_trained.mp4", num_episodes=3, seed=42)

    print("\nFinal avg returns (last200):")
    print("SARSA:", float(rs[-200:].mean()))
    print("Q:    ", float(rq[-200:].mean()))
