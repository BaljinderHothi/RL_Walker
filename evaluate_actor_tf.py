import gymnasium as gym
import numpy as np
import tensorflow as tf
from actor_impl_tf import Actor


def evaluate(weights_path, env_id="Humanoid-v4", episodes=5, render=True):
    env = gym.make(env_id, render_mode="human" if render else None)
    obs_dim = np.prod(env.observation_space.shape)
    act_dim = np.prod(env.action_space.shape)
    act_low = env.action_space.low
    act_high = env.action_space.high

    actor = Actor(obs_dim, act_dim, act_low, act_high)
    actor.build((None, obs_dim))
    actor.load_weights(weights_path)

    episode_rewards = []
    episode_lengths = []
    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        while not done:
            obs_input = tf.convert_to_tensor(obs[None, :], dtype=tf.float32)
            action = actor.get_action(obs_input)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            if render:
                env.render()
            done = terminated or truncated
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"Episode {ep+1}: Reward = {total_reward:.2f}, Length = {steps}")
    env.close()
    print("\nSummary:")
    print(f"Mean reward: {np.mean(episode_rewards):.2f}")
    print(f"Std reward: {np.std(episode_rewards):.2f}")
    print(f"Min/Max reward: {np.min(episode_rewards):.2f}/{np.max(episode_rewards):.2f}")
    print(f"Mean episode length: {np.mean(episode_lengths):.2f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--env-id", type=str, default="Humanoid-v4")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    evaluate(args.weights, args.env_id, args.episodes, not args.no_render) 