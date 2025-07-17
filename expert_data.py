import gymnasium as gym
import numpy as np

ENV_ID = "Humanoid-v4"
EPISODES = 10
STEPS_PER_EPISODE = 1000

# This uses a random policy as a placeholder for expert data
def generate_expert_data():
    env = gym.make(ENV_ID)
    observations = []
    actions = []
    for ep in range(EPISODES):
        obs, _ = env.reset()
        for _ in range(STEPS_PER_EPISODE):
            action = env.action_space.sample()
            observations.append(obs)
            actions.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    env.close()
    np.savez('expert_data_humanoid.npz', observations=np.array(observations), actions=np.array(actions))
    print(f"Saved expert data with {len(observations)} samples.")

if __name__ == "__main__":
    generate_expert_data() 