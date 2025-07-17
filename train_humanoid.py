import gymnasium as gym
import numpy as np
import tensorflow as tf
from actor_impl_tf import Actor

ENV_ID = "Humanoid-v4"
SEED = 42
EPISODES = 200
STEPS_PER_EPISODE = 1000
LEARNING_RATE = 3e-4
BC_EPOCHS = 10
PPO_EPOCHS = 5
BATCH_SIZE = 64
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
SAVE_EVERY = 20

# Load expert data for imitation learning
expert = np.load('expert_data_humanoid.npz')
expert_obs = expert['observations']
expert_act = expert['actions']

# Environment setup
env = gym.make(ENV_ID, render_mode="human")
obs_dim = np.prod(env.observation_space.shape)
act_dim = np.prod(env.action_space.shape)
act_low = env.action_space.low
act_high = env.action_space.high

actor = Actor(obs_dim, act_dim, act_low, act_high)
actor.build((None, obs_dim))
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# Behavior Cloning (Imitation Learning)
def behavior_cloning():
    print("Starting Behavior Cloning...")
    dataset = tf.data.Dataset.from_tensor_slices((expert_obs, expert_act)).shuffle(10000).batch(BATCH_SIZE)
    for epoch in range(BC_EPOCHS):
        losses = []
        for obs_batch, act_batch in dataset:
            with tf.GradientTape() as tape:
                mean, _ = actor(obs_batch)
                loss = tf.reduce_mean(tf.square(mean - act_batch))
            grads = tape.gradient(loss, actor.trainable_variables)
            optimizer.apply_gradients(zip(grads, actor.trainable_variables))
            losses.append(loss.numpy())
        print(f"BC Epoch {epoch+1}/{BC_EPOCHS}, Loss: {np.mean(losses):.4f}")
    print("Behavior Cloning finished.")

# PPO Helper functions
def discount_cumsum(x, discount):
    y = np.zeros_like(x)
    running = 0
    for t in reversed(range(len(x))):
        running = x[t] + discount * running
        y[t] = running
    return y

def ppo_update(obs_buf, act_buf, adv_buf, ret_buf, old_logp_buf):
    dataset = tf.data.Dataset.from_tensor_slices((obs_buf, act_buf, adv_buf, ret_buf, old_logp_buf)).shuffle(10000).batch(BATCH_SIZE)
    for _ in range(PPO_EPOCHS):
        for obs_b, act_b, adv_b, ret_b, old_logp_b in dataset:
            with tf.GradientTape() as tape:
                mean, log_std = actor(obs_b)
                std = tf.exp(log_std)
                dist = tfp.distributions.Normal(mean, std)
                logp = tf.reduce_sum(dist.log_prob(act_b), axis=-1, keepdims=True)
                ratio = tf.exp(logp - old_logp_b)
                clipped = tf.clip_by_value(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
                policy_loss = -tf.reduce_mean(tf.minimum(ratio * adv_b, clipped * adv_b))
                loss = policy_loss
            grads = tape.gradient(loss, actor.trainable_variables)
            optimizer.apply_gradients(zip(grads, actor.trainable_variables))

# Main training loop
def train():
    import tensorflow_probability as tfp
    behavior_cloning()
    for episode in range(EPISODES):
        obs, _ = env.reset(seed=SEED + episode)
        obs_buf, act_buf, rew_buf, logp_buf = [], [], [], []
        done = False
        total_reward = 0
        for step in range(STEPS_PER_EPISODE):
            obs_input = tf.convert_to_tensor(obs[None, :], dtype=tf.float32)
            mean, log_std = actor(obs_input)
            std = tf.exp(log_std)
            dist = tfp.distributions.Normal(mean, std)
            action = dist.sample()[0].numpy()
            logp = tf.reduce_sum(dist.log_prob(action)).numpy()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            obs_buf.append(obs)
            act_buf.append(action)
            rew_buf.append(reward)
            logp_buf.append(logp)
            obs = next_obs
            total_reward += reward
            env.render()
            if terminated or truncated:
                break
        # Compute returns and advantages
        rewards = np.array(rew_buf)
        values = np.zeros_like(rewards)  # No critic, so zeros
        returns = discount_cumsum(rewards, GAMMA)
        advantages = returns - values
        # PPO update
        obs_arr = np.array(obs_buf, dtype=np.float32)
        act_arr = np.array(act_buf, dtype=np.float32)
        adv_arr = np.array(advantages, dtype=np.float32)[:, None]
        ret_arr = np.array(returns, dtype=np.float32)[:, None]
        logp_arr = np.array(logp_buf, dtype=np.float32)[:, None]
        ppo_update(obs_arr, act_arr, adv_arr, ret_arr, logp_arr)
        print(f"Episode {episode+1}: Reward = {total_reward:.2f}")
        if (episode + 1) % SAVE_EVERY == 0:
            actor.save_weights(f"actor_humanoid_ep{episode+1}.weights.h5")
            print(f"Saved weights at episode {episode+1}")
    env.close()
    actor.save_weights("actor_humanoid_final.weights.h5")
    print("Training finished. Final weights saved.")

if __name__ == "__main__":
    train() 