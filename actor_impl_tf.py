import tensorflow as tf
import numpy as np

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, act_low, act_high):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.fc2 = tf.keras.layers.Dense(256, activation='relu')
        self.fc_mean = tf.keras.layers.Dense(act_dim)
        self.fc_logstd = tf.keras.layers.Dense(act_dim)
        self.action_scale = tf.constant((act_high - act_low) / 2.0, dtype=tf.float32)
        self.action_bias = tf.constant((act_high + act_low) / 2.0, dtype=tf.float32)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = tf.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = tf.exp(log_std)
        normal = tf.random.normal(tf.shape(mean))
        x_t = mean + std * normal
        y_t = tf.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        return action.numpy()[0] 