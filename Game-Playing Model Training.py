import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Create the environment
env = gym.make('CartPole-v1')

# Define the Deep Q-Network model
model = Sequential([
    Dense(24, input_shape=(env.observation_space.shape[0],), activation='relu'),
    Dense(24, activation='relu'),
    Dense(env.action_space.n, activation='linear')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train the model using Q-learning
def train_dqn(num_episodes=1000, batch_size=32, gamma=0.99):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = np.argmax(model.predict(np.expand_dims(state, axis=0)))
            next_state, reward, done, _ = env.step(action)
            target = reward + gamma * np.max(model.predict(np.expand_dims(next_state, axis=0)))
            target_vec = model.predict(np.expand_dims(state, axis=0))[0]
            target_vec[action] = target
            model.fit(np.expand_dims(state, axis=0), target_vec.reshape(-1, env.action_space.n), epochs=1, verbose=0)
            state = next_state

train_dqn()
