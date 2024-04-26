import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from collections import deque

# Define Deep Q-Network (DQN) model
class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        return np.argmax(self.model.predict(state)[0])

    def replay(self, batch_size):
        minibatch = np.random.choice(len(self.memory), batch_size, replace=False)
        for idx in minibatch:
            state, action, reward, next_state, done = self.memory[idx]
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Define smart home environment
class SmartHomeEnv:
    def __init__(self):
        self.state_size = 4  # Example: temperature, occupancy, time of day, energy price
        self.action_size = 2  # Example: turn HVAC on/off, turn lights on/off
        self.reward_range = (-10, 10)  # Example: penalize high energy usage, reward energy savings

    def reset(self):
        return np.random.random(self.state_size)

    def step(self, action):
        # Example: Simulate energy usage, update state based on action
        next_state = np.random.random(self.state_size)
        reward = np.random.uniform(*self.reward_range)
        done = False
        return next_state, reward, done, {}

# Initialize environment and agent
env = SmartHomeEnv()
agent = DQN(env.state_size, env.action_size)

# Train agent
for _ in range(1000):
    state = env.reset()
    state = np.reshape(state, [1, env.state_size])
    for _ in range(100):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, env.state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
    if len(agent.memory) > 32:
        agent.replay(32)
