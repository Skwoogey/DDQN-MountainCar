import numpy as np
import random
from IPython.display import clear_output
from collections import deque

import gym

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape, Flatten
from tensorflow.keras.optimizers import Adam


enviroment = gym.make("MountainCar-v0")
#enviroment.render()

print(enviroment.action_space)

#print('Number of states: {}'.format(enviroment.observation_space.n))
#print('Number of actions: {}'.format(enviroment.action_space.n))    

class Agent:
    def __init__(self, enviroment, optimizer):
        
        # Initialize atributes
        self._state_size = enviroment.observation_space.shape[0]
        self._action_size = enviroment.action_space.n
        self._optimizer = optimizer
        self.maxlen = 10000
        self.batch_size = 128
        
        self.expirience_replay = deque(maxlen=self.maxlen)
        self.states = np.ndarray((self.batch_size, 1, self._state_size))
        self.next_states = np.ndarray((self.batch_size,1, self._state_size))
        self.target = np.ndarray((self.batch_size, 1, self._action_size))
        self.next_target = np.ndarray((self.batch_size, 1, self._action_size))
        
        # Initialize discount and exploration rate
        self.gamma = 0.99
        self.tau = 0.01
        self.epsilon_high = 0.5
        self.epsilon_low = 0.1
        self.decay_period = 100000.0
        self.epsilon = self.epsilon_high
        self.replay_size = 0
        
        # Build networks
        self.q_network = self._build_compile_model()
        self.target_network = self._build_compile_model()
        self.hard_update()

    def store(self, state, action, reward, next_state, terminated):
        self.expirience_replay.append((np.reshape(state, (1, self._state_size)), action, reward, np.reshape(next_state, (1, self._state_size)), terminated))
        self.replay_size = min(self.replay_size + 1, self.maxlen)
    
    def _build_compile_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(1, self._state_size)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self._action_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=self._optimizer)
        return model

    def soft_update(self):
        target_weights = np.array(self.target_network.get_weights())
        local_weights = np.array(self.q_network.get_weights())
        self.target_network.set_weights((1.0 - self.tau) * target_weights +local_weights * self.tau)

    def hard_update(self):
        self.target_network.set_weights(self.q_network.get_weights())
    
    def act(self, state, t = 0.0):
        self.epsilon = self.epsilon_high - min(1.0, t / self.decay_period)*(self.epsilon_high - self.epsilon_low)
        if np.random.rand() <= self.epsilon:
            return enviroment.action_space.sample()
        
        q_values = self.q_network.predict(np.reshape(state, (1, 1, self._state_size)))
        return np.argmax(q_values[0])

    def act_only(self, state):
        q_values = self.target_network.predict(np.reshape(state, (1, 1, self._state_size)))
        return np.argmax(q_values[0])

    def retrain(self, use_last=False):
        minibatch = None
        if use_last:
            minibatch = [self.expirience_replay[x] for x in range(-self.batch_size, self.replay_size)]
        else:
            minibatch = random.sample(self.expirience_replay, self.batch_size)
        
        
        for i in range(self.batch_size):
            self.states[i] = minibatch[i][0]
            self.next_states[i] = minibatch[i][3]
            
        self.target = self.q_network.predict(self.states)
        self.next_target = self.target_network.predict(self.next_states)

        for i in range(self.batch_size):
            if minibatch[i][4]:
                self.target[i][minibatch[i][1]] = minibatch[i][2]
            else:
                self.target[i][minibatch[i][1]] = minibatch[i][2] + self.gamma * np.amax(self.next_target[i, 1])
            
        self.q_network.train_on_batch(self.states, self.target)

    def __len__(self):
        return self.replay_size

optimizer = Adam(learning_rate=0.01)
agent = Agent(enviroment, optimizer)

def log(*objects):
    print(*objects, file=log.file)
    print(*objects)
log.file = open("log.txt", "w")

def speed_reward(s, ns):
    return 10 * abs(ns[1])

def accel_reward(s, ns):
    return 300 * (abs(ns[1]) - abs(s[1]))

def none_reward(s, ns):
    return 0

reward_shaping_func = none_reward

num_of_episodes = 25000
agent.q_network.summary()
state = np.ndarray((enviroment.observation_space.shape[0], ))
next_state = np.ndarray((enviroment.observation_space.shape[0], ))
total_steps = 0
showcase_freq = 5
won_episodes = 0

for e in range(0, num_of_episodes):
    # Reset the enviroment
    state= enviroment.reset()
    
    # Initialize variables
    total_reward = 0
    total_modified_reward = 0
    log("episode:", e)
    timestep = 0
    while 1:
        #print(timestep)
        # Run Action
        action = agent.act(state, total_steps)
        
        # Take action    
        next_state, reward, terminated, info = enviroment.step(action) 

        modified_reward = reward + reward_shaping_func(state, next_state)
        
        #print(next_state.shape)
        agent.store(state, action, modified_reward, next_state, terminated)
        
        state = next_state
            
        if len(agent) > agent.batch_size:
            agent.retrain()
            #agent.soft_update()

        if (total_steps + 1) % 1000 == 0:
            agent.hard_update()
            pass

        timestep += 1
        total_steps += 1
        total_reward += reward
        total_modified_reward += modified_reward

        if terminated:
            break

    log("Total reward: ", total_reward, total_modified_reward)

    if (e + 1) % showcase_freq == 0 and True:
        agent.target_network.save_weights("models\\car_" + str(e + 1) + ".h5")
        log("Showcase")
        state = enviroment.reset()
        enviroment.render()
        input_raw()
        terminated = False
        total_reward = 0
        total_modified_reward = 0
        while 1:
            action = agent.act_only(state)
            next_state, reward, terminated, info = enviroment.step(action) 
            modified_reward = reward + reward_shaping_func(state, next_state)
            total_modified_reward += modified_reward
            total_reward += reward
            state = next_state
            #enviroment.render()
            if terminated:
                break

        log("showcase reward: ", total_reward, total_modified_reward)

log.file.close()