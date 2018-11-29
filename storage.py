import numpy as np
import random

from collections import deque

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
            
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)

class ReservoirBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action):
        state = np.expand_dims(state, 0)
        self.buffer.append((state, action))
    
    def sample(self, batch_size):
        reservoir = []
        for t, item in enumerate(self.buffer):
            if t < batch_size:
                reservoir.append(item)
            else:
                m = random.randint(0,t)
                if m < batch_size:
                    reservoir[m] = item
        state, action = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action
    
    def __len__(self):
        return len(self.buffer)
    