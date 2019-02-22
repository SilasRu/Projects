# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 10:23:59 2019

@author: SR
"""
import gym
import numpy as np 
import itertools
import os
import pandas as pd
import random
import numpy as np
import pickle
import time
import numpy as np
import argparse
import re

from gym import spaces
from gym.utils import seeding
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque


def get_data(col='close'):
  # Returns a 3 x n_step array
  msft = pd.read_csv('data/daily_MSFT.csv', usecols=[col])
  ibm = pd.read_csv('data/daily_IBM.csv', usecols=[col])
  qcom = pd.read_csv('data/daily_QCOM.csv', usecols=[col])
  # recent price are at top; reverse it
  return np.array([msft[col].values[::-1],
                   ibm[col].values[::-1],
                   qcom[col].values[::-1]])


def get_scaler(env):
  # Takes a env and returns a scaler for its observation space
  low = [0] * (env.n_stock * 2 + 1)

  high = []
  max_price = env.stock_price_history.max(axis=1)
  min_price = env.stock_price_history.min(axis=1)
  max_cash = env.init_invest * 3 # 3 is a magic number...
  max_stock_owned = max_cash // min_price
  for i in max_stock_owned:
    high.append(i)
  for i in max_price:
    high.append(i)
  high.append(max_cash)

  scaler = StandardScaler()
  scaler.fit([low, high])
  return scaler


def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
        
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import itertools


class TradingEnv(gym.Env):
    def __init__(self, train_data, init_invest=20000):
        # data
        self.stock_price_history = np.around(train_data) # round up to integer to reduce state space
        self.n_stock, self.n_step = self.stock_price_history.shape

        # instance attributes
        self.init_invest = init_invest
        self.cur_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None

        # action space
        self.action_space = spaces.Discrete(3**self.n_stock)

        # observation space: give estimates in order to sample and build scaler
        stock_max_price = self.stock_price_history.max(axis=1)
        stock_range = [[0, init_invest * 2 // mx] for mx in stock_max_price]
        price_range = [[0, mx] for mx in stock_max_price]
        cash_in_hand_range = [[0, init_invest * 2]]
        self.observation_space = spaces.MultiDiscrete(stock_range + price_range + cash_in_hand_range)

        # seed and start
        self._seed()
        self._reset()


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def _reset(self):
        self.cur_step = 0
        self.stock_owned = [0] * self.n_stock
        self.stock_price = self.stock_price_history[:, self.cur_step]
        self.cash_in_hand = self.init_invest
        return self._get_obs()


    def _step(self, action):
        assert self.action_space.contains(action)
        prev_val = self._get_val()
        self.cur_step += 1
        self.stock_price = self.stock_price_history[:, self.cur_step] # update price
        self._trade(action)
        cur_val = self._get_val()
        reward = cur_val - prev_val
        done = self.cur_step == self.n_step - 1
        info = {'cur_val': cur_val}
        return self._get_obs(), reward, done, info


    def _get_obs(self):
        obs = []
        obs.extend(self.stock_owned)
        obs.extend(list(self.stock_price))
        obs.append(self.cash_in_hand)
        return obs


    def _get_val(self):
        return np.sum(self.stock_owned * self.stock_price) + self.cash_in_hand


    def _trade(self, action):
        # all combo to sell(0), hold(1), or buy(2) stocks
        action_combo = map(list, itertools.product([0, 1, 2], repeat=self.n_stock))
        action_vec = action_combo[action]

        # one pass to get sell/buy index
        sell_index = []
        buy_index = []
        for i, a in enumerate(action_vec):
            if a == 0:
                sell_index.append(i)
            elif a == 2:
                buy_index.append(i)

        # two passes: sell first, then buy; might be naive in real-world settings
        if sell_index:
            for i in sell_index:
                self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
                self.stock_owned[i] = 0
        if buy_index:
            can_buy = True
            while can_buy:
                for i in buy_index:
                    if self.cash_in_hand > self.stock_price[i]:
                        self.stock_owned[i] += 1 # buy one share
                        self.cash_in_hand -= self.stock_price[i]
                else:
                    can_buy = False
                    
                    
                    
def mlp(n_obs, n_action, n_hidden_layer=1, n_neuron_per_layer=32,activation='relu', loss='mse'):
    model = Sequential()
    model.add(Dense(n_neuron_per_layer, input_dim=n_obs, activation=activation))
    
    for _ in range(n_hidden_layer):
        model.add(Dense(n_neuron_per_layer, activation=activation))
        model.add(Dense(n_action, activation='linear'))
        model.compile(loss=loss, optimizer=Adam())
    
    print(model.summary())
    return model


class DQNAgent(object):
  # A simple Deep Q agent
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = mlp(state_size, action_size)


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action


    def replay(self, batch_size=32):
    # vectorized implementation; 30x speed up compared with for loop
        minibatch = random.sample(self.memory, batch_size)

        states = np.array([tup[0][0] for tup in minibatch])
        actions = np.array([tup[1] for tup in minibatch])
        rewards = np.array([tup[2] for tup in minibatch])
        next_states = np.array([tup[3][0] for tup in minibatch])
        done = np.array([tup[4] for tup in minibatch])

        # Q(s', a)
        target = rewards + self.gamma * np.amax(self.model.predict(next_states), axis=1)
        # end state target is reward itself (no lookahead)
        target[done] = rewards[done]

        # Q(s, a)
        target_f = self.model.predict(states)
        # make the agent to approximately map the current state to future discounted reward
        target_f[range(batch_size), actions] = target

        self.model.fit(states, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def load(self, name):
        self.model.load_weights(name)


    def save(self, name):
        self.model.save_weights(name)
        
        
if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-e', '--episode', type=int, default=2000,
#                       help='number of episode to run')
#     parser.add_argument('-b', '--batch_size', type=int, default=32,
#                       help='batch size for experience replay')
#     parser.add_argument('-i', '--initial_invest', type=int, default=20000,
#                       help='initial investment amount')
#     parser.add_argument('-m', '--mode', type=str, required=True,
#                       help='either "train" or "test"')
#     parser.add_argument('-w', '--weights', type=str, help='a trained model weights')
#     args = parser.parse_args()
    
    args_initial_invest = 20000
    args_episode = 2000
    args_batch_size = 32
    args_mode = 'train'
    args_weights = None
    
    maybe_make_dir('weights')
    maybe_make_dir('portfolio_val')

    timestamp = time.strftime('%Y%m%d%H%M')

    data = np.around(get_data())
    train_data = data[:, :3526]
    test_data = data[:, 3526:]

    env = TradingEnv(train_data, args_initial_invest)
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    scaler = get_scaler(env)

    portfolio_value = []

    if args_mode == 'test':
        # remake the env with test data
        env = TradingEnv(test_data, args_initial_invest)
        # load trained weights
        agent.load(args_weights)
        # when test, the timestamp is same as time when weights was trained
        timestamp = re.findall(r'\d{12}', args_weights)[0]

    for e in range(args_episode):
        state = env.reset()
        state = scaler.transform([state])
        for time in range(env.n_step):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            next_state = scaler.transform([next_state])
            if args_mode == 'train':
                agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, episode end value: {}".format(
                      e + 1, args_episode, info['cur_val']))
                portfolio_value.append(info['cur_val']) # append episode end portfolio value
                break
            if args_mode == 'train' and len(agent.memory) > args_batch_size:
                agent.replay(args_batch_size)
        if args_mode == 'train' and (e + 1) % 10 == 0:  # checkpoint weights
            agent.save('weights/{}-dqn.h5'.format(timestamp))

        # save portfolio value history to disk
        with open('portfolio_val/{}-{}.p'.format(timestamp, args_mode), 'wb') as fp:
            pickle.dump(portfolio_value, fp)

