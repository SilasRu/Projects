import pickle
import numpy as np
import re
import os
import sys
import time as ti
import progressbar

sys.path.append(os.path.dirname(os.path.__file__))

from data.utils import get_data, get_scaler, maybe_make_dir, save_data
from features.envs import TradingEnv
from features.agent import DQNAgent

class RunModel():
    def __init__(self):
        self.episode = None
        self.batch_size = 32
        self.initial_invest = None
        self.mode = None
        self.weights = None
        self.architecture = None

    def set_episodes(self, episode):
        self.episode = episode

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
    
    def set_initial_invest(self, initial_invest):
        self.initial_invest = initial_invest
    
    def set_mode(self, mode):
        self.mode = mode
    
    def set_weights(self, weights):
        self.weights = weights
    
    def set_architecture(self, architecture):
        self.architecture = architecture  

    def get_parameters(self):
        return (self.episode, self.batch_size, self.initial_invest, self.mode, self.weights, self.architecture)
    
    def run(self): 
        widgets=[
            ' [', progressbar.Timer(), '] ',
            progressbar.Bar(),
            ' (', progressbar.ETA(), ') ',]
                
        maybe_make_dir('weights')
        maybe_make_dir('portfolio_val')
        
        timestamp = ti.strftime('%Y%m%d%H%M')
        
        n_stock = 3 # number of stocks used (max. 3)
        data = np.around(get_data())
        train_len = int(data.shape[1]*0.70)
        train_data = data[0:n_stock, :train_len]
        test_data = data[0:n_stock, train_len:]
        
        save_data(train_data, "train_data.csv")
        save_data(test_data, "test_data.csv")
        
        env = TradingEnv(train_data, self.initial_invest)
        state_size = env.observation_space.shape
        action_size = env.action_space.n
        agent = DQNAgent(state_size, action_size, self.architecture)
        scaler = get_scaler(env)
        
        portfolio_metrics = [[], []]
        
        if self.mode == 'test':
            # remake the env with test data
            env = TradingEnv(test_data, self.initial_invest)
            # load trained weights
            agent.load(self.weights)
            # when test, the timestamp is same as time when weights was trained
            timestamp = re.findall(r'\d{12}', self.weights)[0]
        
        for e in range(self.episode):
            state = env.reset()
            state = scaler.transform([state])        
            for time in progressbar.progressbar(range(env.n_step), widgets=widgets):
                ti.sleep(0.001)
                if self.architecture=='lstm':
                    action = agent.act_lstm(state)
                elif self.architecture=='dense':
                    action = agent.act_dense(state)
                next_state, reward, done, info = env.step(action)
                next_state = scaler.transform([next_state])
                if self.mode == 'train':
                    agent.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    print("episode: {0}/{1}, portfolio value: {2}, sharpe ratio:{3:1.4f}".format(
                        e + 1, self.episode, info['cur_val'], env._get_sharpe()[0])),
                    portfolio_metrics[0].append(info['cur_val']) # append episode end portfolio value
                    portfolio_metrics[1].append(env._get_sharpe()[0]) # append episode end sharpe ratio
                    break
                if self.mode == 'train' and len(agent.memory) > self.batch_size:
                    if self.architecture=='lstm':
                        agent.replay_lstm(self.batch_size)
                    elif self.architecture=='dense':
                        agent.replay_dense(self.batch_size)                    
            if self.mode == 'train' and (e + 1) % 100 == 0:  # checkpoint weights
                agent.save('weights/{}-{}-{}.h5'.format(timestamp, e+1, self.architecture))
                save_data(env._get_sharpe()[1], "portfolio_val_train.csv")
            elif self.mode == 'test' and (e + 1) % 100 == 0:
                save_data(env._get_sharpe()[1], "portfolio_val_test.csv")
        
        # save portfolio metrics history to disk
        with open('portfolio_metrics/{}_{}_{}_{}.p'.format(timestamp, self.episode, self.mode, self.architecture), 'wb') as fp:
            pickle.dump(portfolio_metrics, fp)
        
        # save latest weights to disk 
        if self.mode == 'train':
            agent.save('weights/{}-{}-{}.h5'.format(timestamp, e+1, self.architecture))
        
        # save state- , portfolio-value- and sharpe-ratio-history of last episode to disk
        pickle.dump(env._get_sharpe()[1], open('portfolio_val/portfolio_val_{}_{}_{}_{}.p'.format(timestamp, e+1, self.mode, self.architecture), "wb" ))
        
    #     # save action visuals position to disk
    #     if args.mode == 'test':
    #         data = test_data
    #     else:
    #         data = train_data
    #     action_x = action_visual(env._get_sharpe()[1])
    #     action_y = action_pos(action_x, data)
        
    #     pickle.dump(action_x, open( "data/action_x_{}_{}_{}_{}.p".format(timestamp, e+1, args.mode, args.architecture), "wb" ) )
    #     pickle.dump(action_y, open( "data/action_y_{}_{}_{}_{}.p".format(timestamp, e+1, args.mode, args.architecture), "wb" ) )