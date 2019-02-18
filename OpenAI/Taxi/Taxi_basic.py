# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import gym

env = gym.make('Taxi-v2').env

#Reset the Environment to random state
env.reset()
env.render()

#Action space = 6 possible Actions
#south
#north
#east
#west
#pickup
#dropoff
print("Action Space {}".format(env.action_space))

#State space = 500 States
# 4 pickup locations, 5x5 grid, 1additional passenger (x5)
print("State Space {}".format(env.observation_space))


# =============================================================================
# #Encode a state
# =============================================================================
state = env.encode(3, 1, 2, 0) # (taxi row, taxi column, passenger index, destination index)
print("State:", state)

env.s = state
env.render()

# =============================================================================
# reward Table
# =============================================================================
#{action: [(probability, nextstate, reward, done)]}
# 0-5 corresponds to the actions (south, north, east, west, pickup, dropoff) 
# Probability is always 1.0
# nextstate is state of action at this index of the dict
# movements have a -1 reward 
# pickup/dropoff have -10, at right location +20 reward
env.P[328]


# =============================================================================
# Simulatino with random actions (no RL)
# =============================================================================
env.s = 328  # set environment to illustration's state

epochs = 0
penalties, reward = 0, 0

frames = [] # for animation

done = False

while not done:
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)

    if reward == -10:
        penalties += 1
    
    # Put each rendered frame into dict for animation
    frames.append({
        'frame': env.render(mode='ansi'),
        'state': state,
        'action': action,
        'reward': reward
        }
    )

    epochs += 1
    
    
print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}".format(penalties))

#Simulating epochs to pick up passenger at Y and get to R
from IPython.display import clear_output
from time import sleep

def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'].getvalue())
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)
        
print_frames(frames)

# =============================================================================
# Implementing Q-learning
# =============================================================================
#Initializing the Q-table to a 500x6 Matrix of 0
import numpy as np
q_table = np.zeros([env.observation_space.n, env.action_space.n])

#print(q_table)


# Training the agent
import random
from IPython.display import clear_output

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# Plotting metrics
all_epochs = []
all_penalties = []
frames = [] # for animation
total_reward_list = []
avg_reward = [0]
total_reward, avg_reward_tot, avg_reward_episodes, total_epochs, episodes = 0,0,0,0,0

for i in range(1, 100001):
    state = env.reset()
    
    # Setting initial values
    epochs, penalties, reward = 0,0,0
    done = False

    # Decide wether to pick a random action or exploit the
    # already computed Q-values (comparing to epsilon)
    while not done:
        if random.uniform(0,1) < epsilon:
            # Explore the action space (Number between 1 and 6)
            action = env.action_space.sample()
        else:
            # Exploit learned values (Maximum in Q_table
            # of current state)
            action = np.argmax(q_table[state]) 
        
        #Execute the chosen action to obtain next state and reward
        next_state, reward, done, info = env.step(action)
        
        # Update the Q Value to the new Q Value (of next state)
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        # Q-learning equation:
        # (1-alpha) * Q(state, action) + alpha(reward_t+1 + gamma * maxQ(state_t+1, action))
        new_value = (1-alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action]= new_value
        
        # Update Penalties and epochs
        if reward == -10:
            penalties +=1
        

        # Put each rendered frame into dict for animation
        frames.append({
            'frame': env.render(mode='ansi'),
            'state': state,
            'action': action,
            'reward': reward,
            'avg_reward_tot': avg_reward_tot,
            'avg_reward': avg_reward_episodes
            }
        )
        
        state = next_state
        epochs +=1
        total_epochs+=1
        total_reward+=reward
        total_reward_list.append(reward)
        avg_reward_tot = total_reward/total_epochs
        avg_reward_episodes = avg_reward[episodes]

    


    # Clear output after 100 episodes
    if i%100 ==0:
        clear_output(wait=True)
        print(f"Episode: {i}")
        episodes +=1
        avg_reward.append(sum(total_reward_list)/len(total_reward_list))
        total_reward_list = []
        
        
 
       
def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'].getvalue())
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        print(f"Avg. reward (100 episodes): {frame['avg_reward']}")
        print(f"Avg. total reward: {frame['avg_reward_tot']}")
        sleep(.1)
        
print_frames(frames[len(frames)-1000:len(frames)])
print("Training finished.\n")


# Q value at the illustrations state
state = env.encode(3, 1, 2, 0) # (taxi row, taxi column, passenger index, destination index)
print("State:", state)
env.s = state
env.render()
env.P[328]
#Action space = 6 possible Actions
#south
#north
#east
#west
#pickup
#dropoff
q_table[328]


# =============================================================================
# Evaluating the Agents performance after training
# =============================================================================
#Reset epochs and penalties
total_epochs, total_penalties, total_reward = 0,0,0
episodes = 100

for _ in range(episodes):
    # Reset
    state= env.reset()
    epochs, penalties, reward= 0,0,0

    done = False
    
    while not done:
        # Take action according to Q Table
        action = np.argmax(q_table[state])
        # Execute the chosen action to obtain next state and reward
        state, reward, done, info = env.step(action)

        # Update Penalties and epochs
        if reward == -10:
            penalties += 1
        
        epochs +=1

    total_penalties += penalties
    total_epochs += epochs
    total_reward += reward

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")
print(f"Total Reward: {total_reward}")





























