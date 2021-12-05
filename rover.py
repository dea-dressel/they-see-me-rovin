'''
Mars Rover
Dea Dressel
Jose Nicholas Francisco
AA 228
'''
import sys
import os

import pandas as pd
import numpy as np
import random as rand
import matplotlib.pyplot as plt
import csv

from grid_generator import *

NUM_EPOCHS = 1000
HORIZON = 20
EPSILON_START = 0.9
BETA = 0.9
ALPHA = 0.1
DISCOUNT = 0.9


def unpack_csv(csv_filepath):

    # get number of minerals, holes, states and actions from csv file
    with open(csv_filepath, newline='') as file:
        reader = csv.reader(file)
        mineral_locations = next(reader)
        hole_locations = next(reader)
        num_states =int(next(reader)[0])
        num_actions = int(next(reader)[0])

    # extract grid world data
    grid_data = pd.read_csv(csv_filepath, skiprows=4)
    grid_data = grid_data.to_numpy()

    return list(map(int, mineral_locations)), list(map(int, hole_locations)), num_states,\
            num_actions, grid_data

        
def get_grid(minerals, holes):
    index = 1
    result = []
    while index <= NUM_ROWS*NUM_COLS:
        row = []
        for i in range(NUM_COLS):
            if index in minerals:
                row.append(R_MINERAL)
            elif index in holes:
                row.append(R_HOLE)
            else:
                row.append(R_CLEAR)
            index += 1
        result.append(row)
    return np.array(result)


def reinforcement_learning(num_states, num_actions, grid, type='baseline'):

    if type == 'baseline':
        return random_policy(num_states, num_actions)

    elif type == 'q-learning':
        Q = np.zeros((num_states, num_actions))
        Q_final = Q_learning(Q, grid, DISCOUNT)

    return build_policy(Q_final)



def random_policy(num_states, num_actions):

    result_visual = []
    result = []

    for i in range(10):
        row_visual = []
        row = []

        for j in range(10):
            coin_flip = rand.randint(0,3)
            row.append(coin_flip)
            
            if coin_flip == 0:
                row_visual.append('<')
            elif coin_flip == 1:
                row_visual.append('>')
            elif coin_flip == 2:
                row_visual.append('^')
            elif coin_flip == 3:
                row_visual.append('v')

        result.append(row)
        result_visual.append(row_visual)

    #return result, result_visual
    return np.array(result).reshape((10,10))




def Q_learning(Q, grid, discount):
    shape = Q.shape
    Q_flat = Q.flatten()
    grid_flat = grid.flatten()

    rewardperep = []
    for epoch in range(NUM_EPOCHS):
        epsilon = EPSILON_START
        undiscounted_reward = 0.0

        while True:
            s = rand.randint(0,99)
            if grid_flat[s] == R_MINERAL or grid_flat[s] == R_HOLE:
                break

        for time_step in range(HORIZON):

            a, epsilon = epsilon_greedy(Q_flat,s,shape,epsilon)
            r, sp = next_step(s,a,grid_flat)

            linear_index_s = np.ravel_multi_index((s,a), shape)
            linear_indices_sp = [np.ravel_multi_index((sp,a), shape) for a in range(shape[1])]
            linear_index_sp = linear_indices_sp[np.argmax([Q_flat[ind] for ind in linear_indices_sp])]

            undiscounted_reward += r
            Q_flat[linear_index_s] += ALPHA*(r + (discount * (Q_flat[linear_index_sp] - Q_flat[linear_index_s])))
            s = sp

            if r == R_MINERAL or r == R_HOLE:
                break
        
        rewardperep.append(undiscounted_reward)

    with open('rewardperep.csv', 'a') as file:
        for i,val in enumerate(rewardperep):
            if i+1 == len(rewardperep):
                file.write(str(val) +'\n')
            else:
                file.write(str(val) + ',')

    
    Q = Q_flat.reshape(shape)
    return Q




def epsilon_greedy(Q, s, shape, epsilon):
    if rand.random() <= epsilon:
        action = rand.randint(0,3)
        epsilon *= BETA
    else:
        # pick greedy action
        linear_indices_s = [np.ravel_multi_index((s,a), shape) for a in range(shape[1])]
        action = np.argmax([Q[ind] for ind in linear_indices_s])

    return action, epsilon

def next_step(s,a,grid):
    sp = s
    if a == 0:
        # step left
        if s % 10 != 0:
            sp = s - 1

    elif a == 1:
        # step right
        if s % 10 != 9:
            sp = s + 1

    elif a == 2:
        # step up
        if s not in range(0,10):
            sp = s - 10

    elif a == 3:
        # step down
        if s not in range(90,100):
            sp = s + 10

    if s == sp:
        return R_WALL, sp
    
    return grid[sp], sp


def build_policy(Q):
    return np.array([np.argmax(values) for values in Q]).reshape(10,10)

def test_policy(policy, grid):
    shape = policy.shape
    policy_flat = policy.flatten()
    grid_flat = grid.flatten()
    

    average_total_reward = 0.0
    denominator = 100

    for start in range(1,101):
        s = start
        if grid_flat[s-1] == R_HOLE or grid_flat[s-1] == R_MINERAL:
            denominator -= 1
        
        total_reward = 0.0
        
        for step in range(20):
            
            a = policy_flat[s-1]

            if a == 0:
                sp = s - 1
                if s % 10 == 1: 
                    sp = s

            elif a == 1:
                sp = s + 1
                if s % 10 == 0:
                    sp = s
                    
            elif a == 2:
                sp = s - 10
                if s in range(1,10):
                    sp = s
            else:
                sp = s + 10
                if s in range(90,101):
                    sp = s
            
            reward = grid_flat[sp-1]
            total_reward += reward

            if reward == R_MINERAL or reward == R_HOLE:
                break

            s = sp

        average_total_reward += total_reward

    return average_total_reward/denominator

def plot_reward_per_ep():
    df = pd.read_csv('rewardperep.csv')
    df_plot = df.mean(axis=0)
    graph = df_plot.plot(title="Average Undiscounted Reward per Epoch")
    graph.set_xlabel("Epoch")
    graph.set_ylabel("Undiscounted Reward")
    plt.savefig('output.png')



def main():
    args = sys.argv[1:]

    if 'plot-reward-per-ep' == args[0]:
        plot_reward_per_ep()
    
    elif 'compare' == args[0]:
        grids = args[1]
        random_policies = []
        q_learning_policies = []
        reward_grids = []

        for i, grid_csv in enumerate(os.listdir(grids)):
            if i % 100 == 0: print("TRAINED ", i)
            mineral_locations, hole_locations, num_states, num_actions, grid_data = unpack_csv(grids+grid_csv)
            reward_grids.append(get_grid(mineral_locations, hole_locations))
            
            random_policies.append(reinforcement_learning(num_states, num_actions, grid_data, type='baseline'))
            q_learning_policies.append(reinforcement_learning(num_states, num_actions, grid_data, grid=reward_grids[i], type='q-learning-2'))

        print("\n---\n")

        avg_total_r_random = []
        avg_total_r_q_learning = []
        for i, grid in enumerate(reward_grids):
            if i % 100 == 0: print("TESTED ", i)
            random_p_avg = round(test_policy(random_policies[i], grid), 3)
            q_p_avg = round(test_policy(q_learning_policies[i], grid),3)
            
            if random_p_avg > q_p_avg:
                print("Random outperformed Q-Learning on Grid ", i)

            avg_total_r_random.append(random_p_avg)
            avg_total_r_q_learning.append(q_p_avg)



        with open('test_averages_7.csv', 'w') as file:
            file.write('grid,random,q\n')
            for i in range(len(reward_grids)):
                file.write(str(i) + ',')
                file.write(str(avg_total_r_random[i]) + ',')
                file.write(str(avg_total_r_q_learning[i]) + '\n')

        return

    # otherwise, just test one rl algorithm and stream the policy to a file
    elif 'baseline' == args[0]:
        #run baseline
        rl_type = 'baseline'
        policy_filename = 'baseline.csv'
        
        mineral_locations, hole_locations, num_states, num_actions, grid_data = unpack_csv(args[1])
        grid = get_grid(mineral_locations, hole_locations)
        print("\n---GRID DATA---\n")
        print("\nmineral_locations: ", mineral_locations)
        print("hole_locations: ", hole_locations)
        print(grid)
        
        policy = reinforcement_learning(num_states, num_actions, None, type=rl_type)
        print("\n---POLICY---\n")
        print(policy)
        
        avg_total_reward = test_policy(policy, grid)
        print('\n---AVG TOTAL REWARD---\n')
        print(avg_total_reward)

    elif 'q-learning' == args[0]:
        rl_type = 'q-learning'
        policy_filename = 'q_learning.csv'

        mineral_locations, hole_locations, num_states, num_actions, grid_data = unpack_csv(args[1])

        grid = get_grid(mineral_locations, hole_locations)
        
        print("\n---GRID DATA---\n")
        print("\nmineral_locations: ", mineral_locations)
        print("hole_locations: ", hole_locations)
        print(grid)

        policy = reinforcement_learning(num_states, num_actions, grid_data, grid=grid, type=rl_type)

        print("\n---POLICY---\n")
        print(policy)

        avg_total_reward = test_policy(policy, grid)
        print('\n---AVG TOTAL REWARD---\n')
        print(avg_total_reward,'\n')

    else:
        print("Invalid in-line command")
        return

    with open(policy_filename, 'w') as file:
        for val in policy:
            file.write(str(val) + '\n')


if __name__ == "__main__":
    main()