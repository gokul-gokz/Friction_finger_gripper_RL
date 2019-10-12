from collections import deque
import sys
import math
import numpy as np
import csv
import json
from matplotlib import pyplot as plt
max_steps=1000
def interact(env, agent, num_episodes=100000, window=100):
    """ Monitor agent's performance.
    
    Params
    ======
    - env: instance of OpenAI Gym's Taxi-v1 environment
    - agent: instance of class Agent (see Agent.py for details)
    - num_episodes: number of episodes of agent-environment interaction
    - window: number of episodes to consider when calculating average rewards

    Returns
    =======
    - avg_rewards: deque containing average rewards
    - best_avg_reward: largest value in the avg_rewards deque
    """
    # initialize average rewards
    avg_rewards = deque(maxlen=num_episodes)
    # initialize best average reward
    best_avg_reward = -math.inf
    # initialize monitor for most recent rewards
    samp_rewards = deque(maxlen=window)
    # for each episode
    errors = []
    for i_episode in range(1, num_episodes+1):
        # begin the episode
        state = env.reset()
        # initialize the sampled reward
        samp_reward = 0
        step_num=0
        mean_error=0

        while True:
            # agent selects an action
            action = agent.select_action(state)
            # agent performs the selected action
            next_state, reward, done  = env.step(action)
            #print(next_state, reward, done)
            # agent performs internal updates based on sampled experience
            Q,error=agent.step(state, action, reward, next_state, done)
            # update the sampled reward
            samp_reward += reward
            # update the state (s <- s') to next time step
            state = next_state
            step_num = step_num + 1
            #calculate mean error
            mean_error=(mean_error*(step_num-1)+ error)/step_num

            if done or step_num>=max_steps:
                # save final sampled reward
                errors.append(mean_error)
                print("steps=",step_num)
                samp_rewards.append(samp_reward)
                break

        if (i_episode >= 1):
            # get average reward from last 100 episodes
            avg_reward = np.mean(samp_rewards)
            # append to deque
            avg_rewards.append(avg_reward)
            # update best average reward
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
        # monitor progress

        print("\rEpisode {}/{} || Best average reward {}".format(i_episode, num_episodes, best_avg_reward), end="")
        sys.stdout.flush()
        # check if task is solved (according to OpenAI Gym)
        if best_avg_reward >= -20.0 or i_episode==num_episodes:
            print(len(avg_rewards))
            print(len(list(range(0, 200))))

            print('\nEnvironment solved in {} episodes.'.format(i_episode), end="")
            policy=dict()
            Q_t=dict()
            # w = csv.writer(open("Q_table.csv", "w"))
            # w1 = csv.writer(open("Policy.csv", "w"))
           
                

            for key, val in Q.items():
                # w.writerow([key, val])
                policy[str(key)]=int(np.argmax(Q[key]))
                Q_t[str(key)]=val
                # if(policy[str(key)])==0:
                #     act="Left up"
                # elif(policy[str(key)])==1:
                #     act = "Left down"
                # elif (policy[str(key)]) == 2:
                #     act = "Right up"
                # elif (policy[str(key)]) == 3:
                #     act = "Right down"
                # elif (policy[str(key)]) == 4:
                #     act = "Rotate clock"
                # elif (policy[str(key)]) == 5:
                #     act = "Rotate anticlock"
            with open('Policy.txt', 'w') as pol:
                json.dump(policy, pol)
            with open('Q_table.txt', 'w') as table:
                json.dump(Q_t, table)
                #w1.writerow([key, act])
            # plt.bar(list(range(0, num_episodes)), avg_rewards)
            # plt.plot(avg_rewards)
            # plt.show()
            #break
        if i_episode == num_episodes: print('\n')
    print
    plt.plot(errors)
    plt.show()
    return avg_rewards, best_avg_reward,policy