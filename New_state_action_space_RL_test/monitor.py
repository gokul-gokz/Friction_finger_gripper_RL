from collections import deque
import sys
import math
import numpy as np
import json
from matplotlib import pyplot as plt

def interact(env, agent, num_episodes=150000, max_steps=700, window=100):
    """ Monitor agent's performance.
    
    Params
    ======
    - env: instance of Friction finger environment
    - agent: instance of class Agent (see agent.py for details)
    - num_episodes: number of episodes of agent-environment interaction
    - max_steps: maximum number of steps, before terminating the episode
    - window: number of episodes to consider when calculating average rewards

    Returns
    =======
    - avg_rewards: deque containing average rewards
    - best_avg_reward: largest value in the avg_rewards deque
    - policy: argmax of each state in the Q_table
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

        if (i_episode >= 100):
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
        # check if task is solved
        # if best_avg_reward >= 25.0 or i_episode==num_episodes:
        if i_episode == num_episodes:
            # print(len(avg_rewards))
            # print(len(list(range(0, 200))))
            # print('\nEnvironment solved in {} episodes.'.format(i_episode), end="")
            policy=dict()
            Q_t=dict()
            for key, val in Q.items():

                policy[str(key)]=int(np.argmax(Q[key]))
                Q_t[str(key)]=val

            with open('Policy.txt', 'w') as pol:
                json.dump(policy, pol)
            with open('Q_table.txt', 'w') as table:
                json.dump(Q_t, table)
            #break

            plt.bar(list(range(0, num_episodes)), avg_rewards)
            plt.show()
            plt.savefig('Avg_rewards_bar_graph.png')
            plt.plot(avg_rewards)
            plt.show()
            plt.savefig('Avg_rewards.png')
            break

    plt.plot(errors)
    plt.show()
    plt.savefig('Errors.png')
    return avg_rewards, best_avg_reward,policy