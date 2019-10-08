from collections import deque
import sys
import math
import numpy as np
import csv

def interact(env, agent, num_episodes=200000, window=100):
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
    for i_episode in range(1, num_episodes+1):
        # begin the episode
        state = env.reset()
        # initialize the sampled reward
        samp_reward = 0
        while True:
            # agent selects an action
            action = agent.select_action(state)
            # agent performs the selected action
            next_state, reward, done  = env.step(action)
            #print(next_state, reward, done)
            # agent performs internal updates based on sampled experience
            Q=agent.step(state, action, reward, next_state, done)
            # update the sampled reward
            samp_reward += reward
            # update the state (s <- s') to next time step
            state = next_state
            if done:
                # save final sampled reward
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
        # check if task is solved (according to OpenAI Gym)
        if best_avg_reward >= -4.0 or i_episode==199999:
            print('\nEnvironment solved in {} episodes.'.format(i_episode), end="")
            policy=dict()
            w = csv.writer(open("Q_table.csv", "w"))
            w1 = csv.writer(open("Policy.csv", "w"))
            for key, val in Q.items():
                w.writerow([key, val])
                policy[key]=np.argmax(Q[key])
                if(policy[key])==0:
                    act="Left up"
                elif(policy[key])==1:
                    act = "Left down"
                elif (policy[key]) == 2:
                    act = "Right up"
                elif (policy[key]) == 3:
                    act = "Right down"
                elif (policy[key]) == 4:
                    act = "Rotate clock"
                elif (policy[key]) == 5:
                    act = "Rotate anticlock"
                w1.writerow([key, act])

            break
        if i_episode == num_episodes: print('\n')
    print
    return avg_rewards, best_avg_reward,policy