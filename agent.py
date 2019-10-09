import numpy as np
from collections import defaultdict
import random

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: [0,0,0,0,0,0])    #Since we are storing in a json file, using a list instead of numpy arrays
        self.epsilon=1.0
        self.epsilon_decay=0.00001
        self.epsilon_min=0.00003
        self.alpha=0.24
        self.gamma=0.9

    def update_epsilon(self):
        self.epsilon=max(self.epsilon*self.epsilon_decay,self.epsilon_min)

    def get_action_probs(self,state):
        prob_action_policy=np.ones(self.nA)*(self.epsilon/self.nA)
        best_a=np.argmax(self.Q[state])
        prob_action_policy[best_a]=1.0-self.epsilon+(self.epsilon/self.nA)
        return prob_action_policy

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        n=np.random.random()
        if n>self.epsilon:
            action=np.argmax(self.Q[state])
        else:
            action= np.random.choice(self.nA)

        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        #print("action",action)
        action_prob=self.get_action_probs(next_state)
        self.Q[state][action] = self.Q[state][action]+ self.alpha*(reward+self.gamma*(np.dot(self.Q[next_state],action_prob))-self.Q[state][action])
        self.update_epsilon()
        return self.Q