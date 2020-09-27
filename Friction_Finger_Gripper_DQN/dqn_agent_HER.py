import numpy as np
import random
from collections import namedtuple, deque
from env_mg import compute_reward
from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, fc1_units, fc2_units, buffer_size, batch_size, gamma, tau, lr, HER_update_episodes, bootstrap, network_path):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        global device
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.network_path=network_path

        self.buffer_size = int(buffer_size)  #replay buffer size
        self.batch_size = int(batch_size)   #minibatch size
        self.gamma = float(gamma) #discount factor
        self.tau = float(tau) #soft  update of target params
        self.lr = float(lr) #learning rate
        self.HER_update_episodes = int(HER_update_episodes) #update every

        
        # Q-Network
        if(bootstrap):
            self.qnetwork_local = QNetwork(state_size, action_size, seed, fc1_units, fc2_units).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed,fc1_units, fc2_units).to(device)
            # self.qnetwork_local.load_state_dict(torch.load(self.network_path),map_location=torch.device(device))
            self.qnetwork_local.load_state_dict(torch.load(self.network_path,map_location=torch.device(device)))
            self.qnetwork_target.load_state_dict(torch.load(self.network_path,map_location=torch.device(device)))
        else:
            self.qnetwork_local = QNetwork(state_size, action_size, seed, fc1_units, fc2_units).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed, fc1_units, fc2_units).to(device)
       
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        # self.optimizer = optim.RMSprop(self.qnetwork_local.parameters(), lr = self.lr)
        #To store experience
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        # Episode memory
        self.episode_memory=[]
        # Replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)
        # Initialize time step (for updating every self.HER_update_episodes steps)
        self.t_step = 0
        # Memory to store the current episode's experience
        self.HER_experience_memory=ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)

        self.Q_loss=0


    def step(self, state, action, reward, next_state, done, goal , her_enable=1, future_k=4,EPISODE_LENGTH=100):

        # self.lr=LR
        # Save experience in replay memory

        self.episode_memory.append(self.experience(state, action, reward, next_state, done))
        if done or len(self.episode_memory)==EPISODE_LENGTH:
            steps_taken = len(self.episode_memory)
            for t in range(steps_taken):

                self.memory.add(np.concatenate((self.episode_memory[t].state, goal)),self.episode_memory[t].action,self.episode_memory[t].reward,np.concatenate((self.episode_memory[t].next_state,goal)),self.episode_memory[t].done)


                if(her_enable):
                    for _ in range(future_k):
                        future = random.randint(t, steps_taken-1)
                        new_goal=self.episode_memory[future].next_state
                        new_reward,new_done=compute_reward(self.episode_memory[t].next_state,new_goal)
                        MG_state,MG_next_state=np.concatenate((self.episode_memory[t].state,new_goal)),np.concatenate((self.episode_memory[t].next_state,new_goal))
                        self.memory.add(MG_state,self.episode_memory[t].action,new_reward,MG_next_state,new_done)

            del self.episode_memory[:]


        # Learn every self.HER_update_episodes time steps.
        self.t_step = (self.t_step + 1) % self.HER_update_episodes

        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)



    def act(self, state, eps=0.5):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """


        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        ran=np.random.random()
        #print(ran,eps)
        if ran >eps:
            # print("greedy")
            return np.argmax(action_values.cpu().data.numpy())
        else:
            # print("explore")
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        # loss = F.smooth_l1_loss(Q_expected, Q_targets)
        loss = F.mse_loss(Q_expected, Q_targets)
        self.Q_loss = loss
        # Minimize the loss
        self.optimizer.zero_grad()
        
        loss.backward()
        # print(self.qnetwork_local.fc2.weight.grad)
        # for param in self.qnetwork_local.parameters():
        #   param.grad.data.clamp_(-10, 10)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

        #Save network
        torch.save(self.qnetwork_local.state_dict(), self.network_path)

    def soft_update(self, local_model, target_model):

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def ordered_sample(self):
        """Return ordered sample of batch of experiences from memory."""

        experiences=[]
        for i in range(self.batch_size):
            experiences.append(self.memory.pop())

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)



    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
