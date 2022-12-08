import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import random
from collections import deque
import os
from tqdm import tqdm
total_rewards = []
min = 10000

class replay_buffer():
    '''
    A deque storing trajectories
    '''

    def __init__(self, capacity):
        self.capacity = capacity  # the size of the replay buffer
        self.memory = deque(maxlen=capacity)  # replay buffer itself

    def insert(self, state, action, reward, next_state, done):
        '''
        Insert a sequence of data gotten by the agent into the replay buffer.
        Parameter:
            state: the current state
            action: the action done by the agent
            reward: the reward agent got
            next_state: the next state
            done: the status showing whether the episode finish
        Return:
            None
        '''
        self.memory.append([state, action, reward, next_state, done])

    def sample(self, batch_size):
        '''
        Sample a batch size of data from the replay buffer.
        Parameter:
            batch_size: the number of samples which will be propagated through the neural network
        Returns:
            observations: a batch size of states stored in the replay buffer
            actions: a batch size of actions stored in the replay buffer
            rewards: a batch size of rewards stored in the replay buffer
            next_observations: a batch size of "next_state"s stored in the replay buffer
            done: a batch size of done stored in the replay buffer
        '''
        batch = random.sample(self.memory, batch_size)
        observations, actions, rewards, next_observations, done = zip(*batch)
        return observations, actions, rewards, next_observations, done


class Net(nn.Module):
    '''
    The structure of the Neural Network calculating Q values of each state.
    '''
    def __init__(self,  num_actions, hidden_layer_size=50):
        super(Net, self).__init__()
        self.input_state = 4  # the dimension of state space
        self.num_actions = num_actions  # the dimension of action space
        self.fc1 = nn.Linear(self.input_state, 32)  # input layer
        self.fc2 = nn.Linear(32, hidden_layer_size)  # hidden layer
        self.fc3 = nn.Linear(hidden_layer_size, num_actions)  # output layer

    def forward(self, states):
        '''
        Forward the state to the neural network.
        Parameter:
            states: a batch size of states
        Return:
            q_values: a batch size of q_values
        '''
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class Agent():
    def __init__(self, env, epsilon=0.05, learning_rate=0.0002, GAMMA=0.97, batch_size=32, capacity=10000,exploration_rate=1.0):
        """
        The agent learning how to control the action of the cart pole.
        
        Hyperparameters:
            epsilon: Determines the explore/expliot rate of the agent
            learning_rate: Determines the step size while moving toward a minimum of a loss function
            GAMMA: the discount factor (tradeoff between immediate rewards and future rewards)
            batch_size: the number of samples which will be propagated through the neural network
            capacity: the size of the replay buffer/memory
        """
        self.exploration_rate=exploration_rate
        self.env = env
        self.n_actions = 2  # the number of actions
        self.count = 0  # recording the number of iterations

        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = GAMMA
        self.batch_size = batch_size
        self.capacity = capacity

        self.buffer = replay_buffer(self.capacity)
        self.evaluate_net = Net(self.n_actions)  # the evaluate network
        self.target_net = Net(self.n_actions)  # the target network

        self.optimizer = torch.optim.Adam(
            self.evaluate_net.parameters(), lr=self.learning_rate)  # Adam is a method using to optimize the neural network

    def learn(self):
        '''
        - Implement the learning function.
        - Here are the hints to implement.
        Steps:
        -----
        1. Update target net by current net every 100 times. (we have done for you)
        2. Sample trajectories of batch size from the replay buffer.
        3. Forward the data to the evaluate net and the target net.
        4. Compute the loss with MSE.
        5. Zero-out the gradients.
        6. Backpropagation.
        7. Optimize the loss function.
        -----
        Parameters:
            self: the agent itself.
            (Don't pass additional parameters to the function.)
            (All you need have been initialized in the constructor.)
        Returns:
            None (Don't need to return anything)
        '''
        if self.count % 100 == 0:
            self.target_net.load_state_dict(self.evaluate_net.state_dict())
        if len(self.buffer.memory) < self.batch_size:
            return

        # Begin your code
        b_state, b_action, b_reward, b_next_state, b_done = self.buffer.sample(self.batch_size) # randomly choose from buffer
        b_state = torch.FloatTensor(np.array(b_state))
        b_action = torch.LongTensor(b_action)
        b_reward = torch.FloatTensor(b_reward)
        b_next_state = torch.FloatTensor(np.array(b_next_state))
        b_done = torch.FloatTensor(b_done)
        b_done = 1 - b_done
        batch_index = np.arange(self.batch_size, dtype=np.int64)

        q_values = self.evaluate_net.forward(b_state)
        next_q_values = self.target_net.forward(b_next_state)

        predicted_value_of_now = q_values[batch_index, b_action]
        predicted_value_of_next = torch.max(next_q_values, dim = 1)[0]
        
        q_target = b_reward + self.gamma * predicted_value_of_next * b_done
        loss_function = nn.MSELoss()
        loss = loss_function(predicted_value_of_now, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.exploration_rate = max(0.001, self.exploration_rate * 0.999)
        # End your code
        global min
        if loss < min:
            torch.save(self.target_net.state_dict(), "./Tables/DQN.pt")
            min = loss

    def choose_action(self, state):
        """
        - Implement the action-choosing function.
        - Choose the best action with given state and epsilon
        Parameters:
            self: the agent itself.
            state: the current state of the enviornment.
            (Don't pass additional parameters to the function.)
            (All you need have been initialized in the constructor.)
        Returns:
            action: the chosen action.
        """
        # Begin your code
        with torch.no_grad(): # I don’t have to train the network, because I just need choose one action using current weights.
            if np.random.random() < self.exploration_rate:
                return env.action_space.sample() # I explore an action which is haven't learned
            else:
                _state_ = torch.tensor(state).float().detach().to(torch.device("cpu"))
                _state_ = _state_.unsqueeze(0)
                q_values = self.target_net(_state_)
                return torch.argmax(q_values).item()
        # End your code

    def check_max_Q(self):
        """
        - Implement the function calculating the max Q value of initial state(self.env.reset()).
        - Check the max Q value of initial state
        
        Parameter:
            self: the agent itself.
            (Don't pass additional parameters to the function.)
            (All you need have been initialized in the constructor.)
        
        Return:
            max_q: the max Q value of initial state(self.env.reset())
        """
        # Begin your code
        initial_state = torch.unsqueeze(torch.FloatTensor(self.env.reset()),0)
        return max(self.target_net(initial_state)) #!
        # End your code


def train(env):
    """
    Train the agent on the given environment.
    Paramenters:
        env: the given environment.
    Returns:
        None (Don't need to return anything)
    """
    agent = Agent(env)
    episode = 1000 # training times
    rewards = []
    global min
    min = 10000
    for _ in tqdm(range(episode)):
        state = env.reset()
        count = 0
        while True:
            count += 1
            agent.count += 1
            # env.render()
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.buffer.insert(state, int(action), reward, next_state, int(done))
            if agent.count >= 1000:
                agent.learn()
            if done:
                rewards.append(count)
                break
            state = next_state
    print(f"reward: {np.mean(rewards)}")
    total_rewards.append(rewards)


def test(env):
    """
    Test the agent on the given environment.
    Paramenters:
        env: the given environment.
    Returns:
        None (Don't need to return anything)
    """
    rewards = []
    testing_agent = Agent(env)
    testing_agent.target_net.load_state_dict(torch.load("./Tables/DQN.pt"))
    for _ in range(100):
        state = env.reset()
        count = 0
        while True:
            count += 1
            Q = testing_agent.target_net.forward(
                torch.FloatTensor(state)).squeeze(0).detach()
            action = int(torch.argmax(Q).numpy())
            next_state, _, done, _ = env.step(action)
            if done:
                rewards.append(count)
                break
            state = next_state
    print(f"reward: {np.mean(rewards)}")
    print(f"max Q:{testing_agent.check_max_Q()}")


def seed(seed = 112):
    '''
    It is very IMPORTENT to set random seed for reproducibility of your result!
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    '''
    The main funtion
    '''
    # Please change to the assigned seed number in the Google sheet
    SEED = 112
    env = gym.make('CartPole-v0')
    seed(SEED)
    env.seed(SEED)
    env.action_space.seed(SEED)
        
    if not os.path.exists("./Tables"):
        os.mkdir("./Tables")
    # training section:
    for i in range(5):
        print(f"#{i + 1} training progress")
        train(env)
    # testing section:
    test(env)
    
    if not os.path.exists("./Rewards"):
        os.mkdir("./Rewards")

    np.save("./Rewards/DQN_rewards.npy", np.array(total_rewards))

    env.close()