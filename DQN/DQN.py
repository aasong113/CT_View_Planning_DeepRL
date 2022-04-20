import torch as T
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
import numpy as np


# need to discretize the action space. 
class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input.dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions) 
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    # define forward propagation. 
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # we want the raw number because this could be negative or positive. 
        actions = self.fc3(x)


# gamma = determines weighting of future rewards. 
# epsilon = explore vs. exploit, how often does the agent explore the environment vs. taking a known reward. 
class Agent(): 
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size = 10000, eps_end = 0.01, eps_dec = 5e-4):
        self.gamma = gamma 
        self.epsilon = epsilon
        self.lr = lr
        self.input_dims = input_dims
        self.batch_size = batch_size
        # create an action space for the actions. 
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.memory_counter = 0

        self.Q_eval = DeepQNetwork(self.lr, n_actions = n_actions, input_dims = input_dims, fc1_dims = 256, fc2_dims = 256)
        

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype = np.float32)
        # memory to keep track of the new states the agent encounters. 

        # what is the value of the current state via the previous state. the reward received and action took. 
        # Have to pass in the memory of the state that resulted from its action. 
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype = np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype = np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype = np.float32)

        # if you get to the terminal state the agent is done. 
        # the future value of the terminal state is zero. 
        self.terminal_memory = np.zeros(self.mem_size, dtype = np.bool)


        def store_transition(self, state, action, reward, state_, done ):

            # determine position of first unoccupied memory. modulus makes it warp around back to zero. 
            index = self.memory_counter % self.mem_size

            #store memory 
            self.state_memory[index] = state
            self.new_state_memory[index] = state_
            self.reward_memory[index] = reward
            self.action_memory[index] = action
            self.terminal_memory[index] = done
            
            
            self.memory_counter +=1

        def choose_action(self, observation):

            # take an action based on Q network. 
            if np.random.random() > self.epsilon:
                state = T.tensor([observation]).to(self.Q_eval.device)
                actions = self.Q_eval.forward(state)
                action = T.argmax(actions).item()
            
            # if not greater than epsilon, then take a random action. 
            else:
                action = np.random.choice(self.action_space)
            
            return action


        # start learning once you fill up the batch size of memory .
        def learn(self):
            if self.memory_counter < self.batch_size:
                return

            self.Q_eval.optimizer.zero_grad()

            max_mem = min(self.memory_counter, self.mem_size)

            # once you select a memory, make sure it cannot be selected again. 
            batch = np.random.choice(max_mem, self.batch_size, replace = False)

            batch_index = np.arange(self.batch_size, dtype = np.int32)

            state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
            new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
            reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
            terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

            action_batch = self.action_memory[batch]

            # We want to tilt our agent towards taking maximal actions. 
            # these are the values of actions that we actually took. 
            q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]

            # agents estimates of the next states as well. 
            q_next = self.Q_eval.forward(new_state_batch)
            q_next[terminal_batch] = 0.0

            # update our estimates towards the goal. 
            # This is the maximum value of the next state which is the purely greedy action, this is used to update the loss function. 
            q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

            # Loss function and update. 
            loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
            loss.backward()
            self.Q_eval.optimizer.step()

            # epsilon decrement, each time we learn, we decrement epsilon by 1. 
            self.epsilon = self.epsilon - self.eps_dec \
                if self.epsilon > self.eps_min else self.eps_min 



             


             







