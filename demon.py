"""
 The AI agent that lives in the Conway's Game of Life game environment,
 "game.py"

 Demon version: 1.0
 Date: 12-26-19

 - Neural network and reinforcement learning mechanism based off module 1 of
   Udemy course "Artificial Intelligence A-Z: Learn How To Build An AI"
   by Hadelin de Ponteves, Kirill Eremenko, SuperDataScience Team
   https://www.udemy.com/course/artificial-intelligence-az/


 "A Demon in Conway's Game of Life"
 by James Butcher
 Github: https://github.com/jmsbutcher/A-demon-in-conways-game-of-life
 First created: 8-31-19

 Current version: 1.0
 Date: 12-26-19

 Python version 3.7.3
 Pytorch version 1.2.0
 Numpy version 1.17.2

"""

#import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from vision import Vision


class DemonAgent():
    def __init__(self, env_data, vf, abs_eye_loc, gamma):
        # Agent's vision -- for retrieving visual information from the
        #   environment and controlling movement around the environment
        self.vision = Vision(env_data, vf, abs_eye_loc)
        vision_size = int(vf.sum() - 1)

        # Agent's brain -- neural network for learning how to maximize reward
        self.brain = Brain(vision_size)
        # Discount factor: a number between 0 and 1
        self.gamma = gamma
        # Pytorch Adam Optimizer, learning rate of 0.01
        self.optimizer = optim.Adam(self.brain.parameters(), lr=0.01)

        # Agent's memory -- for batch learning
        self.memory = ReplayMemory(100000)

        self.last_state = torch.Tensor(vision_size).unsqueeze(0)
        self.last_action = 0
#        self.last_reward = 0

    def learn(self, batch_state, batch_next_state, batch_action, batch_reward):
        # TD learning procedure, using batches of 100

        # Get the value of the current state
        outputs = self.brain(batch_state).gather(1,
                                                 batch_action.unsqueeze(1)
                                                 ).squeeze(1)

        # Get the max expected value of the next state
        next_outputs = self.brain(batch_next_state).detach().max(1)[0]

        # Target is the discount factor times the expected future value plus
        #   the reward
        target = self.gamma * next_outputs + batch_reward

        # TD loss is the difference between the target and what we actually get
        td_loss = F.smooth_l1_loss(outputs, target)

        # Adjust the weights in the agent's brain
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()

    def load(self, file):
        # Load a saved brain state using Pytorch load function
        checkpoint = torch.load(file)
        self.brain.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def save(self, file):
        # Save the brain state to a file using Pytorch save function
        torch.save({"state_dict": self.brain.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    }, file)

    def select_action(self, state):
        # The Pytorch Softmax function normalizes the q-values into a
        #   probability distribution, and then one of these is chosen
        #   as the action to take. The distribution aims to balance
        #   exploration vs. exploitation.
        probabilities = F.softmax(self.brain(state), dim=1)

        # Sometimes we encounter a probability distribution < 0, which
        #   causes the program to crash. The cause is unknown.
        #   This is a workaround:
        for p in probabilities[0]:
            if p <= 0.0:
                print("Warning: Encountered Q-value less than 0")
                return 0

        # Select an action code (0 - 5) for one of six actions
        action = probabilities.multinomial(num_samples=1)

        return action.data[0]

    def update(self, reward, new_view):
        # Convert the agent's input view into a Pytorch Tensor
        new_state = torch.Tensor(new_view).float().unsqueeze(0)

        # Add last state, new state, action, and reward to memory
        self.memory.push((self.last_state,
                          new_state,
                          torch.LongTensor([int(self.last_action)]),
                          torch.Tensor([reward])))
#                          torch.Tensor([self.last_reward])))

        # Select action
        action = self.select_action(new_state)

        # Once the agent's memory reaches 100 samples, begin learning
        if len(self.memory.memory) > 100:
            # Get random batch of 100 states, new states, actions, & rewards
            batch_state, batch_next_state, batch_action, batch_reward = \
                self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_action,
                       batch_reward)

        self.last_action = action
        self.last_state = new_state
#        self.last_reward = reward

        return action


class Brain(nn.Module):
    # The agent's brain model -- I will make this customizable in the future.
    #
    #   - Input layer has as many states as there are cells in the visual field
    #   - Output layer has six states, one for each possible action:
    #     Up, Down, Left, Right, Flip, and Wait
    #
    def __init__(self, vision_size):
        super(Brain, self).__init__()
        self.fc1 = nn.Linear(vision_size, 50)   # 1st fully connected layer
        self.fc2 = nn.Linear(50, 50)            # 2nd fully connected layer
        self.fc3 = nn.Linear(50, 6)             # 6 output states

    def forward(self, view):
        # Rectifier function
        view = F.relu(self.fc1(view))
        view = F.relu(self.fc2(view))
        q_values = self.fc3(view)
        return q_values


class ReplayMemory():
    # Agent's memory used for batch learning
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        # An event consists of: (state, next state, action, reward)
        self.memory.append(event)

        # Start deleting the oldest event once memory reaches capacity
        if len(self.memory) > self.capacity:
            del self.memory[0]

    # Return a random batch of states, next states, actions and rewards
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: torch.cat(x, 0), samples)


