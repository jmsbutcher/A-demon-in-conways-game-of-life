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

"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Agent():
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
        self.last_reward = 0

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
                          torch.Tensor([self.last_reward])))

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
        self.last_reward = reward

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


class RewardScheme:
    # Handles the reward mechanism for the agent
    #   - Calculates the reward based on the agent's current view according to
    #     the scheme criteria (maximize live cells, make a shape of cells, etc)
    def __init__(self, vf,
                       vision,
                       schemetype=None,
                       shape_name=None,
                       desired_shape=None):
        # The agent's visual field
        self.vf = vf

        # The agent's vision object for retrieving visual data
        self.vision = vision

        # Scheme types:
        #  - "shape": get high reward for having the desired shape
        #      in the visual field.
        #  - "maximize": get higher reward based on number of live cells
        #      in the visual field.
        #  - "minimize": get higher reward based on number of dead cells
        #      in the visual field
        self.schemetype = schemetype

        # The desired shape, if using "shape" scheme type -- representing the
        #   desired shape of live/dead cells in the agent's visual field.
        #   - 0 -- Live cell
        #   - 1 -- Dead cell
        # Example: If you want the agent to make flippers, do this:
        #   flipper_scheme = RewardScheme(...,
        #                                 desired_shape=np.array([[1, 1, 1],
        #                                                         [0, 0, 0],
        #                                                         [1, 1, 1]]))
        # (Hint: it might be better to make a 5x5 array instead, with 1s
        #   all around to keep the flipper isolated so it works correctly.)
        #
        # *** Note: A "shape" reward scheme will only work in the game
        #   if it is the same exact dimensions as the visual field. For the
        #   above example to work, the visual field must by a 3x3 array.
        self.desired_shape = desired_shape

        # Optional name for the desired shape, if using "shape" scheme type
        self.shape_name = shape_name

        # Variable to signal that an exact match has been found, if using
        #   "shape" scheme type
        self.exact_match = False

        # Print message upon creating a new RewardScheme
        if schemetype == "shape":
            if desired_shape is not None:
                print("Reward scheme: shape")
                print("-Make the following shape:")
                if shape_name is not None:
                    print(shape_name)
                for i in range(len(desired_shape)):
                    for j in range(len(desired_shape[i])):
                        if desired_shape[i][j] == 0:
                            print(" #", end="")
                        else:
                            print("  ", end="")
                    print()
            else:
                print("Must provide a desired shape for 'shape' reward scheme")
        elif schemetype == "maximize":
            print("Reward scheme: maximize")
            print("-Maximize live cells")
        elif schemetype == "minimize":
            print("Reward scheme: minimize")
            print("-Minimize live cells")
        elif schemetype is None:
            print("No reward scheme\n"
                  "Create new reward scheme by clicking"
                  "'New' --> 'Reward Scheme'")
        else:
            print("ERROR: Must provide a reward scheme type.")

    def check_for_exact_match(self):
        return self.exact_match

    def get_reward(self):
        # Calulate and return the reward based on what the agent currently
        #   sees according to the scheme criteria below:
        #
        if self.schemetype == "shape":
            # Get higher reward for each cell in agent's view that matches
            #   the corresponding cell in desired shape.
            # If ALL the cells match, then get maximum reward of 10 and
            #   set self.exact_match variable to True
            view = self.vision.get_view()

            # Check that the view has the same dimensions as the visual field
            if self.vf.shape != view.shape:
                print("ERROR: Reward shape doesn't match visual field shape")
                return

            r = 1
            self.exact_match = True
            for i in range(len(view)):
                for j in range(len(view[i])):
                    # If within the visual field:
                    if self.vf[i][j] > 0:
                        # Check whether view point value matches the shape
                        if view[i][j] == self.desired_shape[i][j]:
                            r += 0.1
                        else:
                            r -= 0.1
                            self.exact_match = False
            reward = r
            if self.exact_match:
                reward = 10

        elif self.schemetype == "maximize":
            # Get higher reward for each live cell in the agent's current view
            viewdata = self.vision.get_viewdata()
            r = 0
            for cell in viewdata:
                if cell == 0:
                    r += 1
            # Scale reward according to size of the visual field
            reward = int(r / len(viewdata) * 20)

        elif self.schemetype == "minimize":
            # Get lower reward for each live cell in the agent's current view
            viewdata = self.vision.get_viewdata()
            r = 20
            for cell in viewdata:
                if cell == 0 and r > 0:
                    r -= 1
            # Scale reward according to size of the visual field
            reward = int(r / len(viewdata) * 20)

        else:
            reward = 0

        return reward

    def get_reward_text(self):
        # Generate schemetype-specific text to display in the display console
        if self.schemetype == "shape":
            text = "Produce shape:\n"
            if self.shape_name:
                text += "\"" + self.shape_name + "\""
            return text
        elif self.schemetype == "maximize":
            return "Maximize life in\n" \
                    "the visual field."
        elif self.schemetype == "minimize":
            return "Minimize life in\n" \
                   "the visual field."
        elif self.schemetype is None:
            return "No reward scheme"
        else:
            return "Other"

    def get_shape(self):
        return self.desired_shape

    def get_shapename(self):
        return self.shape_name

    def get_antishape(self):
        # Return desired shape with all "1"s turned to "0"s and vice versa
        return abs(self.desired_shape - 1)

    def set_schemetype(self, new_schemetype):
        self.schemetype = new_schemetype
        if self.schemetype != "shape":
            self.shape_name = None
            self.desired_shape = None

    def set_shape(self, new_desired_shape, new_shapename=None):
        self.schemetype = "shape"
        self.desired_shape = new_desired_shape
        self.shape_name = new_shapename


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


class Vision:
    # Handle what the agent sees
    #   - Use information from the environment, where the agent's eye is
    #     located, and the shape of the visual field to create a representation
    #     of the agent's current view
    #   - Move the agent's eye location
    def __init__(self, env_data, vf, eye_loc):
        # Game environment data of cell states
        self.environment_data = env_data
        # Shape of the agent's field of view
        self.visual_field = vf
        # Coordinates of the agent's eye in the environment grid
        self.eye_location = eye_loc
        # Data about what the agent currently sees; see update_view() for more
        self.view = np.ones(vf.shape)
        self.viewdata = []

    def get_view(self):
        return self.view

    def get_viewdata(self):
        return self.viewdata

    def move(self, x_shift, y_shift):
        self.eye_location[0] += x_shift
        self.eye_location[1] += y_shift

    def update(self, data):
        self.environment_data = data
        self.update_view()

    def update_view(self):
        # Update the agent's view and its simplified form, viewdata
        #
        # A view point can be one of three values:
        #   - None -- if the point is not part of the visual field
        #   - 0 -- if the point is a live cell within the visual field
        #   - 1 -- if the point is a dead cell within the visual field
        #
        # Viewdata is a simple list of the values in view that are within the
        #   visual field. Any of those points that are off the edge of the
        #   environment grid will be denoted as None.
        # The size and shape for both view and viewdata are always the same
        #   for any given visual field template.
        #
        #  - Example of view:                    ... using this visual field:
        # np.array([[None, None,   1, None, None],  np.array([[0, 0, 1, 0, 0],
        #           [None,    0,   1,    1, None],            [0, 1, 1, 1, 0],
        #           [   1,    1,   0,    0,    0],    <--     [1, 1, 2, 1, 1],
        #           [None,    0,   1,    0, None],            [0, 1, 1, 1, 0],
        #           [None, None,   1, None, None]])           [0, 0, 1, 0, 0]])
        #
        #  - Example of corresponding viewdata:
        #   [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1]

        e = self.environment_data
        eye_x, eye_y = self.eye_location
        vf = self.visual_field
        view = np.ones(vf.shape)
        viewdata = []

        # Find local eye location in visual field
        for i in range(len(vf)):
            for j in range(len(vf[i])):
                if vf[i][j] == 2:
                    loc_eye_x, loc_eye_y = i, j

        # Get environment data within the visual field
        for i in range(len(vf)):
            for j in range(len(vf[i])):
                # Calculate absolute coordinates in the environment grid
                x = i + eye_x - loc_eye_x
                y = j + eye_y - loc_eye_y
                # If within visual field:
                if vf[i][j] > 0:
                    # If within bounds of the environment grid:
                    if 0 <= x < len(e) and 0 <= y < len(e[0]):
                        view[i][j] = e[x][y]
                        viewdata.append(e[x][y])
                    # If within visual field but out of bounds:
                    else:
                        view[i][j] = None
                        viewdata.append(None)
                else:
                    view[i][j] = None
        self.view = view
        self.viewdata = viewdata
