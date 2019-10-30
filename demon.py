"""
 The AI agent that lives in the Conway's Game of Life game environment, 
 "game.py"

"""

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Agent():
    def __init__(self, env_data, vf, abs_eye_loc, gamma):
        vision_size = vf.sum() - 1
        self.vision = Vision(env_data, vf, abs_eye_loc)
        self.brain = Brain(vision_size)
        self.gamma = gamma
        self.memory = ReplayMemory(100000) # For experience replay 9-27-19
        self.optimizer = optim.Adam(self.brain.parameters(), lr = 0.01)
        self.last_state = torch.Tensor(vision_size).unsqueeze(0) # Added unsqueeze 9-27-19
        self.last_action = 0
        self.last_reward = 0
        
    def select_action(self, state):
        #print("Q-values:", self.brain(state))               # For debugging
        probabilities = F.softmax(self.brain(state), dim=1)
        #print("Action Probabilities: ", probabilities)      # For debugging
        action = probabilities.multinomial(num_samples=1)
        return action.data[0]
    
    def learn(self, batch_state, batch_next_state, batch_action, batch_reward):
        outputs = self.brain(batch_state).gather(1, 
            batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.brain(batch_next_state).detach().max(1)[0]
        target = self.gamma * next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()
    
    def update(self, reward, new_view):
        new_state = torch.Tensor(new_view).float().unsqueeze(0)
        self.memory.push((self.last_state,
                          new_state,
                          torch.LongTensor([int(self.last_action)]),
                          torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = \
                self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_action, 
                       batch_reward)        
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        return action
    
    def save(self):
        torch.save({"state_dict": self.brain.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    }, "last_brain.pth")
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            checkpoint = torch.load('last_brain.pth')
            self.brain.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no saved brain found...")
    
    
class Brain(nn.Module):
    def __init__(self, vision_size):
        super(Brain, self).__init__()
        self.vision_size = vision_size  # Number of input states
        self.fc1 = nn.Linear(self.vision_size, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 6)     # 6 possible actions
        
    def forward(self, view):
        view = F.relu(self.fc1(view))
        view = F.relu(self.fc2(view))
        q_values = self.fc3(view)
        return q_values
    
    
class RewardScheme:
    def __init__(self, vf,
                       schemetype, 
                       shape_name=None,
                       desired_shape=None):
        # The agent's visual field; numpy array of rank 2
        self.vf = vf
        
        # Scheme types:
        #  - "shape": get high reward for having the desired shape
        #      in the visual field.
        #  - "maximize": get higher reward based on number of live cells
        #      in the visual field.
        #  - "minimize": get higher reward based on number of dead cells
        #      in the visual field
        self.schemetype = schemetype.lower()
        
        # Optional name for the desired shape
        self.shape_name = shape_name
        
        # Desired shape: a numpy array of rank 2, representing the desired 
        #   shape of live/dead cells in the agent's visual field.
        # Example: If you want the agent to make flippers, do this:
        #   flipper_scheme = RewardScheme(desired_shape=np.array([[1, 1, 1],
        #                                                         [0, 0, 0],
        #                                                         [1, 1, 1]]))
        # (Hint: it might be better to make a 5x5 grid instead, with "1"s
        #  all around to keep the flipper isolated)
        self.desired_shape = desired_shape
        self.reward = 0
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
        else:
            print("ERROR: Must provide a reward scheme type.")
    
    def calc_reward(self, view):
        # Check that the view has the same dimensions as the visual field
        if self.vf.shape != view.shape:
            print("ERROR: Reward shape doesn't match visual field shape")
            return
        reward = 1
        exact_match = True
        for i in range(len(view)):
            for j in range(len(view[i])):
                if self.vf[i][j] > 0:
                    if view[i][j] == self.desired_shape[i][j]:
                        reward += 0.1
                    else:
                        reward -= 0.1
                        exact_match = False
        if exact_match:
            reward = 10
            print("EXACT MATCH")
        return reward
            
    
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
            
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: torch.cat(x, 0), samples)
    
    
class Vision:
    def __init__(self, env_data, vf, eye_loc):
        # Game environment data of cell states; numpy array of rank 2
        self.environment_data = env_data
        # Shape of the agent's field of view; numpy array of rank 2
        self.visual_field = vf
        # Coordinates of the agent's eye in the environment grid; 
        # rank 1 numpy array of size 2
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
        #self.update_view()

    def update(self, data):
        self.environment_data = data
        self.update_view()
        
    def update_view(self):
        # Update the agent's view and its simplified form, viewdata.
        # View displays None if the point is not part of the visual field.
        # Viewdata only consists of values for the points in the visual field,
        # but any of those points that are off the edge of the environment
        # grid will be denoted as None. The size and shape for both are always
        # the same for any given visual field template. 
        #
        #  - Example of view:
        # np.array([[None, None,   1, None, None],
        #           [None,    0,   1,    1, None],
        #           [   1,    1,   0,    0,    0],
        #           [None,    0,   1,    0, None],
        #           [None, None,   1, None, None]])
        #
        #  - Example of its corresponding viewdata:
        #   [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1]
        #  
        e = self.environment_data
        eye_x, eye_y = self.eye_location
        vf = self.visual_field
        view = np.ones(vf.shape)
        viewdata = []
        # Find local eye location in visual field template
        for i in range(len(vf)):
            for j in range(len(vf[i])):
                if vf[i][j] == 2:
                    loc_eye_x, loc_eye_y = i, j
        # Get environment data within the visual field
        for i in range(len(vf)):
            for j in range(len(vf[i])):
                x = i + eye_x - loc_eye_x
                y = j + eye_y - loc_eye_y
                if vf[i][j] > 0: # If within visual field
                    if 0 <= x < len(e) and 0 <= y < len(e[0]): # If in bounds
                        view[i][j] = e[x][y]
                        viewdata.append(e[x][y])
                    else: # If within visual field but out of bounds
                        view[i][j] = None
                        viewdata.append(None)
                else:
                    view[i][j] = None
        self.view = view
        self.viewdata = viewdata
        return view
                        