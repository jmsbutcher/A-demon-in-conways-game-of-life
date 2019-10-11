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
    def __init__(self, env_data, vf, abs_eye_loc, vision_size, gamma):
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
    def __init__(self, env_data, vf, abs_eye_loc):
        # Game environment data of cell states; numpy array of rank 2
        self.environment_data = env_data
        
        # Shape of the agent's field of view; numpy array of rank 2
        self.visual_field = vf
        
        # Coordinates of the agent's eye in the environment grid; 
        # rank 1 numpy array of size 2
        self.absolute_eye_location = abs_eye_loc
        
        # List of coordinates of visual field points relative to  
        # the eye location; list of rank 1 numpy arrays of size 2
        self.relative_visual_field_points = self.get_visual_field_points( \
                                                self.visual_field)[0]
        
        # Coordinates of the agent's eye in the visual field; 
        # rank 1 numpy array of size 2
        self.relative_eye_location = self.get_visual_field_points( \
                                                self.visual_field)[1]
        
        # List of coordinates of visual field points in 
        # the environment grid; list of tuples of size 2
        self.absolute_visual_field_points = \
                        self.get_absolute_visual_field_points( \
                                    self.relative_visual_field_points, 
                                    self.absolute_eye_location)  
                        
        # List of values representing the cell states in the visual field;
        # "0" for alive, "1" for dead, "None" for out of bounds
        self.viewdata = self.get_environment_data_points_in_visual_field( \
                                    self.environment_data, 
                                    self.absolute_visual_field_points)
        
        # Representation of the agent's view of the cell states in 
        # its visual field; numpy array of rank 2
        self.view = self.get_view()
        
    def get_visual_field_points(self, vf):
        # Take in a rank 2 numpy array with:
        #  - "0"s marking cells not in the visual field
        #  - "1"s marking cells in the visual field
        #  - A single "2" marking the agent's eye location
        # Return two things:
        #  - A list of rank 1 numpy arrays of size 2 holding coordinates of  
        #    all the "1"s relative to the eye location
        #  - A numpy array of size 2 holding coordinates of 
        #    the agent's eye location
        # Example: (all sequences below are numpy arrays)
        #       vf:                    vf_points:                    eye_loc:
        #  [[0, 1, 0],  
        #   [1, 2, 1],  -->  [(0, 1), (1, 0), (1, 2), (2, 1)]   ,     (1, 1)
        #   [0, 1, 0]]
        vf_points = []
        eye_loc = np.array((0, 0))
        for i in range(vf.shape[0]):
            for j in range(vf.shape[1]):
                if vf[i][j] == 1 or vf[i][j] == 2:
                    vf_points.append(np.array((i, j)))
                if vf[i][j] == 2:
                    eye_loc = np.array((i, j))
        # Turn list of visual field points into points relative to eye location
        # Example:
        #           vf_points:                         rel_vf_points:   
        #  [(0, 1), (1, 0), (1, 2), ...] --> [(-1, 0), (0, -1), (0, 1), ...]
        rel_vf_points = []
        for point in vf_points:
            rel_point = point - eye_loc
            rel_vf_points.append(rel_point)
        return rel_vf_points, eye_loc

    def get_absolute_visual_field_points(self, rel_vf_points, abs_eye_loc):
        # Take numpy array of visual field coordinates relative to the
        # agent's eye location in its visual field, and change it to a numpy 
        # array of absolute coordinates relative to the (0, 0) point in the 
        # environment.
        # Example:
        # eye location in environment: (4, 4)
        #           rel_vf_points:                       abs_vf_points:
        # [(-1, 0), (0, -1), (0, 1), ...]  -->  [(3, 4), (4, 3), (4, 5), ...]
        absolute_vis_field_points = []
        for point in rel_vf_points:
            absolute_vis_field_points.append(abs_eye_loc + point)
        return absolute_vis_field_points
    
    def get_environment_data_points_in_visual_field(self, data, abs_vf_points):
        # Take the absolute visual field coordinates of all the points in
        # the agent's field of view and return a list of the environment 
        # values in that field of view. If any visual field coordinates are off
        # the edge of the environment window, add <None> to this list.
        # Example:
        # agent location: (4, 4)
        #           abs_vf_points:                       view_data:
        # [(3, 4), (4, 4), (4, 5), ...]  -->  [0, 1, 1, 0, None, None, 1, ... ]
        view_data = []
        for point in abs_vf_points:
            if  point[0] < 0 or point[0] >= len(data) or \
                point[1] < 0 or point[1] >= len(data[0]):
                view_data.append(None)
            else:
                view_data.append(data[point[0]][point[1]])
        return view_data
    
    def get_view(self):
        # Turn list of 0s and 1s in viewdata into a numpy array of 0s and 1s in
        # the shape of the visual field
        # This 2D array will be used to make a view of what the agent
        # sees and display it in the display console.
        vf = self.visual_field
        view = np.ones(vf.shape, dtype="float")   
        d = self.viewdata
        index = 0
        
        for i in range(len(vf)):
            for j in range(len(vf[0])):
                if vf[i][j] == 0:
                    view[i][j] = 1   # White - Outside the visual field. 
                elif vf[i][j] == 1:
                    if d[index] == 0:
                        view[i][j] = 0   # Black - Live cell in visual field
                    elif d[index] == 1:
                        view[i][j] = 0.8   # Gray - Dead cell in visual field
                    index += 1
                else: 
                    view[i][j] = self.environment_data[self.absolute_eye_location[0], self.absolute_eye_location[1]]
                    index += 1
        return view
    
    def move(self, x_shift, y_shift):
        self.absolute_eye_location[0] += x_shift
        self.absolute_eye_location[1] += y_shift
    
    def update_view(self):
        # Update what the agent sees: the state of the cells in its 
        # visual field, which depends on its location and the environment data
        self.absolute_visual_field_points = \
            self.get_absolute_visual_field_points( \
                self.relative_visual_field_points, self.absolute_eye_location)
        self.viewdata = self.get_environment_data_points_in_visual_field( \
                self.environment_data, self.absolute_visual_field_points)
        return self.viewdata
        
    def print_visual_field(self):
        # Print the shape of the visual field to the terminal using symbols
        print("Visual field: ")
        for row in self.visual_field:
            for item in row:
                if item == 1:
                    print("*", end=" ")
                elif item == 2:
                    print("@", end=" ")
                else:
                    print(" ", end=" ")
            print()
            
    def print_agent_view(self):
        # Print what the agent currently sees in its visual field
        # to the terminal using symbols
        print("Agent view: ")
        vf = self.visual_field
        env_view_points = self.viewdata
        index = 0
        for i in range(len(vf)):
            for j in range(len(vf[0])):
                if vf[i][j] == 0:
                    print(" ", end=" ")
                elif vf[i][j] == 1:
                    d = env_view_points[index]
                    if d == 1:
                        print("-", end=" ")
                    elif d == 0:
                        print("#", end=" ")
                    else:
                        print(" ", end=" ")
                    index += 1
                elif vf[i][j] == 2:
                    print("@", end=" ")
            print()
            
    def print_full_evironment_view(self):
        # Print entire environment with:
        #   - "0" representing live cells,
        #   - "1" representing dead cells,
        #   - Brackets "[ ]" surrounding cells in the agent's visual field, and
        #   - Angle brackets "< >" surrounding the cell at the eye location
        data = self.environment_data
        abs_vf_points = self.get_absolute_visual_field_points( \
            self.get_visual_field_points(self.visual_field)[0], 
                                         self.absolute_eye_location)
        abs_eye_loc = self.absolute_eye_location        
        print("Full environment view: ")
        coordinates = []
        for p in abs_vf_points:
            coordinates.append(list(p))
        for i in range(len(data)):
            for j in range(len(data[0])):
                if abs_eye_loc[0] == i and abs_eye_loc[1] == j:
                    print("<{}>".format(data[i][j]), end="")
                elif [i, j] in coordinates:
                    print("[{}]".format(data[i][j]), end="")
                else:
                    print("", data[i][j], end=" ")
            print()    
