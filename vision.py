#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 09:31:16 2020

@author: JamesButcher
  
"""

import numpy as np

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

