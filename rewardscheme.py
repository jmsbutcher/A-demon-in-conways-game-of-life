#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 09:22:39 2020

@author: JamesButcher
       
"""

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
                        if self.vf[i][j] > 0:
                            if desired_shape[i][j] == 0:
                                print(" #", end="")
                            else:
                                print("  ", end="")
                        else:
                            print(" O", end="")
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


