#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 10:56:35 2020

@author: JamesButcher

 An alternative AI agent that lives in the Conway's Game of Life game 
 environment, "game.py"

 Scamp version: 1.0
 Date: 6-28-2020

 - A prototype for my ideas on a new kind of artificial intelligence


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

import numpy as np
import random
from vision import Vision

def generate_random_state(size=(7, 7), percentage=10):
    state = np.ones((size[0] * size[1], 1), dtype=int)
    for i in range(len(state)):
        chance = random.randint(0, 100)
        if chance < percentage:
            state[i] = 0
    state = state.reshape(size)
    return state

def compare_states(state1, state2):
    if state1.shape != state2.shape:
        print(state1.shape)
        print(len(state1))
        print(len(state1[0]))
        print("ERROR: cannot compare two states of different size: {} and {}"\
              .format(state1.shape, state2.shape))
        return state1.size
    num_differences = 0
    for i in range(len(state1)):
        for j in range(len(state1[0])):
            if state1[i][j] != state2[i][j]:
                num_differences += 1
    return num_differences

#import matplotlib.pyplot as plt
#test_size = (5, 6)
#
#standard = generate_random_state(test_size)
#wrong_size_test = generate_random_state((9, 9))
#compare_states(standard, wrong_size_test)
#
#nums = []
#
#for i in range(10000):
#    s = generate_random_state(test_size)
##    print(s)
##    print(standard)
#    num = compare_states(s, standard)
##    print(num)
#    nums.append(num)
#    if num == 0:
#        print("#############################################\n"
#              "#                Exact Match                #\n"
#              "#############################################")
#    elif num == 1:
#        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#        
#plt.hist(nums, bins=max(nums))
#plt.show()


class ScampAgent:
    
    def __init__(self, env_data, vf, abs_eye_loc):
        
        self.vision = Vision(env_data, vf, abs_eye_loc)
        
        self.goals = []
        
        self.theories = []
        
#    def check_match(self, view, i, j, target):
#        """ Check whether the target state matches the sub-view at i, j."""
#        x_size = len(target)
#        y_size = len(target[0])
#        matches = 0
#        for x in range(0, x_size):
##            print()
#            for y in range(0, y_size):
##                print(view[i + x][j + y], end=" ")
#                if target[x][y] == view[i + x][j + y]:
#                    matches += 1
##        print()            
##        print("Num matches:", matches)            
#        if matches == x_size * y_size:
#            return True
#        else:
#            return False
        
    def initialize_goals(self, number=5):
        pass
        
    def save(self, file):
        pass
    
    def scan(self, target):
        """ Scan over the agent's current view for a given pattern """
        view = self.vision.get_view()
        print("View:\n{}".format(view))
        
        x_len = len(target)
        y_len = len(target[0])
        for i in range(len(view) - x_len + 1):
            for j in range(len(view[0]) - y_len + 1):
#                if self.check_match(view, i, j, target):
                print("View slice:\n{}".format(view[i:i+x_len, j:j+y_len]))
                if compare_states(view[i:i+x_len, j:j+y_len], target) == 0:
                    print("Found")
                    return i, j
        print("Not found")
                
    
    def search_theories_for_goal(self):
        pass
        
    def update(self, reward, viewdata):
        
#        test_target = np.array([[1, 1, 1],
#                                [0, 0, 0],
#                                [1, 1, 1]])
        test_target = np.array([[1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1],
                                [1, 0, 0, 0, 1],
                                [1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1]])
        
        self.scan(test_target)
        
        
        action = random.choice(range(0, 6))
        
        return action

    
    
class Goal:
    """ A target state of given size with an associated interest level.
    
        A new goal is either initialized by supplying a definite 2-D numpy 
        array (shape must match required [size] argument), or, if omitted, 
        generate a random 2-D numpy array of size [size], which is a tuple
        defining the number of rows and columns: (x, y).
        
        A new goal starts in an "active" state, with an interest_level of 100.
        If the interest_level drops below 1, the goal deactivates.
        
        Example:  goal1 = Goal((4, 5))  -->   np.array[[0, 0, 1, 1, 0],
                                                       [0, 1, 0, 0, 0],
                                                       [0, 0, 0, 1, 0],
                                                       [1, 0, 0, 0, 0],]
    """
    def __init__(self, size, state=None, interest_level=100):
        self.size = size
        if state:
            if size == state.shape:
                self.state = state
            else:
                print("ERROR: Goal size and initial state size do no match")
        else:
            self.state = generate_random_state(size)
        self.interest_level = interest_level
        self.active = True
        
    def lose_interest(self):
        self.interest_level -= 5
        if self.interest_level <= 0:
            self.interest_level = 0
            self.active = False
            
    def gain_interest(self):
        self.interest_level += 5
            
        
    
        
        
class Theory:
    """ A sequence of states (numpy 2-D arrays of size [size]) """
    def __init__(self, size, first_state=None):
        self.size = size
        if first_state:
            self.first_state = first_state,
        else:
            self.first_state = generate_random_state(size)
        self.symbol = generate_random_state()
        self.sequence = [self.first_state]
        
    def __str__(self):
        print("Theory Symbol:")
        for row in self.symbol:
            for cell in row:
                print(cell, end=" ")
            print()
        print()
                
        for i, state in enumerate(self.sequence):
            for row in state:
                for cell in row:
                    print(cell, end=" ")
                print()
            if i < len(self.sequence) - 1:
                print("\n"
                      "  |  \n"
                      " \\|/ \n"
                      "  V  #{}\n".format(i + 1))
        return ""
        
    def add_state(self, state):
        self.sequence.append(state)
        
    def search(self, target_state):
        for index in range(len(self.sequence)):
            if compare_states(self.sequence[index], target_state) == 0:
                return index
        return -1
    
    
    
    
#test_size_2 = (4, 5)
#test_theory = Theory(test_size_2)
#for i in range(10):
#    test_theory.add_state(generate_random_state(test_size_2))
#print(test_theory)
#
#target = generate_random_state(test_size_2)
#print("Target:\n", target)
#print(test_theory.search(target))

    

    
    
