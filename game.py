"""
"A Demon in Conway's Game of Life"
 by James Butcher
 Github: https://github.com/jmsbutcher/A-demon-in-conways-game-of-life
 First created: 8-31-19
 
 Current version: 0.7
 Created: 10-7-19
  
 Python version 3.7.3
 Pytorch version 1.2.0

"""
# Import the "demon" (the intelligent agent) from demon.py
from demon import Agent, RewardScheme

import random
from random import randint
import time
import numpy as np
from tkinter import Tk, Frame, Button, Menu, Label, Canvas, BOTH, BOTTOM, RIGHT, LEFT
from PIL import Image, ImageTk

# Environment shapes; used for making reward schemes
glider = np.array([[1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 0, 1, 1, 1],
                   [1, 1, 1, 1, 0, 1, 1],
                   [1, 1, 0, 0, 0, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1]])

# Visual field shapes
#   0 - Not part of the visual field
#   1 - Part of the visual field
#   2 - The agent's eye location
extralarge_vf = np.array( \
    [[0, 0, 1, 1, 1, 1, 1, 0, 0],
     [0, 1, 1, 1, 1, 1, 1, 1, 0],
     [1, 1, 1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 2, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 1],
     [0, 1, 1, 1, 1, 1, 1, 1, 0],
     [0, 0, 1, 1, 1, 1, 1, 0, 0]])
large_vf = np.array( \
    [[0, 0, 1, 1, 1, 0, 0],
     [0, 1, 1, 1, 1, 1, 0],
     [1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 2, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1],
     [0, 1, 1, 1, 1, 1, 0],
     [0, 0, 1, 1, 1, 0, 0]])
medium_vf = np.array( \
    [[0, 0, 1, 1, 0, 0],
     [0, 1, 1, 1, 1, 0],
     [1, 1, 1, 1, 1, 1],
     [1, 1, 2, 1, 1, 1],
     [0, 1, 1, 1, 1, 0],
     [0, 0, 1, 1, 0, 0]])
small_vf = np.array( \
    [[0, 0, 1, 0, 0],
     [0, 1, 1, 1, 0],
     [1, 1, 2, 1, 1],
     [0, 1, 1, 1, 0],
     [0, 0, 1, 0, 0]])


class MainWindow(Frame):
    def __init__(self, master=None, size=(50, 50), scale=10):
        Frame.__init__(self, master)
        self.master = master
        self.size = size        # (x,y) dimensions of rectangular cell grid
        self.scale = scale      # pixel size of each cell---10 by default
        self.view_scale = 15    # pixel size of agent view in display console
        self.data = np.ones(self.size, dtype="uint8")  # initialize cell states
        self.running = False    # game is paused until start_game() is called
        self.interval = 0.51    # time between steps; how fast the game runs
        # Beginning location of demon's eye:
        self.eye_location = np.array((self.size[0]//2, self.size[1]//2))
        # Select shape of the demon's visual field from the list above:
        self.vis_field = large_vf  
        # Create glider reward scheme (for testing)
        self.glider_reward_scheme = RewardScheme(self.vis_field,
                                                 shape_name="Glider",
                                                 desired_shape=glider)
        #self.glider_reward_scheme_display = 
        
        # Create agent object (the demon)
        self.agent = Agent(self.data, 
                           self.vis_field, 
                           self.eye_location, 
                           gamma=0.9)
        
        self.last_reward = 0
        self.agent_speed = 10 # How many times the demon updates every game step
        self.wait = False     # Allows the demon to wait until the next step
        
        # Calculate size of app window
        win_X = self.size[1] * self.scale + 188
        win_Y = self.size[0] * self.scale + 50
        self.master.geometry("{width}x{height}".format(width=win_X, height=win_Y))
        self.master.minsize(win_X, win_Y)
        
        # Right Frame, Display Console: for displaying demon's readouts
        self.display_console = Frame(self.master, borderwidth=5,
                                     relief="ridge", width=200, bg="gray")
        self.display_console.pack(fill=BOTH, side=RIGHT, expand=1)
        
        # Initialize and display a close-up of what the demon currently sees
        self.agentview_display = Label(self.display_console, image=None)
        self.agentview_display.grid(row=0, columnspan=2)
        self.display_agent_view()
        
        # Initialize and display meter readings in display console
        reward_label = Label(self.display_console, text="Reward:", 
                             #relief="groove", bg="#dddddd")
                             fg="white", bg="gray")
        reward_label.grid(row=1, column=0, padx=4, pady=7, sticky="w")
        self.reward_meter = Canvas(self.display_console, width=100, height=15)
        self.reward_meter.grid(row=1, column=1)
        self.reward_meter_level = self.reward_meter.create_rectangle( \
                                  5, 5, 15, 15, width=2, fill="black")
        
        # Initialize and display reward scheme in display console
        reward_scheme_label = Label(self.display_console, 
                                    text="\nCurrent reward scheme:", 
                                    fg="white", bg="gray")
        reward_scheme_label.grid(row=2, columnspan=2, pady=10, sticky="s")
        reward_scheme_name = Label(self.display_console, 
                                    text="Produce this shape:", 
                                    fg="white", bg="gray")
        reward_scheme_name.grid(row=3, columnspan=2, pady=10, sticky="s")        
        self.reward_scheme_view = Label(self.display_console, image=None)
        self.reward_scheme_view.grid(row=4, columnspan=2)
        self.display_reward_scheme()

        # Bottom Frame: for game control buttons
        button_menu = Frame(self.master, bg="#444444", 
                            relief="ridge", borderwidth=5)
        button_menu.pack(fill=BOTH, side=BOTTOM, expand=0)  
        
        quit_button = Button(button_menu, text="Quit", command=self.master.quit)
        seed_button = Button(button_menu, text="Seed", command=self.seed)
        random_button = Button(button_menu, text="Random", command=self.randomize)
        clear_button = Button(button_menu, text="Clear", command=self.clear)
        start_button = Button(button_menu, text="Start", command=self.start_game)
        step_button = Button(button_menu, text="Step", command=self.step)
        stop_button = Button(button_menu, text="Stop", command=self.stop_game)
        speed_up_button = Button(button_menu, text="Speed Up", command=self.speed_up)
        slow_down_button = Button(button_menu, text="Slow Down", command=self.slow_down)
        
        quit_button.pack(side=LEFT, padx=10, pady=5)
        seed_button.pack(side=LEFT, padx=2)
        random_button.pack(side=LEFT, padx=2)
        clear_button.pack(side=LEFT, padx=2)
        start_button.pack(side=LEFT, padx=2)
        step_button.pack(side=LEFT, padx=2)
        stop_button.pack(side=LEFT, padx=2)
        speed_up_button.pack(side=LEFT, padx=2)
        slow_down_button.pack(side=LEFT, padx=2)
        
        # Environment Frame: the main cell grid
        self.environment = Frame(self.master, 
                                 width=win_X, 
                                 height=win_Y, 
                                 bg="black")
        self.environment.pack(fill=BOTH, side=LEFT, expand=1)
        
        # Initialize and display the environment cells
        self.env_img = Label(self.environment, image=None)
        self.env_img.place(x=0, y=0)
        self.display_data()
        
        # Menus
        menu = Menu(self.master)
        self.master.config(menu=menu)
        
        file = Menu(menu)
        file.add_command(label="Exit", command=self.master.quit)
        file.add_command(label="Save", command=self.save)
        file.add_command(label="Load", command=self.load)
        
        new = Menu(menu)
        new.add_command(label="Random", command = self.randomize)
        new.add_command(label="Seed", command = self.seed)
        new.add_command(label="Gun", command = self.gun)
        
        run = Menu(menu)
        run.add_command(label="Start", command = self.start_game)
        run.add_command(label="Step", command = self.step)
        run.add_command(label="Stop", command = self.stop_game)
        run.add_command(label="Clear", command = self.clear)
        run.add_command(label="Speed Up", command = self.speed_up)
        run.add_command(label="Slow Down", command = self.slow_down)
        
        menu.add_cascade(label="File", menu=file)
        menu.add_cascade(label="New", menu=new)
        menu.add_cascade(label="Run", menu=run)
    
    def act(self, a):
        # Make demon take an action based on int parameter <a>:
        # 0 --> Wait until next step
        # 1 --> Move up
        # 2 --> Move right
        # 3 --> Move down
        # 4 --> Move left
        # 5 --> Flip cell
        loc = self.agent.vision.absolute_eye_location # Demon's (x, y) location
        if a == 0:
            self.wait = True                # ; print("Wait", end=" ")
        elif a == 1 and loc[0] < size[0]-1:
            self.agent.vision.move(1, 0)    # ; print("Up", end=" ")
        elif a == 2 and loc[0] > 0:
            self.agent.vision.move(-1, 0)   # ; print("Down", end=" ")
        elif a == 3 and loc[1] < size[1]-1:
            self.agent.vision.move(0, 1)    # ; print("Left", end=" ")
        elif a == 4 and loc[1] > 0:
            self.agent.vision.move(0, -1)   # ; print("Right", end=" ")
        elif a == 5:
            self.flip_cell()                # ; print("Flipped", end=" ")
            
    def clear(self):
        blank_data = np.ones(self.size, dtype="uint8")
        self.data = blank_data
        self.display_data()
        self.display_agent_view()
        
    def conway_rule(self, x, y):
        # Get state of cell at coordinates (x, y) and those of its neighbors
        state = self.data[x][y]
        neighbors = [self.data[x-1][y-1] , self.data[x][y-1] , self.data[x+1][y-1],
                     self.data[x-1][y]   , 0                , self.data[x+1][y],
                     self.data[x-1][y+1] , self.data[x][y+1] , self.data[x+1][y+1]]
        neighbor_total = sum(neighbors)
        
        # Calculate the resulting state of the cell according to the rules
        if state == 0:   # if cell is alive...
            if neighbor_total > 6:   return 1   # dies from underpopulation
            elif neighbor_total > 4: return 0   # remains alive
            else:                    return 1   # dies from overpopulation
            # if cell is dead, but has exactly 3 live neighbors...
        elif state == 1 and neighbor_total == 5: return 0   # comes to life
        else: return state

    def display_agent_view(self):
        # Update demon view and display it in the display console
        self.agent.vision.environment_data = self.data
        self.agent.vision.update_view()
        view = self.agent.vision.get_view()
        colorweighted_data = view * 255
        self.scale_render_place(colorweighted_data, 
                                self.view_scale,
                                self.agentview_display)

    def display_data(self):
        # Display the environment data in the main window
        colorweighted_data = self.data * 255
        # Display white cells in demon's visual field as gray (255 --> 204)
        for p in self.agent.vision.absolute_visual_field_points:
            if 0 <= p[0] < self.size[0] and 0 <= p[1] < self.size[1] and \
                    self.data[tuple(p)] == 1:
                colorweighted_data[p[0]][p[1]] = 200
        if self.data[tuple(self.eye_location)] == 1:
            colorweighted_data[tuple(self.eye_location)] = 255 # Make eye location white
        self.scale_render_place(colorweighted_data, self.scale, self.env_img)
        
    def display_reward_scheme(self):
        colorweighted_data = self.glider_reward_scheme.desired_shape * 255
        for i in range(len(self.vis_field)):
            for j in range(len(self.vis_field[i])):
                if self.vis_field[i][j] == 1 and colorweighted_data[i][j] == 255:
                    colorweighted_data[i][j] = 200
        self.scale_render_place(colorweighted_data, self.view_scale,
                                self.reward_scheme_view)
  
    def flip_cell(self, chance=1):
        # When called, agent has a 1 in <chance> chance of flipping the
        # cell at its current location from 1 to 0 or vice versa.
        loc = tuple(self.eye_location)
        chance = random.randint(1, chance)
        if chance == 1:
            self.data[loc] = abs(self.data[loc] - 1)    # flip cell
            
    def get_reward(self):
        
        #return self.glider_reward_scheme.calc_reward(self.agent.vision.get_view())
        

        # Calculate reward based on how many live cells are in the visual field
        r = 0
        for cell in self.agent.vision.viewdata:
            if cell == 0:
                r += 1
        reward = r   
        return reward

        
    def gun(self):
        # Initialize a "Gosper's glider gun"
        gun_data = np.ones(self.size, dtype="uint8")
        x = 5
        y = 5
        gun_data[x][y+25] = 0
        gun_data[x+1][y+23:y+26] = [0, 1, 0]
        gun_data[x+2][y+13:y+37] = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
        gun_data[x+3][y+12:y+37] = [0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
        gun_data[x+4][y+1:y+23]  = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0]
        gun_data[x+5][y+1:y+26]  = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0]
        gun_data[x+6][y+11:y+26] = [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0]
        gun_data[x+7][y+12:y+17] = [0, 1, 1, 1, 0]
        gun_data[x+8][y+13:y+15] = [0, 0]
        self.data = gun_data
        self.display_data()
        self.display_agent_view()
        
    def load(self):
        print("Loading last saved brain...")
        self.agent.load()
        
    def move_chance(self, chance=10):
        # When called, demon has a 1 in <chance> chance of moving in one
        # of four directions.
        # Increasing <chance> decreases the chance of moving.
        loc = self.agent.vision.absolute_eye_location
        chance = random.randint(0, chance)
        if chance == 1 and loc[0] < size[0]-1:
            self.agent.vision.move(1, 0)
        elif chance == 2 and loc[0] > 0:
            self.agent.vision.move(-1, 0)
        elif chance == 3 and loc[1] < size[1]-1:
            self.agent.vision.move(0, 1)
        elif chance == 4 and loc[1] > 0:
            self.agent.vision.move(0, -1)
        
    def randomize(self):
        # Initialize a random state on the whole grid
        self.data = np.random.randint(0, 2, self.size, dtype="uint8")        
        self.display_data()
        self.display_agent_view()
        
    def save(self):
        print("Saving brain...")
        self.agent.save()
        print("Brain saved.")
        
    def scale_render_place(self, colorweighted_data, scale, placement_label):
        # Take a rank 2 numpy array of gray-colorweighted data (each
        #   value from 0 - 255), scale it according to the integer provided,
        #   render it, and place it into the Label widget provided.
        # Example (all arrays are numpy arrays):
        # colorweighted_data:                scaled_data:
        #   [[255, 200],   scale 2      [[255, 255, 200, 200],
        #    [0  , 255]]    --->         [255, 255, 200, 200],
        #                                [0,   0,   255, 255],
        #                                [0,   0,   255, 255]]
        scaled_data = np.kron(colorweighted_data, np.ones((scale, scale)))
        raw_img = Image.fromarray(scaled_data)
        render = ImageTk.PhotoImage(raw_img)
        placement_label.configure(image = render)
        placement_label.image = render
    
    def seed(self):
        # Initialize a random state in the center 8th of the grid
        seed_data = np.ones(self.size, dtype="uint8")
        dx = self.size[0] // 8
        midx = self.size[0] // 2
        dy = self.size[1] // 8
        midy = self.size[1] // 2
        for i in range(midx - dx, midx + dx):
            for j in range(midy - dy, midy + dy):
                r = randint(0, 1)
                seed_data[i][j] = r
        self.data = seed_data
        self.display_data()
        self.display_agent_view()
        
    def slow_down(self):
        self.interval += .1
        
    def speed_up(self):
        if self.interval > 0.1:
            self.interval -= 0.1
        else:
            self.interval = 0.01
        
    def start_game(self):
        self.running = True
        while self.running:
            self.step()
            
    def step(self):
        # Apply the Conway game rules to all the cells
        newdata = np.ones(self.size, dtype="uint8")
        for x in range(1, self.size[0]-1):
            for y in range(1, self.size[1]-1):
                result = self.conway_rule(x, y)
                newdata[x][y] = result
        self.data = newdata
        self.display_data()
        self.display_agent_view()
        
        # Update the demon X number of times --- set by self.agent_speed
        for i in range(self.agent_speed):
            self.update_agent()
            self.update()
            self.display_data()
            self.display_agent_view()
            time.sleep(self.interval / self.agent_speed)
            if self.wait == True:
                break
        self.wait = False
        
        self.update()
        time.sleep(self.interval)
  
    def stop_game(self):
        self.running = False

    def update_agent(self):
        view = self.agent.vision.viewdata
        for v in range(0, len(view)):
            if view[v] is None:     # Treat cells off the edge of the grid as
                view[v] = 1         # dead for learning and acting purposes
        reward = self.get_reward()
        action = self.agent.update(reward, view)
        self.act(action)
        self.last_reward = reward
        self.update_meters()
        
    def update_meters(self):        
        reward_meter_scale = 5
        x0, y0, x1, y1 = self.reward_meter.coords(self.reward_meter_level)
        x1 = reward_meter_scale * self.last_reward
        self.reward_meter.coords(self.reward_meter_level, x0, y0, x1, y1)        
        if x1 > 80: 
            color = "green yellow"
        elif x1 > 60: 
            color = "yellow2"
        elif x1 > 40:
            color = "orange"
        elif x1 > 20:
            color = "red"
        else:
            color = "black"
        self.reward_meter.itemconfig(self.reward_meter_level, fill=color)


root = Tk()
root.title("A Demon In Conway's Game Of Life")
        
size = (50, 50)     # Default: (50, 50)
scale = 10          # Default: 10 

main_window = MainWindow(root, size, scale)

root.mainloop()