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

from tkinter import Tk, Frame, Button, Checkbutton, Menu, Label, Canvas, \
                    BOTH, BOTTOM, RIGHT, LEFT, Toplevel, Entry, filedialog
                    
from PIL import Image, ImageTk

# Environment shapes; used for making reward schemes
glider = np.array([[1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 0, 1, 1, 1],
                   [1, 1, 1, 1, 0, 1, 1],
                   [1, 1, 0, 0, 0, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1]])

flipper = np.array([[1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 0, 0, 0, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
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
        self.interval = 0.5    # time between steps; how fast the game runs
        self.generation = 0     # how many steps (generations) has passed
        # Beginning location of demon's eye:
        self.eye_location = np.array((self.size[0]//2, self.size[1]//2))
        # Select shape of the demon's visual field from the list above:
        self.vis_field = large_vf  
        # Create glider reward scheme (for testing)
        self.match_count = 0
        self.reward_scheme = RewardScheme(self.vis_field,
                                                 schemetype="shape",
                                                 shape_name="Flipper",
                                                 desired_shape=flipper)
        
        # Create agent object (the demon)
        self.agent = Agent(self.data, 
                           self.vis_field, 
                           self.eye_location, 
                           gamma=0.9)
        self.agent.vision.update(self.data)
        self.agent_enabled = False
        self.manual_mode = False
        
        self.last_reward = 0
        self.agent_speed = 10 # How many times the demon updates every game step
        self.wait = False     # Allows the demon to wait until the next step
        
        # Calculate size of app window
        win_X = self.size[1] * self.scale + 193
        win_Y = self.size[0] * self.scale + 101
        self.master.geometry("{width}x{height}".format(width=win_X, height=win_Y))
        #self.master.minsize(win_X, win_Y)
        self.master.minsize(max(win_X, 690), max(win_Y, 551))
        
        # Right Frame, Display Console: for displaying demon's readouts
        self.display_console = Frame(self.master, borderwidth=5,
                                     relief="ridge", width=200, bg="gray")
        self.display_console.pack(fill=BOTH, side=RIGHT, expand=1)
        
        # Initialize and display generation count
        generation_label = Label(self.display_console, text="Gen #:",
                                      fg="white", bg="gray")
        generation_label.grid(row=0, column=0)
        self.generation_count = Label(self.display_console, 
                                      text=str(self.generation),
                                      font=("Arial", 20), 
                                      fg="white", bg="gray")
        self.generation_count.grid(row=0, column=1, sticky="w")
        
        # Initialize and display a close-up of what the demon currently sees
        self.agentview_display = Label(self.display_console, image=None)
        self.agentview_display.grid(row=1, columnspan=2)
        self.display_agent_view()
        
        # Initialize and display meter readings in display console
        reward_label = Label(self.display_console, text="Reward:", 
                             #relief="groove", bg="#dddddd")
                             fg="white", bg="gray")
        reward_label.grid(row=2, column=0, padx=4, pady=7, sticky="w")
        self.reward_meter = Canvas(self.display_console, width=100, height=15)
        self.reward_meter.grid(row=2, column=1)
        self.reward_meter_level = self.reward_meter.create_rectangle( \
                                  5, 5, 6, 15, width=1, fill="black")
        
        # Initialize and display reward scheme in display console
        reward_scheme_label = Label(self.display_console, 
                                    text="\nCurrent reward scheme:", 
                                    fg="white", bg="gray")
        reward_scheme_label.grid(row=4, columnspan=2, pady=10, sticky="s")
        reward_scheme_name = Label(self.display_console, 
                                    text="Produce this shape:",
                                    fg="white", bg="gray")
        reward_scheme_name.grid(row=5, columnspan=2, pady=10, sticky="s")        
        self.reward_scheme_view = Label(self.display_console, image=None)
        self.reward_scheme_view.grid(row=6, columnspan=2)
        self.display_reward_scheme()
        
        self.message_box = Frame(self.display_console, 
                                 borderwidth=2,
                                 relief=None, 
                                 #width=180,
                                 #height=180,
                                 bg="gray")
        self.message_box.grid(row=7, columnspan=2, pady= 10)

        # Bottom Frame: for game control buttons
        button_menu = Frame(self.master, bg="#DDDDDD", 
                            relief="ridge", borderwidth=5)
        button_menu.pack(fill=BOTH, side=BOTTOM, expand=0)  
        
        quit_button = Button(button_menu, text="Quit", command=self.quit_game)
        seed_button = Button(button_menu, text="Seed", command=self.seed)
        random_button = Button(button_menu, text="Random", 
                               command=self.randomize)
        clear_button = Button(button_menu, text="Clear", command=self.clear)
        start_button = Button(button_menu, text="Start", 
                              command=self.start_game)
        step_button = Button(button_menu, text="Step", command=self.step)
        stop_button = Button(button_menu, text="Stop", command=self.stop_game)
        speed_up_button = Button(button_menu, text="Speed Up", 
                                 command=self.speed_up)
        slow_down_button = Button(button_menu, text="Slow Down", 
                                  command=self.slow_down)
        toggle_agent_button = Checkbutton(button_menu, text="Enable Demon",
                                          bg="#DDDDDD",
                                          command=self.toggle_agent)
        self.manual_mode_button = Checkbutton(button_menu, text="Manual Mode",
                                          bg="#DDDDDD",
                                          bd=3,
                                          variable=1,
                                          command=self.toggle_manual_mode)

        quit_button.grid(row=0, column=1, padx=10, pady=2)
        seed_button.grid(row=0, column=2, padx=1)
        random_button.grid(row=0, column=3, padx=1)
        clear_button.grid(row=0, column=4, padx=1)
        start_button.grid(row=0, column=5, padx=1)
        step_button.grid(row=0, column=6, padx=1)
        stop_button.grid(row=0, column=7, padx=1)
        speed_up_button.grid(row=0, column=8, padx=1)
        slow_down_button.grid(row=0, column=9, padx=1)
        toggle_agent_button.grid(row=1, columnspan=4, sticky="w", 
                                 padx=3, pady=3)
        self.manual_mode_button.grid(row=2, columnspan=4, sticky="w", 
                                padx=3, pady=3)
        
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
        file.add_command(label="Exit", command=self.quit_game)
        file.add_command(label="Save", command=self.save)
        file.add_command(label="Load", command=self.load)
        file.add_command(label="Settings", command=self.change_settings)
        
        new = Menu(menu)
        new.add_command(label="Random", command = self.randomize)
        new.add_command(label="Seed", command = self.seed)
        new.add_command(label="Glider Gun", command = self.gun)
        
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
        
        # Key Bindings
        #   Manual mode controls:
        master.bind("<space>", self.manual_action)
        master.bind("<Up>", self.manual_action)
        master.bind("<Down>", self.manual_action)
        master.bind("<Left>", self.manual_action)
        master.bind("<Right>", self.manual_action)
        master.bind("<c>", self.manual_action)
        
        #   General controls:
        master.bind("<Escape>", self.quit_game)
        master.bind("<q>", self.quit_game)
        master.bind("<s>", self.change_settings)
        master.bind("<p>", self.speed_up)
        master.bind("<o>", self.slow_down)
        master.bind("<m>", self.toggle_manual_mode)
        master.bind("<e>", self.toggle_agent)
    
    def act(self, a):
        # Make demon take an action based on int parameter <a>:
        # 0 --> Wait until next step
        # 1 --> Move up
        # 2 --> Move right
        # 3 --> Move down
        # 4 --> Move left
        # 5 --> Flip cell
        loc = self.agent.vision.eye_location # Demon's (x, y) location
        if a == 0:
            self.wait = True                 #; print("Wait", end=" ")
        elif a == 1 and loc[0] < self.size[0]-1:
            self.agent.vision.move(1, 0)     #; print("Up", end=" ")
        elif a == 2 and loc[0] > 0:
            self.agent.vision.move(-1, 0)    #; print("Down", end=" ")
        elif a == 3 and loc[1] < self.size[1]-1:
            self.agent.vision.move(0, 1)     #; print("Left", end=" ")
        elif a == 4 and loc[1] > 0:
            self.agent.vision.move(0, -1)    #; print("Right", end=" ")
        elif a == 5:
            self.flip_cell()                 #; print("Flipped", end=" ")
            
    def clear(self):
        self.generation = 0
        blank_data = np.ones(self.size, dtype="uint8")
        self.data = blank_data
        self.agent.vision.update(blank_data)
        self.display_data()
        self.display_agent_view()
        
    def change_settings(self, *args):
        settings_popup = SettingsWindow(self.master, self)
        self.master.wait_window(settings_popup.top)
        
    def change_agent_speed(self, newspeed):
        self.agent_speed = newspeed
        
    def change_environment_grid(self, newwidth, newheight, newscale):
        # Change the number of cells wide, number of cells high, 
        # and pixel size of each cell
        oldsize = self.size
        old_data = self.data
        self.size = (newwidth, newheight)
        self.scale = newscale
        self.data = np.ones(self.size, dtype="uint8")
        # Preserve existing environment data after resizing
        for i in range(min(len(old_data), len(self.data))):
            for j in range(min(len(old_data[i]), len(self.data[i]))):
                self.data[i][j] = old_data[i][j]
        # Re-center the agent if making the grid smaller
        if newwidth < oldsize[0] or newheight < oldsize[1]:
            self.agent.vision.eye_location = np.array((self.size[0]//2, 
                                                       self.size[1]//2))
        self.agent.vision.update(self.data)
        self.display_agent_view()
        self.display_data()
        
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
        # Display a close-up of the demon's view in the display console
        if self.agent_enabled:
            view = self.agent.vision.get_view()
            img_data = np.ones(view.shape, dtype="float")
            for i in range(len(view)):
                for j in range(len(view[i])):
                    if view[i][j] is not None:
                        if view[i][j] == 1 and self.vis_field[i][j] != 2:
                            img_data[i][j] = 0.8
                        elif view[i][j] == 0:
                            img_data[i][j] = 0
            colorweighted_data = img_data * 255
            self.scale_render_place(colorweighted_data, 
                                    self.view_scale,
                                    self.agentview_display)
        else:
            view = self.agent.vision.get_view()
            img_data = np.ones(view.shape, dtype="float")
            colorweighted_data = img_data * 255
            self.scale_render_place(colorweighted_data, 
                                    self.view_scale,
                                    self.agentview_display)
        
    def display_data(self):
        # Display the environment in the main window
        colorweighted_data = self.data * 255
        eye_x, eye_y = self.agent.vision.eye_location
        vf = self.vis_field
        # Find local eye location in visual field template
        for i in range(len(vf)):
            for j in range(len(vf[i])):
                if vf[i][j] == 2:
                    loc_eye_x, loc_eye_y = i, j         
        # Make agent visual field appear gray so it can be seen in environment
        if self.agent_enabled:
            for i in range(len(vf)):
                for j in range(len(vf[i])):
                    x = i + eye_x - loc_eye_x
                    y = j + eye_y - loc_eye_y
                    # Keep the eye white
                    if x == eye_x and y == eye_y:
                        continue
                    # If within visual field and within bounds of the env. :
                    if 0 <= x < self.size[0] and 0 <= y < self.size[1]:
                        if colorweighted_data[x][y] == 255 and vf[i][j] > 0:
                            colorweighted_data[x][y] = 200
        self.scale_render_place(colorweighted_data, self.scale, self.env_img)

    def display_reward_scheme(self):
        rshape = self.reward_scheme.desired_shape
        colorweighted_data = np.ones(rshape.shape)
        for i in range(len(rshape)):
            for j in range(len(rshape[i])):
                if self.vis_field[i][j] == 0:
                    colorweighted_data[i][j] = 255
                elif self.vis_field[i][j] == 1:
                    if rshape[i][j] == 1:
                        colorweighted_data[i][j] = 200
                    else:
                        colorweighted_data[i][j] = 0
                elif self.vis_field[i][j] == 2:
                    if rshape[i][j] == 1:
                        colorweighted_data[i][j] = 255
                    else:
                        colorweighted_data[i][j] = 0
  
    def flip_cell(self, chance=1):
        # When called, agent has a 1 in <chance> chance of flipping the
        # cell at its current location from 1 to 0 or vice versa.
        loc = tuple(self.agent.vision.eye_location)
        chance = random.randint(1, chance)
        if chance == 1:
            self.data[loc] = abs(self.data[loc] - 1)    # flip cell
            
    def get_reward(self):
        """
        r = self.reward_scheme.calc_reward(self.agent.vision.get_view())
        if r == 10:
            self.match_count += 1 
            exact_match_msg = Label(self.message_box, 
                                    text="Exact Match {0:2d}\n"
                                    "Generation: {1:}".format( \
                                                 self.match_count, 
                                                 self.generation),
                                    fg="white", bg="gray")
            exact_match_msg.pack()
            print("EXACT MATCH {0:2d} - Generation {1:2d}".format( \
                                                 self.match_count,
                                                 self.generation))
        return r
        """
        # Calculate reward based on how many live cells are in the visual field
        r = 0
        for cell in self.agent.vision.viewdata:
            if cell == 0:
                r += 1
        reward = r   
        return reward
        
        
    def gun(self):
        # Initialize a "Gosper's glider gun"
        self.generation = 0
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
        self.agent.vision.update(gun_data)
        self.display_data()
        self.display_agent_view()
        
    def load(self):
        print("Loading last saved brain...")
        brain_filename = filedialog.askopenfilename(initialdir="/Users/JamesButcher/Documents/Programming/Game-of-Life-AI/A-demon-in-conways-game-of-life/Saved_brains", 
            title="Select file",
            filetypes=[("Path file", "*.pth")])
        self.agent.load(brain_filename)
        
    def manual_action(self, event):
        if self.agent_enabled and self.manual_mode:
            #print("Manual action taken:", repr(event.char))
            # Spacebar causes game to advance one step
            if str(event.char) == " ":
                self.step()
                action = 0
            # Move actions - up/down and left/right have been reversed
            elif str(event.char) == "\uf700":   # Up key pressed
                action = 2
            elif str(event.char) == "\uf701":   # Down key pressed
                action = 1
            elif str(event.char) == "\uf702":   # Left key pressed
                action = 4
            elif str(event.char) == "\uf703":   # Right key pressed
                action = 3
            # "c" key flips cell
            else:
                action = 5
            self.act(action)
            self.update_agent()
        elif not self.manual_mode:
            if str(event.char) == " ":
                if self.running:
                    self.stop_game()
                else:
                    self.start_game()
        
    def move_chance(self, chance=10):
        # When called, demon has a 1 in <chance> chance of moving in one
        # of four directions.
        # Increasing <chance> decreases the chance of moving.
        loc = self.agent.vision.eye_location
        chance = random.randint(0, chance)
        if chance == 1 and loc[0] < size[0]-1:
            self.agent.vision.move(1, 0)
        elif chance == 2 and loc[0] > 0:
            self.agent.vision.move(-1, 0)
        elif chance == 3 and loc[1] < size[1]-1:
            self.agent.vision.move(0, 1)
        elif chance == 4 and loc[1] > 0:
            self.agent.vision.move(0, -1)
            
    def quit_game(self, *args):
        paused = False
        if self.running:
            self.stop_game()
            paused = True
        quit_popup = QuitWindow(root)
        root.wait_window(quit_popup.top)
        if quit_popup.cancelled and paused:
            self.start_game()

        
    def randomize(self):
        # Initialize a random state on the whole grid
        self.generation = 0
        self.data = np.random.randint(0, 2, self.size, dtype="uint8")  
        self.agent.vision.update(self.data)
        self.display_data()
        self.display_agent_view()
        
    def save(self):
        print("Saving brain...")
        save_popup = SaveWindow(self.master, self.agent)
        self.master.wait_window(save_popup.top)
        
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
        self.generation = 0
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
        self.agent.vision.update(seed_data)
        self.display_data()
        self.display_agent_view()
        
    def slow_down(self, *args):
        self.interval += .1
        
    def speed_up(self, *args):
        if self.interval > 0.1:
            self.interval -= 0.1
        else:
            self.interval = 0
        
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
        self.agent.vision.update(newdata)
        self.display_agent_view()
        if self.agent_enabled and not self.manual_mode:
            self.agent.vision.update(newdata)
            self.display_agent_view()
            # Update the demon X number of times --- set by self.agent_speed
            for i in range(self.agent_speed):
                self.update_agent()
                time.sleep(self.interval / self.agent_speed)
                if self.wait == True:
                    break
            self.wait = False
        self.update()
        self.generation += 1
        self.generation_count.config(text=str(self.generation))
        # Force program to wait for an interval of time. This controls the
        # game speed after pressing Start
        if self.running:# and not self.manual_mode:
            t = time.time() + self.interval
            while time.time() < t:
                pass
    
    def stop_game(self):
        self.running = False
        
    def toggle_agent(self, *args):
        # Turn the agent on or off so you can see how the game evolves
        # with or without the agent
        self.agent_enabled = not self.agent_enabled
        self.display_data()
        self.display_agent_view()
        
    def toggle_manual_mode(self, *args):
        # Enable or disable manual control over the agent
        for event in args:
            if event.char == "m":
                # Toggle the check graphic if activated by pressing 
                #   the "m" key instead of clicking on the button
                self.manual_mode_button.toggle()
        if not self.manual_mode:
                # Pause the game when entering manual mode
            self.stop_game()
        self.manual_mode = not self.manual_mode

    def update_agent(self):
        self.agent.vision.update(self.data)
        viewdata = self.agent.vision.get_viewdata()
        # Treat cells off the edge of the environment grid as dead: 1, not None
        viewdata = [1 if x is None else x for x in viewdata]
        reward = self.get_reward()
        action = self.agent.update(reward, viewdata)
        if not self.manual_mode:
            self.act(action)        
            self.agent.vision.update(self.data)
        self.last_reward = reward
        self.update()
        self.display_data()
        self.display_agent_view()
        self.update_meters()
        
    def update_meters(self):        
        reward_meter_scale = 5
        x0, y0, x1, y1 = self.reward_meter.coords(self.reward_meter_level)
        x1 = reward_meter_scale * self.last_reward
        if x1 < 2:
            x1 = 2
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
        
        
class SaveWindow(Frame):
    def __init__(self, master, agent):
        self.agent = agent
        top = self.top = Toplevel(master)
        top.title("Save")
        self.top.geometry("350x150")
        save_message = Label(top, text="Enter name of saved brain:")
        save_message.pack()
        self.entry = Entry(top)
        self.entry.pack()
        ok_button = Button(top, text="OK", command=self.execute)
        ok_button.pack()
        
    def execute(self):
        value = self.entry.get()
        self.agent.save(value)
        self.top.destroy()
        
        
class SettingsWindow(Frame):
    def __init__(self, master, main_window):
        self.main_window = main_window
        top = self.top = Toplevel(master)
        self.top.title("Settings")
        
        s = Frame(top, borderwidth=10, bg="#CCCCCC")
        s.pack()
        speed_label = Label(s, text="Agent Speed (actions per game step):",
                            bg="#CCCCCC")
        speed_label.grid(row=0, column=0, sticky="e")
        self.speed_entry = Entry(s, width=3)
        self.speed_entry.bind("<Return>", self.execute)
        self.speed_entry.grid(row=0, column=1, sticky="w")
        self.speed_entry.insert(0, str(self.main_window.agent_speed))
        
        grid_width_label = Label(s, text="Environment width:", bg="#CCCCCC")
        grid_width_label.grid(row=1, column=0, sticky="e")
        self.grid_width_entry = Entry(s, width=3)
        self.grid_width_entry.bind("<Return>", self.execute)
        self.grid_width_entry.grid(row=1, column=1, sticky="w")
        self.grid_width_entry.insert(0, str(self.main_window.size[0]))
        
        grid_height_label = Label(s, text="Environment height:", bg="#CCCCCC")
        grid_height_label.grid(row=2, column=0, sticky="e")
        self.grid_height_entry = Entry(s, width=3)
        self.grid_height_entry.bind("<Return>", self.execute)
        self.grid_height_entry.grid(row=2, column=1, sticky="w")
        self.grid_height_entry.insert(0, str(self.main_window.size[1]))
        
        scale_label = Label(s, text="Scale:", bg="#CCCCCC")
        scale_label.grid(row=3, column=0, sticky="e")
        self.scale_entry = Entry(s, width=3)
        self.scale_entry.bind("<Return>", self.execute)
        self.scale_entry.grid(row=3, column=1, sticky="w")
        self.scale_entry.insert(0, str(self.main_window.scale))
        
        ok_button = Button(s, text="OK", command=self.execute)
        ok_button.bind("<Return>", self.execute)
        ok_button.grid(row=4, column=0)
        
    def execute(self, *args):
        speed = int(self.speed_entry.get())
        grid_width = int(self.grid_width_entry.get())
        grid_height = int(self.grid_height_entry.get())
        scale = int(self.scale_entry.get())
        self.main_window.change_agent_speed(speed)
        self.main_window.change_environment_grid(grid_width, grid_height, scale)
        self.top.destroy()
        
        
class QuitWindow(Frame):
    def __init__(self, master):
        self.master = master
        self.cancelled = False
        top = self.top = Toplevel(master)
        top.geometry("230x100")
        
        msg = Label(top, text="Are you sure you want to quit?", 
                    padx=20, pady=20)
        msg.grid(row=0, columnspan=2)
        
        yes_button = Button(top, text="Yes", command=self.quit_app)
        yes_button.bind("<Return>", self.quit_app)
        yes_button.grid(row=1, column=0, sticky="e")
        
        cancel_button = Button(top, text="Cancel", command=self.cancel)
        cancel_button.bind("<Return>", self.cancel)
        cancel_button.grid(row=1, column=1, sticky="w")
        
    def quit_app(self, *args):
        self.top.destroy()
        quit()
        
    def cancel(self, *args):
        self.cancelled = True
        self.top.destroy()
        
    
root = Tk()
root.title("A Demon In Conway's Game Of Life")
        
size = (50, 50)     # Default: (50, 50)
scale = 10          # Default: 10 

main_window = MainWindow(root, size, scale)

root.mainloop()