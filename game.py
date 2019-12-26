"""
"A Demon in Conway's Game of Life"
 by James Butcher
 Github: https://github.com/jmsbutcher/A-demon-in-conways-game-of-life
 First created: 8-31-19

 Current version: 0.7
 Date: 12-26-19

 Python version 3.7.3
 Pytorch version 1.2.0

"""

# Import the "demon" (the intelligent agent) from demon.py
from demon import Agent, RewardScheme

import matplotlib.pyplot as plt
import numpy as np
import random
import time
from pathlib import Path
from PIL import Image, ImageTk
from random import randint
from tkinter import BOTH, BOTTOM, END, LEFT, RIGHT, Y, \
                    Button, Canvas, Checkbutton, Entry, \
                    filedialog, Frame, Label, Listbox, Menu, \
                    OptionMenu, Scrollbar, StringVar, Tk, Toplevel

# Default visual field shapes --- used in GridEditorWindow
#   0 - Not part of the visual field
#   1 - Part of the visual field
#   2 - The agent's eye location
extralarge_vf = np.array(
    [[0, 0, 1, 1, 1, 1, 1, 0, 0],
     [0, 1, 1, 1, 1, 1, 1, 1, 0],
     [1, 1, 1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 2, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 1],
     [0, 1, 1, 1, 1, 1, 1, 1, 0],
     [0, 0, 1, 1, 1, 1, 1, 0, 0]])
large_vf = np.array(
    [[0, 0, 1, 1, 1, 0, 0],
     [0, 1, 1, 1, 1, 1, 0],
     [1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 2, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 1],
     [0, 1, 1, 1, 1, 1, 0],
     [0, 0, 1, 1, 1, 0, 0]])
medium_vf = np.array(
    [[0, 0, 1, 1, 0, 0],
     [0, 1, 1, 1, 1, 0],
     [1, 1, 1, 1, 1, 1],
     [1, 1, 2, 1, 1, 1],
     [0, 1, 1, 1, 1, 0],
     [0, 0, 1, 1, 0, 0]])
small_vf = np.array(
    [[0, 0, 1, 0, 0],
     [0, 1, 1, 1, 0],
     [1, 1, 2, 1, 1],
     [0, 1, 1, 1, 0],
     [0, 0, 1, 0, 0]])


class MainWindow(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        # General game parameters:
        self.master = master
        self.size = (50, 50)    # (x,y) dimensions of rectangular cell grid
        self.scale = 10         # pixel size of each cell
        self.view_scale = 15    # pixel size of agent view in display console
        self.data = np.ones(self.size, dtype="uint8")  # initialize cell states
        self.running = False    # game is paused until start_game() is called
        self.interval = 0.5     # time between steps; how fast the game runs
        self.generation = 0     # how many steps (generations) has passed
        self.vis_field = large_vf  # Default shape of the agent's visual field
        # Initialize agent eye location coordinates to the center of the grid
        self.eye_location = np.array((self.size[0]//2, self.size[1]//2))

        # Create and initialize the agent object (the demon)
        #   The method below initializes the following:
        #     - self.agent = Agent(...)
        #     - self.agent.vision.update(data)
        #     - gamma = 0.9 by default
        self.initialize_agent()
        self.agent_enabled = False  # Start game with agent disabled
        self.manual_mode = False    # Start game with manual mode off
        self.agent_speed = 10       # Maximum agent actions per game step
        self.wait = False           # For handling wait actions during game

        #---------------------------------------------------------------------
        # Frames and Widgets
        #---------------------------------------------------------------------
        # Right Frame - Display Console: for displaying demon's readouts
        self.display_console = Frame(self.master, borderwidth=5,
                                     relief="ridge", width=400, bg="gray")
        self.display_console.pack(fill=BOTH, side=RIGHT, expand=1)

        # Initialize and display generation count
        generation_label = Label(self.display_console, text="Generation #:",
                                 fg="white", bg="gray")
        generation_label.grid(row=0, columnspan=2)
        self.generation_count = Label(self.display_console,
                                      text=str(self.generation),
                                      font=("Arial", 20), width=3,
                                      fg="white", bg="gray")
        self.generation_count.grid(row=0, column=2, sticky="w")

        # Initialize and display a close-up of what the demon currently sees
        self.agentview_display = Label(self.display_console, image=None)
        self.agentview_display.grid(row=1, columnspan=3)
        self.display_agent_view()

        # Initialize and display meter readings in display console
        # Reward meter
        reward_label = Label(self.display_console, text="Reward:",
                             fg="white", bg="gray")
        reward_label.grid(row=2, column=0, padx=4, pady=7, sticky="w")
        self.reward_meter = Canvas(self.display_console, width=100, height=15)
        self.reward_meter.grid(row=2, column=1, columnspan=2)
        self.reward_meter_level = self.reward_meter.create_rectangle(
                                  5, 5, 6, 15, width=1, fill="black")

        # Initialize and display reward scheme fields
        #   Format:
        #           Current reward scheme:
        #
        #           [ reward scheme type ]
        #
        #         [ New Reward Scheme Button ] - if reward scheme is None
        #                   or
        #            [ reward shape view ]     - if reward scheme is "shape"
        #                   or
        #               [ Empty ]              - if "maximize" or "minimize"
        reward_scheme_label = Label(self.display_console,
                                    text="\nCurrent reward scheme:",
                                    fg="white", bg="gray")
        reward_scheme_label.grid(row=4, columnspan=2, pady=5, sticky="s")

        self.reward_scheme_name = Label(self.display_console,
                                        text="No reward scheme",
                                        fg="white", bg="gray")
        self.reward_scheme_name.grid(row=5, columnspan=2, pady=5, sticky="s")

        self.reward_shape_view = Label(self.display_console, image=None)

        self.new_reward_scheme_button = Button(
                                        self.display_console,
                                        text="New Reward Scheme",
                                        command=self.change_reward_scheme)
        self.new_reward_scheme_button.grid(row=6, columnspan=2,
                                           pady=30, sticky="N")

        # Initialize and display reward scheme in display console
        #   The method below initializes the following parameters:
        #     - self.reward_scheme = RewardScheme(...)
        #     - self.match_count = 0
        #     - self.last_reward = 0
        #     - self.reward_window = []
        #     - self.avg_running_reward = []
        self.initialize_reward_scheme()

        # Initialize and display message list in display console
        self.message_frame = Frame(self.display_console)
        self.message_frame.grid(row=7, columnspan=3, pady=10)
        self.message_list = Listbox(self.message_frame,
                                    width=28, height=16,
                                    font=("Courier", 11))
        self.message_list.pack(side=LEFT, fill=BOTH, expand=1)
        self.message_scrollbar = Scrollbar(self.message_frame,
                                           command=self.message_list.yview)
        self.message_scrollbar.pack(side=RIGHT, fill=Y)
        self.message_list.configure(yscrollcommand=self.message_scrollbar.set)

        # Bottom Frame - Button toolbar: for game control buttons
        button_menu = Frame(self.master, bg="#DDDDDD",
                            relief="ridge", borderwidth=5)
        button_menu.pack(fill=BOTH, side=BOTTOM, expand=0)

        # Main Frame - The cell grid environment
        self.win_X = self.size[0] * self.scale
        self.win_Y = self.size[1] * self.scale
        self.environment = Frame(self.master,
                                 width=self.win_Y,
                                 height=self.win_X,
                                 bg="black")
        self.environment.pack(fill=BOTH, side=LEFT, expand=1, ipadx=6, ipady=6)

        # Initialize and display the environment cells
        self.env_img = Label(self.environment, image=None)
        self.env_img.place(x=3, y=3)
        self.display_data()

        #---------------------------------------------------------------------
        # Buttons
        #---------------------------------------------------------------------
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
        self.toggle_agent_button = Checkbutton(button_menu,
                                               text="Enable Demon",
                                               bg="#DDDDDD",
                                               command=self.toggle_agent)
        self.manual_mode_button = Checkbutton(button_menu, text="Manual Mode",
                                              bg="#DDDDDD",
                                              bd=3,
                                              variable=1,
                                              command=self.toggle_manual_mode)
        self.paused_message = Label(button_menu, text="Paused",
                                    bg="#DDDDDD", fg="red",
                                    font=("Arial Black", 25))

        quit_button.grid(row=0, column=1, padx=10, pady=2)
        seed_button.grid(row=0, column=2, padx=1)
        random_button.grid(row=0, column=3, padx=1)
        clear_button.grid(row=0, column=4, padx=1)
        start_button.grid(row=0, column=5, padx=1)
        step_button.grid(row=0, column=6, padx=1)
        stop_button.grid(row=0, column=7, padx=1)
        speed_up_button.grid(row=0, column=8, padx=1)
        slow_down_button.grid(row=0, column=9, padx=1)
        self.toggle_agent_button.grid(row=1, column=0, columnspan=4,
                                      sticky="w", padx=3, pady=3)
        self.manual_mode_button.grid(row=2, column=0, columnspan=4,
                                     sticky="w", padx=3, pady=3)
        self.paused_message.grid(row=1, column=1, rowspan=2, columnspan=8)

        #---------------------------------------------------------------------
        # Menus
        #---------------------------------------------------------------------
        menu = Menu(self.master)
        self.master.config(menu=menu)

        # "File" menu
        file = Menu(menu)
        file.add_command(label="Exit\t\t\t\t[Esc]",
                         command=self.quit_game)
        file.add_separator()

        file.add_command(label="Save Brain",
                         command=self.save)
        file.add_command(label="Save Visual Field",
                         command=self.save_visual_field)
        file.add_command(label="Save Reward Shape",
                         command=self.save_reward_scheme_shape)
        file.add_separator()

        file.add_command(label="Load Brain",
                         command=self.load)
        file.add_command(label="Load Visual Field",
                         command=self.load_visual_field)
        file.add_command(label="Load Reward Shape",
                         command=self.load_reward_scheme_shape)
        file.add_separator()

        file.add_command(label="Settings\t\t\t[s]",
                         command=self.change_settings)
        menu.add_cascade(label="File", menu=file)

        # "New" menu
        new = Menu(menu)
        new.add_command(label="Visual Field\t\t[v]",
                        command=self.change_visual_field)
        new.add_separator()

        new.add_command(label="Reward Scheme\t[r]",
                        command=self.change_reward_scheme)
        new.add_separator()

        env = Menu(new)
        env.add_command(label="Empty",
                        command=self.clear)
        env.add_command(label="Random",
                        command=self.randomize)
        env.add_command(label="Seed",
                        command=self.seed)
        env.add_command(label="Glider Gun",
                        command=self.gun)

        new.add_cascade(label="Environment", menu=env)
        menu.add_cascade(label="New", menu=new)

        # "Run" menu
        run = Menu(menu)
        run.add_command(label="Start\t\t\t[Space]",
                        command=self.start_game)
        run.add_command(label="Step\t\t\t[Space] (in manual mode)",
                        command=self.step)
        run.add_command(label="Stop\t\t\t[Space]",
                        command=self.stop_game)
        run.add_separator()

        run.add_command(label="Speed Up\t\t[p]",
                        command=self.speed_up)
        run.add_command(label="Slow Down\t\t[o]",
                        command=self.slow_down)

        menu.add_cascade(label="Run", menu=run)

        # "Window" menu
        win = Menu(menu)
        win.add_command(label="Reward Plot", command=self.display_reward_plot)
        menu.add_cascade(label="Window", menu=win)

        #---------------------------------------------------------------------
        # Key Bindings
        #---------------------------------------------------------------------
        # Manual mode controls:
        master.bind("<space>", self.manual_action)
        master.bind("<Up>", self.manual_action)
        master.bind("<Down>", self.manual_action)
        master.bind("<Left>", self.manual_action)
        master.bind("<Right>", self.manual_action)
        master.bind("<c>", self.manual_action)

        # General controls:
        master.bind("<Escape>", self.quit_game)
        master.bind("<e>", self.toggle_agent)
        master.bind("<m>", self.toggle_manual_mode)
        master.bind("<o>", self.slow_down)
        master.bind("<p>", self.speed_up)
        master.bind("<q>", self.quit_game)
        master.bind("<s>", self.change_settings)
        master.bind("<v>", self.change_visual_field)
        master.bind("<r>", self.change_reward_scheme)


    def act(self, a):
        # Make agent take an action based on int argument <a>:
        # 0 --> Wait until next step
        # 1 --> Move up
        # 2 --> Move right
        # 3 --> Move down
        # 4 --> Move left
        # 5 --> Flip cell
        loc = self.agent.vision.eye_location  # Agent's (x, y) location
        if a == 0:
            self.wait = True                  # ; print("Wait")
        elif a == 1 and loc[0] < self.size[0]-1:
            self.agent.vision.move(1, 0)      # ; print("Down")
        elif a == 2 and loc[0] > 0:
            self.agent.vision.move(-1, 0)     # ; print("Up")
        elif a == 3 and loc[1] < self.size[1]-1:
            self.agent.vision.move(0, 1)      # ; print("Right")
        elif a == 4 and loc[1] > 0:
            self.agent.vision.move(0, -1)     # ; print("Left")
        elif a == 5:
            self.flip_cell()                  # ; print("Flipped")

    def clear(self):
        # Clear the environment grid of all live cells
        self.data = np.ones(self.size, dtype="uint8")
        self.generation = 0
        self.refresh_data_view()

    def change_settings(self, *args):
        # Open the settings window
        settings_popup = SettingsWindow(self.master, self)
        self.master.wait_window(settings_popup.top)

    def change_agent_speed(self, newspeed):
        # Change the number of agent actions per game step
        self.agent_speed = newspeed

    def change_environment_grid(self, newwidth, newheight, newscale):
        # Change the number of cells wide, number of cells high,
        #   and/or pixel size of each cell
        old_size = self.size
        old_data = self.data
        self.size = (newheight, newwidth)
        self.scale = newscale
        self.win_X = self.size[0] * self.scale
        self.win_Y = self.size[1] * self.scale
        self.data = np.ones(self.size, dtype="uint8")

        # Preserve existing environment data after resizing
        for i in range(min(len(old_data), len(self.data))):
            for j in range(min(len(old_data[i]), len(self.data[i]))):
                self.data[i][j] = old_data[i][j]

        # Re-center the agent if making the grid smaller
        if newwidth < old_size[0] or newheight < old_size[1]:
            self.agent.vision.eye_location = np.array((self.size[0]//2,
                                                       self.size[1]//2))

        self.environment.configure(width=self.win_Y, height=self.win_X)
        self.generation = 0
        self.refresh_data_view()

    def change_reward_scheme(self, *args):
        # Open the reward scheme editor
        popup = RewardSchemeWindow(self.master, self)
        self.master.wait_window(popup.top)

    def change_visual_field(self, *args):
        # Open the visual field editor
        popup = GridEditorWindow(self.master, self, None,
                                 self.vis_field, "visual field")
        self.master.wait_window(popup.top)

    def display_exact_match(self):
        # Add a message to the message list whenever the agent sees an exact
        #   match when using a "Produce shape" reward scheme
        if self.reward_scheme.check_for_exact_match():
            self.match_count += 1
            self.message_list.insert(END,
                                     "Exact Match #{0:2d} - \n"
                                     "Generation: {1:}".format(
                                         self.match_count, self.generation))
            print("EXACT MATCH {0:2d} - Generation {1:2d}".format(
                                                 self.match_count,
                                                 self.generation))
            self.message_list.yview(END)

    def conway_rule(self, x, y):
        # Get state of cell at coordinates (x, y) and those of its neighbors
        state = self.data[x][y]
        nghbrs = [self.data[x-1][y-1], self.data[x][y-1], self.data[x+1][y-1],
                  self.data[x-1][y],          0,          self.data[x+1][y],
                  self.data[x-1][y+1], self.data[x][y+1], self.data[x+1][y+1]]

        # Calculate the resulting state of the cell according to the rules
        neighbor_total = sum(nghbrs)
        # If cell is alive:
        if state == 0:
            if neighbor_total > 6:
                state = 1  # dies from underpopulation
            elif neighbor_total > 4:
                state = 0  # remains alive
            else:
                state = 1  # dies from overpopulation

        # If cell is dead, but has exactly 3 live neighbors:
        elif state == 1 and neighbor_total == 5:
            state = 0   # comes to life

        return state

    def display_agent_view(self):
        # Display a close-up of the agent's view in the display console
        #  - Initialize to an all white grid the same size of the view, so the
        #    display stays the same size whether the agent is enabled or not
        view = self.agent.vision.get_view()
        colorweighted_data = np.ones(view.shape) * 255      # White

        if self.agent_enabled:
            for i in range(len(view)):
                for j in range(len(view[i])):
                    # If part of the visual field:
                    if view[i][j] is not None:
                        # If cell is dead and not the eye location:
                        if view[i][j] == 1 and self.vis_field[i][j] != 2:
                            colorweighted_data[i][j] = 200  # Gray
                        # If cell is alive (regardless of eye location):
                        elif view[i][j] == 0:
                            colorweighted_data[i][j] = 0    # Black

        # Place the image
        self.scale_render_place(colorweighted_data,
                                self.view_scale,
                                self.agentview_display)

    def display_data(self):
        # Display the state of the environment grid in the main window
        #  - Dead cells appear white: Data value 1 --> Colorweighted value 255
        #  - Live cells appear black: Data value 0 --> Colorweighted value 0
        colorweighted_data = self.data * 255

        # Make agent visual field appear gray so it can be seen in environment
        #  - Dead cells appear gray: change colorweighted value from 255 to 200
        #  - Live cells still appear black: 0
        if self.agent_enabled:
            # Find local eye location in visual field template
            vf = self.vis_field
            for i in range(len(vf)):
                for j in range(len(vf[i])):
                    if vf[i][j] == 2:
                        loc_eye_x, loc_eye_y = i, j
            # Find absolute cell coordinates that fall within the visual field
            eye_x, eye_y = self.agent.vision.eye_location
            for i in range(len(vf)):
                for j in range(len(vf[i])):
                    x = i + eye_x - loc_eye_x
                    y = j + eye_y - loc_eye_y
                    # Keep the eye either black or white
                    if x == eye_x and y == eye_y:
                        continue
                    # If within visual field and within bounds of the env. :
                    if 0 <= x < self.size[0] and 0 <= y < self.size[1]:
                        if colorweighted_data[x][y] == 255 and vf[i][j] > 0:
                            colorweighted_data[x][y] = 200

        # Place the image
        self.scale_render_place(colorweighted_data, self.scale, self.env_img)

    def display_reward_scheme(self):
        # Change the reward scheme title text
        self.reward_scheme_name.configure(
            text=self.reward_scheme.get_reward_text())

        # Add or remove the "New Reward Scheme" button
        if self.reward_scheme.schemetype is None:
            self.new_reward_scheme_button.grid(row=6, columnspan=2,
                                               pady=30, sticky="N")
        else:
            self.new_reward_scheme_button.grid_remove()

        # Add or remove reward scheme shape image
        if self.reward_scheme.schemetype == "shape":
            rshape = self.reward_scheme.get_shape()
            colorweighted_data = np.ones(rshape.shape)
            for i in range(len(rshape)):
                for j in range(len(rshape[i])):
                    # If not part of the visual field:
                    if self.vis_field[i][j] == 0:
                        colorweighted_data[i][j] = 255      # White
                    # If part of the visual field (but not the eye location):
                    elif self.vis_field[i][j] == 1:
                        # If not part of reward shape:
                        if rshape[i][j] == 1:
                            colorweighted_data[i][j] = 200  # Gray
                        # If part of reward shape:
                        else:
                            colorweighted_data[i][j] = 0    # Black
                    # If at eye location:
                    elif self.vis_field[i][j] == 2:
                        # If not part of reward shape:
                        if rshape[i][j] == 1:
                            colorweighted_data[i][j] = 255  # Keep eye white
                        # If part of reward shape:
                        else:
                            colorweighted_data[i][j] = 0    # Black
            # Place the image
            self.reward_shape_view.grid(row=6, columnspan=3)
            self.scale_render_place(colorweighted_data, self.view_scale,
                                    self.reward_shape_view)
        else:
            self.reward_shape_view.grid_remove()

    def display_reward_plot(self):
        # Show a matplotlib plot of the average running reward over time
        plt.plot(self.avg_running_reward)
        plt.show()

    def flip_cell(self, chance=1):
        # When called, agent has a 1 in <chance> chance of flipping the
        #   cell at its current location from 1 to 0 or vice versa.
        eye_loc = tuple(self.agent.vision.eye_location)
        chance = random.randint(1, chance)
        if chance == 1:
            self.data[eye_loc] = abs(self.data[eye_loc] - 1)    # flip cell

    def gun(self):
        # Initialize a "Gosper's glider gun" on the environment grid
        #
        # Check to make sure environment is big enough to fit
        if self.size[0] <= 43 or self.size[1] <= 20:
            print("Cannot initialize glider gun -- environment grid is"
                  " too small.\nWidth must be at least 43 and height must"
                  " be at least 20.")
            return
        self.generation = 0
        gun_data = np.ones(self.size, dtype="uint8")
        x = 5
        y = 5
        gun_data[x][y+25] = 0
        gun_data[x+1][y+23:y+26] = [0, 1, 0]
        gun_data[x+2][y+13:y+37] = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
        gun_data[x+3][y+12:y+37] = [0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
        gun_data[x+4][y+1:y+23]  = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
                                    1, 1, 0, 1, 1, 1, 0, 0]
        gun_data[x+5][y+1:y+26]  = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
                                    0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0]
        gun_data[x+6][y+11:y+26] = [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
                                    0]
        gun_data[x+7][y+12:y+17] = [0, 1, 1, 1, 0]
        gun_data[x+8][y+13:y+15] = [0, 0]
        self.data = gun_data
        self.refresh_data_view()

    def initialize_agent(self):
        # Create new agent object
        data = self.data
        vf = self.vis_field
        eye_loc = self.eye_location
        gamma = 0.9  # Future discount factor for reinforcement learning
        self.agent = Agent(data, vf, eye_loc, gamma)
        self.agent.vision.update(data)

    def initialize_reward_scheme(self, schemetype=None, name=None, shape=None):
        # Create new reward scheme object
        vf = self.vis_field
        vision = self.agent.vision
        self.reward_scheme = RewardScheme(vf, vision, schemetype, name, shape)

        self.display_reward_scheme()
        self.match_count = 0
        self.last_reward = 0
        self.reset_reward_window()

    def load(self):
        # Load a brain model previously saved as a Pytorch .pth file
        print("Loading brain...")

        folder = Path.cwd() / "Saved_brains"

        filename = filedialog.askopenfilename(
            title="Load brain file",
            filetypes=[("Path file", "*.pth")],
            initialdir=folder)

        # Cancel load if the user cancels out of the file dialog
        if filename == "":
            return

        file = folder / filename

        self.agent.load(file)

        print("Brain loaded: ", file)
        self.message_list.insert(END, "Brain loaded at gen {}".format(
                                 self.generation))
        self.message_list.insert(END, " \"{}\"".format(file.stem))
        self.message_list.yview(END)

    def load_reward_scheme_shape(self):
        # Load a reward scheme shape previously saved or
        #   written manually as a text file
        print("Loading reward scheme shape...")

        folder = Path.cwd() / "Reward_scheme_shapes"

        filename = filedialog.askopenfilename(
            title="Load reward scheme shape file",
            filetypes=[("Text file", "*.txt")],
            initialdir=folder)

        # Cancel load if the user cancels out of the file dialog
        if filename == "":
            return

        file = folder / filename

        new_shape = [[]]
        new_shapename = None

        # Read in the new shape and the corresponding shape name, if included
        f = open(file, "r")
        row = 0
        while True:
            cell = f.read(1)
            # Check for shape name - should be at top of file within brackets
            if cell == "<":
                new_shapename = ""
                char = f.read(1)
                while char != ">" and f.tell() < 50:
                    new_shapename += char
                    char = f.read(1)
                f.read(1)
            elif cell == "0":
                new_shape[row].append(int(0))
            elif cell == "1":
                new_shape[row].append(int(1))
            elif cell == "\n":
                new_shape.append([])
                row += 1
            elif cell == "":
                break
            else:
                print("ERROR: Bad reward scheme shape file.\n"
                      "File must contain only '0's and '1's without spaces.\n"
                      "Detected '{}'.".format(cell))
                f.close()
                return
        f.close()
        new_shape = np.array(new_shape)

        # Check to make sure the shape dimensions match the visual field's
        if new_shape.shape != self.vis_field.shape:
            print("ERROR: Reward scheme shape file must have the same\n"
                  "number of rows and columns as the current visual field.\n"
                  "  Visual field dimensions:         {0}\n"
                  "  Reward scheme shape dimensions:  {1}".format(
                                                     self.vis_field.shape,
                                                     new_shape.shape))
            return

        self.initialize_reward_scheme("shape", new_shapename, new_shape)
        self.display_reward_scheme()
        print("Reward scheme shape loaded:", file)
        if new_shapename is not None:
            print("New shape name:", new_shapename)

    def load_visual_field(self):
        # Load a visual field file previously saved or
        #   written manually as a text file
        print("Loading visual field...")

        folder = Path.cwd() / "Visual_field_files"

        filename = filedialog.askopenfilename(
            title="Load visual field file",
            filetypes=[("Text file", "*.txt")],
            initialdir=folder)

        # Cancel load if the user cancels out of the file dialog
        if filename == "":
            return

        file = folder / filename

        new_vf = [[]]

        # Read in the new visual field file
        f = open(file, "r")
        # Look for one and only one "2" in the file - this is the eye location
        eye_location_given = False
        row = 0
        while True:
            cell = f.read(1)
            if cell == "0":
                new_vf[row].append(0)
            elif cell == "1":
                new_vf[row].append(1)
            elif cell == "2":
                if not eye_location_given:
                    new_vf[row].append(2)
                    eye_location_given = True
                else:
                    print("ERROR: Bad visual field file.\n"
                          "File contains more than one '2'.\n"
                          "Must provide only a single '2' to indicate\n"
                          "eye location.")
                    f.close()
                    return
            elif cell == "\n":
                new_vf.append([])
                row += 1
            elif cell == "":
                break
            else:
                # If anything other than "0," "1," "2," or "\n" detected:
                print("ERROR: Bad visual field file.\n"
                      "File must only contain:\n"
                      " - A single '2'\n"
                      " - Any number of '0's and '1's\n"
                      " Detected '{}'.".format(cell))
                f.close()
                return
        f.close()
        new_vf = np.array(new_vf)

        if not eye_location_given:
            print("ERROR: Bad visual field file.\n"
                  "Must provide a '2' to indicate eye location.")
            return

        if len(new_vf.shape) != 2:
            print("ERROR: Bad visual field file.\n"
                  "File must provide a RECTANGULAR grid of:\n"
                  " - A single '2'\n"
                  " - Any number of '0's and '1's")
            return

        self.vis_field = new_vf
        self.initialize_agent()
        self.initialize_reward_scheme()
        self.refresh_data_view()
        print("Visual field file loaded: ", file)

    def manual_action(self, event):
        # Receive keyboard event and cause corresponding agent action
        if self.agent_enabled and self.manual_mode:

            # Spacebar causes game to advance one step
            if str(event.char) == " ":
                self.step()
                action = 0

            # Move actions - up/down and left/right have been reversed
            elif str(event.char) == "\uf700":   # Up key pressed
                action = 2  # Move up
            elif str(event.char) == "\uf701":   # Down key pressed
                action = 1  # Move down
            elif str(event.char) == "\uf702":   # Left key pressed
                action = 4  # Move left
            elif str(event.char) == "\uf703":   # Right key pressed
                action = 3  # Move right
            else:                               # "c" key pressed
                action = 5  # Flip cell

            self.act(action)
            self.update_agent()

        elif not self.manual_mode:
            # Spacebar pauses or unpauses continuous running of the game
            if str(event.char) == " ":
                if self.running:
                    self.stop_game()
                else:
                    self.start_game()

    def move_chance(self, chance=10):
        # For debugging - allows the agent to move randomly without a brain
        #   - When called, demon has a 1 in <chance> chance of moving in one
        #       of four directions
        #   - Increasing <chance> decreases the chance of moving at
        #       each game step
        loc = self.agent.vision.eye_location
        chance = random.randint(0, chance)
        if chance == 1 and loc[0] < self.size[0]-1:
            self.agent.vision.move(1, 0)
        elif chance == 2 and loc[0] > 0:
            self.agent.vision.move(-1, 0)
        elif chance == 3 and loc[1] < self.size[1]-1:
            self.agent.vision.move(0, 1)
        elif chance == 4 and loc[1] > 0:
            self.agent.vision.move(0, -1)

    def quit_game(self, *args):
        # Pause the game and open a popup window asking for confirmation
        #   before closing the program
        paused = False
        if self.running:
            self.stop_game()
            paused = True
        quit_popup = QuitWindow(root, self.agent)
        root.wait_window(quit_popup.top)
        if quit_popup.cancelled and paused:
            self.start_game()

    def randomize(self):
        # Initialize a random state of cells on the whole grid
        self.generation = 0
        self.data = np.random.randint(0, 2, self.size, dtype="uint8")
        self.refresh_data_view()

    def refresh_data_view(self):
        self.agent.vision.update(self.data)
        self.display_data()
        self.display_agent_view()

    def reset_reward_window(self):
        self.reward_window = []
        self.avg_running_reward = []

    def save(self):
        # Save the brain model (state and optimizer) as a Pytorch .pth file
        print("Saving brain...")

        folder = Path.cwd() / "Saved_brains"

        filename = filedialog.asksaveasfilename(
            title="Save brain file",
            filetypes=[("Path file", "*.pth")],
            initialdir=folder)

        # Cancel save if the user cancels out of the file dialog
        if filename == "":
            return

        file = folder / filename

        self.agent.save(file)

        print("Brain saved as {}".format(file))
        self.message_list.insert(END, "Brain saved at gen {}".format(
                                 self.generation))
        self.message_list.insert(END, " as \"{}\"".format(file.stem))
        self.message_list.yview(END)

    def save_visual_field(self):
        # Save the visual field as a text file
        print("Saving visual field...")

        vf = self.vis_field

        folder = Path.cwd() / "Visual_field_files"

        filename = filedialog.asksaveasfilename(
            title="Save visual field file",
            filetypes=[("Text file", "*.txt")],
            initialdir=folder)

        # Cancel load if the user cancels out of the file dialog
        if filename == "":
            return

        file = folder / filename

        f = open(file, "w+")
        for i in range(len(vf)):
            for j in range(len(vf[i])):
                f.write(str(int(vf[i][j])))
            if i < len(vf) - 1:
                f.write("\n")
        f.close()

        print("Visual field saved as {}".format(filename))
        self.message_list.insert(END, "Saved visual field as")
        self.message_list.insert(END, " \"{}\"".format(file.stem))
        self.message_list.yview(END)

    def save_reward_scheme_shape(self):
        # Save the reward scheme as a text file
        print("Saving reward scheme shape...")

        shape = self.reward_scheme.get_shape()
        shapename = self.reward_scheme.get_shapename()

        if shape is None:
            print("ERROR: No reward scheme shape detected.")
            self.message_list.insert(END, "Cannot save file -\nNo shape found")
            self.message_list.yview(END)
            return

        folder = Path.cwd() / "Reward_scheme_shapes"

        filename = filedialog.asksaveasfilename(
            title="Save reward scheme shape file",
            filetypes=[("Text file", "*.txt")],
            initialdir=folder)

        # Cancel save if the user cancels out of the file dialog
        if filename == "":
            return

        file = folder / filename

        f = open(file, "w+")
        if shapename:
            f.write("<{}>".format(shapename))
            f.write("\n")
        for i in range(len(shape)):
            for j in range(len(shape[i])):
                f.write(str(int(shape[i][j])))
            if i < len(shape) - 1:
                f.write("\n")
        f.close()

        print("Reward scheme shape saved as {}".format(filename))
        self.message_list.insert(END, "Saved reward shape \"{}\"".format(
                                 self.reward_scheme.get_shapename()))
        self.message_list.insert(END, " as \"{}\"".format(file.stem))
        self.message_list.yview(END)

    def scale_render_place(self, colorweighted_data, scale, placement_label):
        # Take a rank 2 numpy array of gray-colorweighted data (each
        #   value from 0 - 255), scale it according to the integer provided,
        #   render it, and place it into the Label widget provided.
        #
        # Example (all arrays are numpy arrays):
        # colorweighted_data:                scaled_data:
        #   [[255, 200],   scale 2      [[255, 255, 200, 200],
        #    [0  , 255]]    --->         [255, 255, 200, 200],
        #                                [0,   0,   255, 255],
        #                                [0,   0,   255, 255]]
        scaled_data = np.kron(colorweighted_data, np.ones((scale, scale)))
        raw_img = Image.fromarray(scaled_data)
        render = ImageTk.PhotoImage(raw_img)
        placement_label.configure(image=render)
        placement_label.image = render

    def seed(self):
        # Initialize a random state of cells in the center 8th of the grid
        self.generation = 0
        self.data = np.ones(self.size, dtype="uint8")
        dx, dy = self.size[0]//8, self.size[1]//8
        midx, midy = self.size[0]//2, self.size[1]//2
        for i in range(midx - dx, midx + dx):
            for j in range(midy - dy, midy + dy):
                r = randint(0, 1)
                self.data[i][j] = r
        self.refresh_data_view()

    def slow_down(self, *args):
        self.interval += .1

    def speed_up(self, *args):
        if self.interval > 0.1:
            self.interval -= 0.1
        else:
            self.interval = 0

    def start_game(self):
        self.running = True
        self.paused_message.grid_remove()
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

        # Update agent visual data and displays after applying Conway rule
        self.display_data()
        self.agent.vision.update(newdata)
        self.display_agent_view()

        # Update the agent <self.agent_speed> number of times
        if self.agent_enabled and not self.manual_mode:
            for i in range(self.agent_speed):
                self.update_agent()
                self.display_exact_match()
                time.sleep(self.interval / self.agent_speed)
                # If the agent selects the "wait" action, stop early
                if self.wait:
                    break
            self.wait = False

        self.update()
        self.generation += 1
        self.generation_count.config(text=str(self.generation))

        # Wait for an interval of time - This controls the game speed
        #   This implementation seems smoother than time.sleep(self.interval):
        if self.running:
            t = time.time() + self.interval
            while time.time() < t:
                pass

    def stop_game(self):
        self.running = False
        self.paused_message.grid(row=1, column=1, rowspan=2, columnspan=8)

    def toggle_agent(self, *args):
        # Turn the agent on or off so you can see how the game evolves
        #   with or without the agent
        self.agent_enabled = not self.agent_enabled
        self.display_data()
        self.display_agent_view()

        # Toggle the checkbox graphic if activated by pressing
        #   the "e" key instead of clicking on the button
        for event in args:
            if event.char == "e":
                self.toggle_agent_button.toggle()

    def toggle_manual_mode(self, *args):
        # Enable or disable manual control over the agent
        self.manual_mode = not self.manual_mode

        # Pause the game when entering manual mode
        if self.manual_mode:
            self.stop_game()

        # Toggle the checkbox graphic if activated by pressing
        #   the "m" key instead of clicking on the button
        for event in args:
            if event.char == "m":
                self.manual_mode_button.toggle()

    def update_agent(self):
        # Update visual data
        self.agent.vision.update(self.data)
        viewdata = self.agent.vision.get_viewdata()
        # Treat cells off the edge of the environment grid as dead: 1, not None
        viewdata = [1 if x is None else x for x in viewdata]

        # Update reward information
        reward = self.reward_scheme.get_reward()
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        self.avg_running_reward.append(
            sum(self.reward_window) / (len(self.reward_window) + 1.0))
        self.last_reward = reward

        # Take action
        action = self.agent.update(reward, viewdata)
        if not self.manual_mode:
            self.act(action)

        # Update displays
        self.update()
        self.refresh_data_view()
        self.update_meters()

    def update_meters(self):
        # Update "meter" readouts in the display console
        #
        # Update reward meter
        #  - Resize rectangle and change color in proportion to the last reward
        reward_meter_scale = 10
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


class SettingsWindow(Frame):
    # Popup allowing user to change settings
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
        self.grid_width_entry.insert(0, str(self.main_window.size[1]))

        grid_height_label = Label(s, text="Environment height:", bg="#CCCCCC")
        grid_height_label.grid(row=2, column=0, sticky="e")
        self.grid_height_entry = Entry(s, width=3)
        self.grid_height_entry.bind("<Return>", self.execute)
        self.grid_height_entry.grid(row=2, column=1, sticky="w")
        self.grid_height_entry.insert(0, str(self.main_window.size[0]))

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
        # Apply changes and close the window
        speed = int(self.speed_entry.get())
        grid_width = int(self.grid_width_entry.get())
        grid_height = int(self.grid_height_entry.get())
        scale = int(self.scale_entry.get())
        self.main_window.change_agent_speed(speed)
        self.main_window.change_environment_grid(grid_width, grid_height,
                                                 scale)
        self.top.destroy()


class QuitWindow(Frame):
    # Popup asking for confirmation before quitting the program
    #   - Automatically saves the brain file as "last_brain.pth"
    def __init__(self, master, agent):
        self.master = master
        self.agent = agent
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
        self.agent.save(Path.cwd() / "Saved_brains" / "last_brain.pth")
        print("Brain saved as \"last_brain.pth\"")
        self.top.destroy()
        quit()

    def cancel(self, *args):
        self.cancelled = True
        self.top.destroy()


class RewardSchemeWindow(Frame):
    # Popup window for choosing a new Reward Scheme
    #   - Changing the reward scheme DOES NOT reset the agent's brain.
    #     To start with a fresh brain, close and re-open the program.
    #   - Selecting the "shape" option and
    def __init__(self, master, main_window):
        self.master = master
        self.main_window = main_window
        self.schemetype = self.main_window.reward_scheme.schemetype
        self.shapename = None
        self.shape = None
        top = self.top = Toplevel(master)
        self.top.title("Reward Scheme Editor")

        msg = Label(top, text="Choose a new reward scheme:", pady=10)
        msg.grid(row=0, columnspan=3)

        self.variable = StringVar(top)
        self.reward_scheme_list = ["None",
                                   "Maximize Life",
                                   "Minimize Life",
                                   "Make shape"]

        self.variable.set(self.reward_scheme_list[0])

        self.reward_scheme_menu = OptionMenu(top, self.variable,
                                             *self.reward_scheme_list)
        self.reward_scheme_menu.grid(row=1, columnspan=3)

        # Buttons
        ok_button = Button(top, text="Ok", command=self.execute)
        ok_button.bind("<Return>", self.execute)
        ok_button.grid(row=3, column=0, sticky="e", pady=10)

        cancel_button = Button(top, text="Cancel", command=self.cancel)
        cancel_button.bind("<Return>", self.cancel)
        cancel_button.grid(row=3, column=2, sticky="w")

        # Key bindings
        self.top.bind("<Escape>", self.cancel)
        self.top.bind("<Return>", self.execute)

    def apply(self, *args):
        new_reward_scheme = self.variable.get()

        if new_reward_scheme == "None":
            self.schemetype = None
        elif new_reward_scheme == "Maximize Life":
            self.schemetype = "maximize"
        elif new_reward_scheme == "Minimize Life":
            self.schemetype = "minimize"
        elif new_reward_scheme == "Make shape":
            self.schemetype = "shape"
            shape = np.copy(self.main_window.vis_field)
            for i in range(len(shape)):
                for j in range(len(shape[i])):
                    if shape[i][j] == 2:
                        shape[i][j] = 1
            vf = np.copy(self.main_window.vis_field)
            popup = GridEditorWindow(self.master, self.main_window, self,
                                     shape, "reward scheme shape", vf)
            self.master.wait_window(popup.top)

        self.main_window.initialize_reward_scheme(self.schemetype,
                                                  self.shapename,
                                                  self.shape)

    def cancel(self, *args):
        self.top.destroy()

    def execute(self, *args):
        # Apply changes and close the window
        self.apply()
        self.top.destroy()


class GridEditorWindow(Frame):
    def __init__(self, master, main_window, reward_scheme_window,
                 input_grid, grid_type, limiter_grid=None):
        self.master = master
        self.main_window = main_window
        self.reward_scheme_window = reward_scheme_window
        self.grid = np.copy(input_grid)
        self.grid_backup = np.copy(input_grid)
        self.grid_type = grid_type
        self.limiter_grid = limiter_grid
        top = self.top = Toplevel(master)
        self.top.title("{} Editor".format(self.grid_type.capitalize()))

        msg = Label(top, text=" Edit the {} by\nclicking on the cells below"
                    .format(self.grid_type), pady=10)
        msg.grid(row=0, columnspan=3)

        if self.grid_type == "visual field":
            # Include expand grid button
            self.expand_button = Button(top, text="Expand grid",
                                        command=self.grid_expand)
            self.expand_button.grid(row=1, column=1)

            # Include default visual field sizes option menu
            self.defaults_label = Label(top, text="Starting size:")
            self.defaults_label.grid(row=3, column=0, columnspan=2, sticky="E")
            self.variable = StringVar(top)
            self.vf_list = ["Small",
                            "Medium",
                            "Large",
                            "Extra Large"]
            self.variable.set(self.vf_list[2])
            self.vf_menu = OptionMenu(top, self.variable, *self.vf_list)
            self.vf_menu.grid(row=3, column=2, pady=10, sticky="W")
            # Make it so selecting an option calls select_default_size function
            self.variable.trace("w", self.select_starting_size)

        elif self.grid_type == "reward scheme shape":
            # Include entry field for shape name
            self.shapename_label = Label(top, text="Shape name:")
            self.shapename_label.grid(row=1, column=0)
            self.shapename_entry = Entry(top, width=20)
            self.shapename_entry.grid(row=1, column=1, columnspan=2)

        # Frame for holding clickable grid cells for editing
        self.click_frame = Frame(top)
        self.clickable_grid = [[]]
        self.cell_size = 20
        self.refresh_grid()

        # Main buttons
        ok_button = Button(top, text="Ok", command=self.execute)
        ok_button.bind("<Return>", self.execute)
        ok_button.grid(row=4, column=0, sticky="e", pady=10)

        if self.grid_type == "visual field":
            apply_button = Button(top, text="Apply", command=self.apply)
            apply_button.bind("<Return>", self.apply)
            apply_button.grid(row=4, column=1)

        cancel_button = Button(top, text="Cancel", command=self.cancel)
        cancel_button.bind("<Return>", self.cancel)
        cancel_button.grid(row=4, column=2, sticky="w")

        # Key bindings
        self.top.bind("<e>", self.grid_expand)
        self.top.bind("<Escape>", self.cancel)
        self.top.bind("<Return>", self.execute)

    def apply(self, *args):
        # Apply changes to the original grid to be edited
        #   - Changes vary based on grid type
        if self.grid_type == "reward scheme shape":
            self.reward_scheme_window.shape = self.get_grid()
            self.reward_scheme_window.shapename = self.get_shapename()

        elif self.grid_type == "visual field":
            self.trim()
            self.main_window.vis_field = np.copy(self.grid)
            self.main_window.initialize_agent()
            self.main_window.initialize_reward_scheme()
            self.main_window.refresh_data_view()

        # Update backup grid
        self.grid_backup = np.copy(self.grid)

    def cancel(self, *args):
        # Restore visual field to the way it was since opening editor or
        #   since last clicking "Apply" and then close the editor window
        self.grid = np.copy(self.grid_backup)
        self.top.destroy()

    def execute(self, *args):
        # Apply changes and close the editor window
        self.apply()
        self.top.destroy()

    def flip(self, row, col):
        # Change the state of the cell at [row, column]
        #   - Changes vary based on grid type

        if self.grid_type == "reward scheme shape":
            # Only allow flipping a cell if it's within the visual field
            if self.limiter_grid[row][col] == 0:
                return

            if self.grid[row][col] == 0:
                self.grid[row][col] = 1
                render = self.render_cell_image(1)
            elif self.grid[row][col] == 1:
                self.grid[row][col] = 0
                render = self.render_cell_image(2)

        elif self.grid_type == "visual field":
            if self.grid[row][col] == 0:
                self.grid[row][col] = 1
            elif self.grid[row][col] == 1:
                self.grid[row][col] = 0
            render = self.render_cell_image(self.grid[row][col])

        self.clickable_grid[row][col].configure(image=render)
        self.clickable_grid[row][col].image = render

    def get_grid(self):
        return self.grid

    def get_shapename(self):
        return self.shapename_entry.get()

    def grid_expand(self, *args):
        # Expand the grid dimensions by one cell in all four directions
        if self.grid_type == "visual field":
            new_grid = np.zeros((len(self.grid)+2, len(self.grid[0])+2))
            for i in range(len(self.grid)):
                for j in range(len(self.grid[i])):
                    new_grid[i + 1][j + 1] = self.grid[i][j]
            self.grid = np.copy(new_grid)
            self.refresh_grid()

        # Grid expand not allowed when editing reward scheme shape
        else:
            return

    def refresh_grid(self):
        # Destroy old grid
        for i in range(len(self.clickable_grid)):
            for j in range(len(self.clickable_grid[i])):
                self.clickable_grid[i][j].destroy()
        self.click_frame.destroy()

        # Create new clickable grid
        self.click_frame = Frame(self.top, borderwidth=4,
                                 relief="ridge", bg="gray")
        self.click_frame.grid(row=2, columnspan=3, padx=20, pady=20)
        self.clickable_grid = [[] for row in self.grid]

        # Create a clickable square image for each cell in the grid
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):

                # Display square colors differently based on grid type
                if self.grid_type == "reward scheme shape":
                    # Reward scheme grid is limited by the visual field
                    if self.limiter_grid[i][j] == 0:
                        render = self.render_cell_image(0)
                    elif self.grid[i][j] == 0:
                        render = self.render_cell_image(2)
                    elif self.grid[i][j] == 1:
                        render = self.render_cell_image(1)

                elif self.grid_type == "visual field":
                    render = self.render_cell_image(self.grid[i][j])

                else:
                    render = self.render_cell_image(0)

                cell_image = Label(self.click_frame, image=render,
                                   borderwidth=1, bg="#EEEEEE")
                self.clickable_grid[i].append(cell_image)
                self.clickable_grid[i][j].image = render
                self.clickable_grid[i][j].grid(row=i, column=j)
                self.clickable_grid[i][j].bind("<Button-1>",
                                               lambda event, row=i, col=j:
                                               self.flip(row, col))

    def render_cell_image(self, cell):
        # Create a grayscale image of a square with color determined by <cell>
        if cell == 0:
            # White
            colorweighted_data = 255
        elif cell == 1:
            # Light gray
            colorweighted_data = 200
        else:
            # Dark gray
            colorweighted_data = 100

        scaled_data = np.kron(colorweighted_data,
                              np.ones((self.cell_size, self.cell_size)))
        raw_img = Image.fromarray(scaled_data)
        render = ImageTk.PhotoImage(raw_img)
        return render

    def select_starting_size(self, *args):
        new_grid = self.variable.get()

        if new_grid == "Small":
            self.grid = np.copy(small_vf)
        elif new_grid == "Medium":
            self.grid = np.copy(medium_vf)
        elif new_grid == "Large":
            self.grid = np.copy(large_vf)
        elif new_grid == "Extra Large":
            self.grid = np.copy(extralarge_vf)

        self.refresh_grid()

    def trim(self):
        # Trim any outermost rows or columns that have only zeros
        # Example:  0 0 0 0 0                   1 0 1 1
        #           1 0 1 1 0       --->        0 0 0 0
        #           0 0 0 0 0                   0 0 1 0
        #           0 0 1 0 0
        trimmed_grid = np.copy(self.grid)
        # Top rows
        all_zeros = True
        while(all_zeros):
            for i in trimmed_grid[0]:
                if i == 1 or i == 2:
                    all_zeros = False
            if all_zeros:
                trimmed_grid = np.copy(trimmed_grid[1:])
        # Bottom rows
        all_zeros = True
        while(all_zeros):
            for i in trimmed_grid[-1]:
                if i == 1 or i == 2:
                    all_zeros = False
            if all_zeros:
                trimmed_grid = np.copy(trimmed_grid[:-1])
        # Left columns
        all_zeros = True
        while(all_zeros):
            for i in trimmed_grid:
                if i[0] == 1 or i[0] == 2:
                    all_zeros = False
            if all_zeros:
                trimmed_grid = np.copy(trimmed_grid[:, 1:])
        # Right columns
        all_zeros = True
        while(all_zeros):
            for i in trimmed_grid:
                if i[-1] == 1 or i[-1] == 2:
                    all_zeros = False
            if all_zeros:
                trimmed_grid = np.copy(trimmed_grid[:, :-1])
        self.grid = np.copy(trimmed_grid)
        self.refresh_grid()


root = Tk()
root.title("A Demon In Conway's Game Of Life")
MainWindow(root)
root.mainloop()
