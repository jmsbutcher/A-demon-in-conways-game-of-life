# A Demon in Conway's Game of Life

## Conway's Game of Life as GUI-based Environment for Artificial Intelligence

Life's evolution has run without interference... until now.

An intelligent agent lives within the game, moving around
and flipping cells --- causing chaos, or perhaps
some order!

The agent has a "brain," a neural network that learns via
reinforcement learning. It "sees" the state of cells in the
environment within its visual field. This information is
fed as input into the neural net, which then outputs an action: 
move, flip, or wait.

Depending on the reward scheme you choose, the agent learns to
accomplish different things: maximize life, minimize life, or
create definite shapes.

#### Purpose
As of right now, this project is meant for experimenting and playing
around with how Conway's Game of Life works with or without
interference

#### Mission
The next stage of development for this project will be to make
the AI system customizeable. Experiment with different parameters
for the neural network (more layers, less neurons per layer, convolutional
instead of linear, and so on) and see how it affects the agent's behavior.
The goal for this project is to become a sort of studio for developing 
and testing AI systems.

<br>

## Prerequisites:
<ul>
  <li>Python 3.7</li>
  <li>Numpy 1.17</li>
  <li>Pytorch 1.2.0</li>
</ul>

## Setup:
1. Make sure the above requirements have been installed
2. Clone or download this repository
3. Using the terminal, navigate into the folder containing this repository
4. Run this command:
```
python game.py
```

- If you are using Windows 10, you may need to download Anaconda and use
**Anaconda Prompt** instead of **Command Prompt** to install Pytorch and
also to run the above command.

<br>

# Usage:

### The GUI Interface
Once you run the command `python game.py`, this GUI should pop up:

<img src="https://github.com/jmsbutcher/A-demon-in-conways-game-of-life/blob/master/Usage_images/Empty_GUI_Labelled2.png">

- The **Game Environment** is where the grid of cells live and die
- The **Agent view display** shows a close-up of the agent's field of view when enabled
- The **Reward meter** shows the amount of reward the agent receives on a given game step
- The **Reward scheme** shows the current reward mechanism for the agent

<br>

### Original Conway's Game of Life

The game starts with the demon disabled by default. This is a good opportunity to become familiar with the GUI interface and game control buttons. Also, if you are unfamiliar with Conway's Game of Life, here you can see how it works normally without interference.

To begin, click the "Seed" button to generate some live cells in the middle of the environment grid.

<p align="center">
  <img width="320" height="280" src="https://github.com/jmsbutcher/A-demon-in-conways-game-of-life/blob/master/Usage_images/Seed.png">
</p>

Then click the "Step" button to see the cells change based on the [rules of the game](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life). This is the next "generation" of cells. Notice that the Generation # has increased by one --- located in the top right corner of the window.

To run the game continuously, click the "Start" button, or press the Spacebar. You can speed up the game by clicking the "Speed Up" button or pressing the P key, and slow it down by clicking the "Slow Down" button or pressing the O key. These commands with their associated keyboard shortcuts can also be accessed via the "Run" menu.

You may notice that the cells reach a steady state fairly quickly. Left to itself, they will stay that way forever. That will change once we enable the agent.

<br>

### Settings

Access the Settings Menu by clicking "File" --> "Settings" or by pressing the S key.

Here you can change the height and width of the environment grid, measured in number of cells.

You can also change the "Scale" of each cell, making all of them bigger or smaller to better fit your screen.

<br>

### Enabling the Demon

To introduce the agent into the environment, click "Enable Demon" in the bottom left corner of the window or press the E key.

The agent should appear in the center of the grid like so:

<p align="center">
  <img src="https://github.com/jmsbutcher/A-demon-in-conways-game-of-life/blob/master/Usage_images/DemonEnabled.png">
</p>

The gray cells represent the agent's **visual field**. The agent can "sense" whether each cell within the visual field is alive or dead. This information is fed into its "brain" and will be used later for decision making. A close-up of the agent's current view is displayed in the white square to the upper right.

The white square in the center of the visual field is the agent's **eye location**. This is the one and only square the agent uses to move around and interact with the environment. At any given time, the agent may "flip" a cell from black to white or vice versa at that location.

<p align="center">
  <img src="https://github.com/jmsbutcher/A-demon-in-conways-game-of-life/blob/master/Usage_images/DemonCloseUpLabelled.png">
</p>

<br>

### Manual Mode

Take manual control over the agent by cliking "Manual Mode" in the bottom left of the window or by pressing the M key.
Now you can move the agent and flip cells wherever you want. This is useful for creating custom cell state arrangements.

- **Move Up, Down, Left, and Right --- Arrow keys**
- **Flip cell --- C key**
- **Advance game by 1 step --- Spacebar** (To run game continuously, you must click the "Start" button while in Manual Mode)

<br>

### Automatic Mode

Time to unleash the demon :smiling_imp: 

With the agent enabled and Manual Mode disabled, start the game and watch the chaos begin. The agent will begin moving around and flipping cells, disrupting the natural evolution of the game. The agent ensures the cells will never reach a steady state.

- There are six possible agent actions:
  - Move up
  - Move down
  - Move left
  - Move right
  - Flip cell
  - Wait until next game step

- These actions are random until you choose a **Reward Scheme** (see below).

- The **agent speed** is the maximum number of actions the agent takes per game step. If the agent selects the *Wait* action, the next generation proceeds immediately. For example, if the agent speed is 5, the actions may look like this:

Generation | Actions
---------- | -------
0 | Up - Up - Left - Flip - Right
1 | Flip - Down - Flip - Left - **Wait**
2 | Right - **Wait**
3 | **Wait**
4 | Down - Right - Up - Up - Flip

- The agent speed is 10 by default, therefore it takes up to 10 actions per game step. This can be changed in the Settings menu. 

 <br>
 
 ### Choosing a Reward Scheme
 
 A *Reward Scheme* is a system for assigning rewards to the agent. **Once you choose a Reward Scheme, the agent will begin learning**. It will no longer act randomly (although it may look like it at first.) It will try to maximize the reward by moving and flipping cells based on the output of the neural network. 
 
 To choose a new Reward Scheme, click the button in the display console, or open the "New" menu and select "Reward Scheme", or press the R key.
 
<p align="center">
  <img width="280" height="220" src="https://github.com/jmsbutcher/A-demon-in-conways-game-of-life/blob/master/Usage_images/NewRewardSchemeMenu.png">
</p>
 
 There are currently 3 Reward Schemes types to choose from. They are all determined by the state of the cells within the agent's visual field:
 - Maximize Life - More live cells means higher reward
 - Minimize Life - More live cells means lower reward
 - Make shape - Exact match yields maximum reward. Partial match yields less reward.

<img src="https://github.com/jmsbutcher/A-demon-in-conways-game-of-life/blob/master/Usage_images/WithMaximizeReward_Labelled3.png">


