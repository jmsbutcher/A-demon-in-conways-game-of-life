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


# Usage:

### The GUI Interface
Once you run the command `python game.py`, this GUI should pop up:
<>


### Original, Demon-Free Conway's Game of Life
