# HyperNEAT
A repo for the NC course

# Report
The report may be found [here](https://www.sharelatex.com/8872427238cwcqgxhkhzny)

# Installation
Required packages are numpy, pygame, Keras, Pillow and tqdm.

Pygame installation:
* Unix
```bash
sudo python3 -m pip install -U pygame
```

* Windows
```bash
python -m pip install -U pygame --user
```

If this does not work you have to [compile from source](https://www.pygame.org/wiki/MingW?parent=).

# Run Instructions
The commands that should be ran for replicating the results are:
## DQN
```bash
python3 main_dqn_features.py --train --n-games 500
```
## NEAT
```bash
python3 neat_flappy_custom.py
```
## XOR
```bash
python3 xor_python.py
```
