# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 13:31:39 2018

@author: Pauline LAURON
"""
from flappybird.game import FlappyGame
import neat

MAX_FITNESS = 0
BEST_GENOME = None

def game(genome, config):
    
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    bird = FlappyGame(return_rgb=False, display_screen=False)
    fitness = 0 
    
    state = bird.get_game()
    
    while True: 
        
        output = net.activate(state)
        
        reward, state, done = bird.do_action()
        
        fitness += reward
        
        if done :
            break        
    
    return fitness
        
        

def eval_genomes(genomes, config):
    global MAX_FITNESS, BEST_GENOME
    
    for genome_id, genome in genomes:
        #genome.fitness = 0 
        genome.fitness = game(genome, config)
        
        if genome.fitness >= MAX_FITNESS:
            MAX_FITNESS = genome.fitness
            BEST_GENOME = genome

def run(config_file):
    # Load configuration.
    config = neat.Config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config_file')
    
    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    
    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)
    
    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    
if __name__ == '__main__':
    run('config')