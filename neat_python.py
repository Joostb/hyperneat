# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 13:31:39 2018

@author: Pauline LAURON
"""
from flappybird.game import FlappyGame, normalize_state
import neat
import visualize 

MAX_FITNESS = 0
BEST_GENOME = None
GENERATION = 0

def game(genome, config):
    
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    bird = FlappyGame(return_rgb=False, display_screen=False, reward_clipping=True)
    fitness = 1
    
    state = normalize_state(bird.get_state())
    actions = bird.game.getActionSet()
    
    while True: 
        
        output = net.activate(state)
        
        if output[0] >= 0.5:
            #up
            reward, state, done = bird.do_action(actions[0])
        else :
            #do nothing
            reward, state, done = bird.do_action(actions[1])
            
        state = normalize_state(state)
        
        fitness += reward
        
        if done :
            state = bird.reset()
            break        
    
    return fitness
        
        

def eval_genomes(genomes, config):
    global MAX_FITNESS, BEST_GENOME, GENERATION
     
    i = 0
    for genome_id, genome in genomes:
        #genome.fitness = 0 
        genome.fitness = game(genome, config)
        #print("Gen : %d Genome # : %d  Fitness : %f Max Fitness : %f"%(GENERATION,i,genome.fitness, MAX_FITNESS))
        if genome.fitness >= MAX_FITNESS:
            MAX_FITNESS = genome.fitness
            BEST_GENOME = genome
        i += 1
    GENERATION += 1

def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))
    
    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 30)
    
    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))
    
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    bird = FlappyGame(return_rgb=False, display_screen=False, reward_clipping=True)
    fitness = 1
    
    state = normalize_state(bird.get_state())
    actions = bird.game.getActionSet()
    
    while True: 
        
        output = winner_net.activate(state)
        
        if output[0] >= 0.5:
            #up
            reward, state, done = bird.do_action(actions[0])
        else :
            #do nothing
            reward, state, done = bird.do_action(actions[1])
            
        state = normalize_state(state)
        
        fitness += reward
        
        if done :
            state = bird.reset()
            break        
        
    visualize.draw_net(config, winner, True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)
    
    
    #p.run(eval_genomes, 10)
    

    
if __name__ == '__main__':
    run('config')