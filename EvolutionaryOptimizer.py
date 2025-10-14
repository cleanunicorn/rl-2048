from joblib import Parallel, delayed
import joblib
from Player import Player
from SimpleNeuralNetwork import SimpleNeuralNetwork
from Game import Game2048Env
from typing import List
from typing import Tuple
import random
import numpy as np
import numpy

class EvolutionaryOptimizer:
    def __init__(
            self, 
            population_size: int = 50, 
            elite_size: int = 10,
            new_members: int = 10,
            mutation_rate: float = 0.1, 
            mutation_strength: float = 0.5,
            hidden_layers: List[int] = [32]
        ):
        self.population_size = population_size
        self.elite_size = elite_size
        self.new_members = new_members
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.hidden_layers = hidden_layers
        
        # Create initial population
        self.population = []
        for _ in range(population_size):
            network = SimpleNeuralNetwork(hidden_layers=hidden_layers)
            self.population.append(network)

    def evaluate(self, env: Game2048Env, games_per_player: int = 5, max_steps: int = 100) -> List[Tuple[SimpleNeuralNetwork, int, int]]:
        def eval_network(network):
            # network_cpu = network.cpu()
            player = Player(network)
            env = Game2048Env()
            total_score = 0
            best_score = 0
            best_tile = 0
            for _ in range(games_per_player):
                game_result = player.play(env, max_steps=max_steps)
                total_score += game_result.score
                if game_result.score > best_score:
                    best_score = game_result.score
                if game_result.max_tile > best_tile:
                    best_tile = game_result.max_tile
            avg_score = total_score / games_per_player
            return (network, best_tile, best_score)

        results = Parallel(n_jobs=joblib.cpu_count(), prefer="threads")(delayed(eval_network)(net) for net in self.population)

        return results

    def select_and_breed(self, evaluated: List[Tuple[SimpleNeuralNetwork, float, int]]) -> None:
        # Sort by score descending
        evaluated.sort(key=lambda x: x[1], reverse=True)
        elite = evaluated[:self.elite_size] 

        new_population = []
        # Keep elite networks
        for net, _, _ in elite:
            new_population.append(net)
        
        # Create offspring by mutating elite networks
        while len(new_population) < self.population_size:
            parent = random.choice(elite)[0]
            
            # Create a child by copying the parent's state
            child = SimpleNeuralNetwork(hidden_layers=self.hidden_layers)
            child.load_state_dict(parent.state_dict())
            
            # Mutate the child
            child.mutate(self.mutation_rate, self.mutation_strength)
            new_population.append(child)

        # Add random new members
        for _ in range(self.new_members):
            network = SimpleNeuralNetwork(hidden_layers=self.hidden_layers)
            new_population.append(network)    

        self.population = new_population[:self.population_size]

    def run_generation(self, env: Game2048Env, games_per_player: int = 5, max_steps: int = 1000) -> Tuple[List[SimpleNeuralNetwork], float, int]:
        evaluated = self.evaluate(env, games_per_player, max_steps=max_steps)
        avg_max_tile = sum(max_tile for _, max_tile, _ in evaluated) / len(evaluated)
        best_score = max(score for _, _, score in evaluated)
        
        self.select_and_breed(evaluated)
        
        return self.population, avg_max_tile, best_score