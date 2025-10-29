from joblib import Parallel, delayed
import joblib
from Player import Player
from SimpleNeuralNetwork import SimpleNeuralNetwork
from Game import Game2048Env
from typing import List
from typing import Tuple
import random
import copy


class EvolutionaryOptimizer:
    def __init__(
        self,
        population_size: int = 50,
        elite_size: int = 10,
        new_members: int = 10,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.5,
        hidden_layers: List[int] = [32],
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

    def evaluate(
        self,
        games_per_player: int = 5,
        max_steps: int = 100,
        base_random_seed: int = 42,
    ) -> List[Tuple[SimpleNeuralNetwork, int, float]]:
        def eval_network(network, base_random_seed):
            scores = []
            best_tile = 0
            for game_index in range(games_per_player):
                player = Player(network)
                env = Game2048Env(random_seed=base_random_seed + game_index)
                game_result = player.play(env, max_steps=max_steps)
                scores.append(game_result.score)
                if game_result.max_tile > best_tile:
                    best_tile = game_result.max_tile
            return (network, best_tile, sum(scores) / games_per_player)

        results = Parallel(n_jobs=joblib.cpu_count())(
            delayed(eval_network)(net, base_random_seed) for net in self.population
        )

        return results

    def select_and_breed(
        self, evaluated: List[Tuple[SimpleNeuralNetwork, int, float]]
    ) -> None:
        # Sort by score descending
        evaluated.sort(key=lambda x: x[2], reverse=True)
        elite = evaluated[: self.elite_size]

        new_population = []
        # Keep elite networks
        for net, _, _ in elite:
            net.age += 1
            new_population.append(net)

        # Create offspring by mutating elite networks
        while len(new_population) < self.population_size:
            parent = random.choice(elite)[0]

            # Create a child by copying the parent's state
            child = copy.deepcopy(parent)
            child.age = 0
            # child.load_state_dict(parent.state_dict())

            # Mutate the child
            child.mutate(self.mutation_rate, self.mutation_strength)
            new_population.append(child)

        # Add random new members
        for _ in range(self.new_members):
            network = SimpleNeuralNetwork(hidden_layers=self.hidden_layers)
            new_population.append(network)

        self.population = new_population

    def run_generation(
        self,
        games_per_player: int = 5,
        max_steps: int = 1000,
        base_random_seed: int = 42,
    ) -> Tuple[List[SimpleNeuralNetwork], float, int]:
        evaluated = self.evaluate(
            games_per_player, max_steps=max_steps, base_random_seed=base_random_seed
        )
        elites = evaluated[: self.elite_size]
        avg_score = sum(avg_score_player for _, _, avg_score_player in elites) / len(
            elites
        )
        best_score = max(score for _, _, score in elites)

        self.select_and_breed(evaluated)

        return self.population, avg_score, best_score
