
"""
Neural Network Evolution for the Cow Path Problem
=================================================

This program evolves neural networks to solve the classic "Cow Path Problem" 
(also known as the Linear Search Problem) where an agent must find a target 
at unknown distance on a line, using strategies that compete against the 
mathematically optimal doubling strategy.

The evolved networks learn search strategies through competitive evolution,
potentially discovering variations of or improvements to known optimal approaches.
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import copy
from dataclasses import dataclass
from enum import Enum

class Direction(Enum):
    LEFT = -1
    RIGHT = 1

@dataclass
class SearchResult:
    found: bool
    total_distance: float
    target_distance: float
    competitive_ratio: float
    steps_taken: int
    path: List[Tuple[float, Direction]]

class CowPathEnvironment:
    """Environment implementing the Cow Path Problem"""

    def __init__(self, target_distance: float, max_fuel_multiplier: float = 10.0):
        self.target_distance = abs(target_distance)
        self.target_side = 1 if target_distance > 0 else -1
        self.max_fuel = self.target_distance * max_fuel_multiplier

    def run_search(self, decisions: List[Tuple[float, Direction]]) -> SearchResult:
        """Run a search given a sequence of (distance, direction) decisions"""
        position = 0.0
        total_distance = 0.0
        path = []

        for step, (distance, direction) in enumerate(decisions):
            start_pos = position
            position += distance * direction.value
            total_distance += distance
            path.append((distance, direction))

            # Check if we found the target
            if (self.target_side == 1 and position >= self.target_distance) or \
               (self.target_side == -1 and position <= -self.target_distance):
                exact_distance_to_target = abs(abs(position) - self.target_distance)
                total_distance -= exact_distance_to_target
                competitive_ratio = total_distance / self.target_distance
                return SearchResult(True, total_distance, self.target_distance, 
                                  competitive_ratio, step + 1, path)

            # Check if we ran out of fuel
            if total_distance >= self.max_fuel:
                return SearchResult(False, total_distance, self.target_distance, 
                                  float('inf'), step + 1, path)

            # Return to origin (essential for cow path strategy)
            if position != 0:
                total_distance += abs(position)
                position = 0.0

        return SearchResult(False, total_distance, self.target_distance, 
                          float('inf'), len(decisions), path)

class SimpleNeuralNetwork:
    """Simple feedforward neural network for the cow path problem"""

    def __init__(self, input_size: int = 4, hidden_sizes: List[int] = [8, 6], output_size: int = 2):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        self.weights = []
        self.biases = []

        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.weights.append(np.random.randn(prev_size, hidden_size) * 0.5)
            self.biases.append(np.random.randn(hidden_size) * 0.5)
            prev_size = hidden_size

        self.weights.append(np.random.randn(prev_size, output_size) * 0.5)
        self.biases.append(np.random.randn(output_size) * 0.5)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def tanh(self, x):
        return np.tanh(np.clip(x, -500, 500))

    def forward(self, inputs):
        """Forward pass through the network"""
        activation = np.array(inputs)

        for i, (weight, bias) in enumerate(zip(self.weights[:-1], self.biases[:-1])):
            activation = self.tanh(np.dot(activation, weight) + bias)

        output = np.dot(activation, self.weights[-1]) + self.biases[-1]
        return output

    def get_decision(self, state):
        """Get a search decision based on current state"""
        raw_output = self.forward(state)

        distance = np.exp(raw_output[0]) * 2.0
        direction_prob = self.sigmoid(raw_output[1])
        direction = Direction.RIGHT if direction_prob > 0.5 else Direction.LEFT

        return distance, direction

    def copy(self):
        """Create a deep copy of this network"""
        new_net = SimpleNeuralNetwork(self.input_size, self.hidden_sizes, self.output_size)
        new_net.weights = [w.copy() for w in self.weights]
        new_net.biases = [b.copy() for b in self.biases]
        return new_net

    def mutate(self, mutation_rate=0.1, mutation_strength=0.1):
        """Mutate the network weights and biases"""
        for i in range(len(self.weights)):
            mask = np.random.random(self.weights[i].shape) < mutation_rate
            self.weights[i] += mask * np.random.randn(*self.weights[i].shape) * mutation_strength

            mask = np.random.random(self.biases[i].shape) < mutation_rate
            self.biases[i] += mask * np.random.randn(*self.biases[i].shape) * mutation_strength

class CowPathAgent:
    """Agent that uses a neural network to make search decisions"""

    def __init__(self, network: SimpleNeuralNetwork, max_decisions: int = 20):
        self.network = network
        self.max_decisions = max_decisions

    def search(self, environment: CowPathEnvironment) -> SearchResult:
        """Perform a search using the neural network"""
        decisions = []
        current_step = 0
        total_distance = 0.0
        last_direction = 0
        last_distance = 0.0

        for step in range(self.max_decisions):
            state = [
                current_step / self.max_decisions,
                min(total_distance / 100.0, 1.0),
                (last_direction + 1) / 2.0,
                min(last_distance / 10.0, 1.0)
            ]

            distance, direction = self.network.get_decision(state)
            distance = max(0.1, min(distance, 50.0))

            decisions.append((distance, direction))

            current_step += 1
            total_distance += distance * 2
            last_direction = direction.value
            last_distance = distance

            if total_distance > environment.max_fuel:
                break

        return environment.run_search(decisions)

class OptimalDoublingAgent:
    """Implementation of the optimal doubling strategy (competitive ratio = 9)"""

    def search(self, environment: CowPathEnvironment) -> SearchResult:
        decisions = []
        search_distance = 1.0
        direction = Direction.RIGHT

        while True:
            decisions.append((search_distance, direction))

            result = environment.run_search(decisions)
            if result.found or not result.found and result.total_distance >= environment.max_fuel:
                return result

            search_distance *= 2
            direction = Direction.LEFT if direction == Direction.RIGHT else Direction.RIGHT

class EvolutionaryOptimizer:
    """Evolutionary algorithm to evolve neural networks"""

    def __init__(self, population_size: int = 50, elite_size: int = 10, 
                 mutation_rate: float = 0.2, mutation_strength: float = 0.15):
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.population = []
        self.fitness_history = []
        self.best_agent = None
        self.best_fitness = float('inf')

    def initialize_population(self):
        """Create initial random population"""
        self.population = []
        for _ in range(self.population_size):
            network = SimpleNeuralNetwork()
            agent = CowPathAgent(network)
            self.population.append(agent)

    def evaluate_fitness(self, test_cases: List[float]) -> List[float]:
        """Evaluate fitness of all agents across multiple test cases"""
        fitness_scores = []

        for agent in self.population:
            total_fitness = 0.0
            successful_searches = 0

            for target_distance in test_cases:
                env = CowPathEnvironment(target_distance)
                result = agent.search(env)

                if result.found:
                    fitness = result.competitive_ratio
                    successful_searches += 1
                else:
                    fitness = 100.0

                total_fitness += fitness

            avg_fitness = total_fitness / len(test_cases)
            success_rate = successful_searches / len(test_cases)
            final_fitness = avg_fitness + (1 - success_rate) * 50.0
            fitness_scores.append(final_fitness)

        return fitness_scores

    def selection_and_reproduction(self, fitness_scores: List[float]):
        """Select best agents and create next generation"""
        population_fitness = list(zip(self.population, fitness_scores))
        population_fitness.sort(key=lambda x: x[1])

        if population_fitness[0][1] < self.best_fitness:
            self.best_fitness = population_fitness[0][1]
            self.best_agent = population_fitness[0][0].network.copy()

        elite = [agent for agent, _ in population_fitness[:self.elite_size]]
        new_population = []

        for agent in elite:
            new_population.append(CowPathAgent(agent.network.copy()))

        while len(new_population) < self.population_size:
            parent = random.choice(elite)
            child_network = parent.network.copy()
            child_network.mutate(self.mutation_rate, self.mutation_strength)
            child_agent = CowPathAgent(child_network)
            new_population.append(child_agent)

        self.population = new_population
        return population_fitness[0][1]

    def evolve(self, generations: int, test_cases: List[float]):
        """Run the evolutionary process"""
        print(f"Starting evolution for {generations} generations...")
        print(f"Test cases: {test_cases}")

        self.initialize_population()

        for generation in range(generations):
            fitness_scores = self.evaluate_fitness(test_cases)

            best_fitness = min(fitness_scores)
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            self.fitness_history.append((best_fitness, avg_fitness))

            self.selection_and_reproduction(fitness_scores)

            if generation % 10 == 0:
                print(f"Generation {generation}: Best={best_fitness:.3f}, Avg={avg_fitness:.3f}")

        print(f"Evolution complete! Best fitness: {self.best_fitness:.3f}")
        return self.best_agent

def test_strategy(agent, test_cases, strategy_name):
    """Test a strategy and return statistics"""
    results = []
    competitive_ratios = []
    success_count = 0

    for target_distance in test_cases:
        env = CowPathEnvironment(target_distance)
        result = agent.search(env)
        results.append(result)

        if result.found:
            competitive_ratios.append(result.competitive_ratio)
            success_count += 1

    if competitive_ratios:
        avg_ratio = sum(competitive_ratios) / len(competitive_ratios)
        max_ratio = max(competitive_ratios)
    else:
        avg_ratio = float('inf')
        max_ratio = float('inf')

    success_rate = success_count / len(test_cases)

    print(f"\n{strategy_name} Results:")
    print(f"  Success rate: {success_rate:.1%}")
    print(f"  Average competitive ratio: {avg_ratio:.3f}")
    print(f"  Max competitive ratio: {max_ratio:.3f}")

    return {
        'success_rate': success_rate,
        'avg_ratio': avg_ratio,
        'max_ratio': max_ratio,
        'results': results
    }

def main():
    """Main function to run the neural network evolution experiment"""

    # Training cases - include both positive and negative targets
    training_cases = [-20.0, -10.0, -5.0, -2.0, -1.0, 1.0, 2.0, 5.0, 10.0, 20.0]

    # Create evolutionary optimizer
    optimizer = EvolutionaryOptimizer(
        population_size=40,
        elite_size=8,
        mutation_rate=0.3,
        mutation_strength=0.2
    )

    # Run evolution
    best_network = optimizer.evolve(generations=60, test_cases=training_cases)

    # Test on validation set
    validation_cases = [-15.0, -7.0, -3.0, 3.0, 7.0, 15.0]

    print("\n" + "="*60)
    print("FINAL COMPARISON RESULTS")
    print("="*60)

    # Test evolved agent
    evolved_agent = CowPathAgent(best_network)
    evolved_stats = test_strategy(evolved_agent, validation_cases, "Evolved Neural Network")

    # Test optimal doubling strategy
    optimal_agent = OptimalDoublingAgent()
    optimal_stats = test_strategy(optimal_agent, validation_cases, "Optimal Doubling Strategy")

    # Plot fitness evolution
    plt.figure(figsize=(10, 6))
    generations = list(range(len(optimizer.fitness_history)))
    best_fitnesses = [f[0] for f in optimizer.fitness_history]
    avg_fitnesses = [f[1] for f in optimizer.fitness_history]

    plt.plot(generations, best_fitnesses, 'b-', label='Best Fitness', linewidth=2)
    plt.plot(generations, avg_fitnesses, 'r--', label='Average Fitness', linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score (Lower is Better)')
    plt.title('Neural Network Evolution for Cow Path Problem')
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_network, optimizer

if __name__ == "__main__":
    best_network, optimizer = main()
