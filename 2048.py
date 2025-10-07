import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum
import time


class Direction(Enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3

import numpy as np

class Game2048Env:
    def __init__(self):
        self.grid_size = 4
        self.reset()
        
    def reset(self):
        self.board = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.spawn_tile()
        self.spawn_tile()
        self.score = 0
        return self.board.copy()
    
    def spawn_tile(self):
        empty = list(zip(*np.where(self.board == 0)))
        if empty:
            x, y = empty[np.random.randint(len(empty))]
            self.board[x, y] = 2 if np.random.random() < 0.9 else 4
        
    def step(self, action: Direction):
        moved, reward = self.move(action.value)
        if moved:
            self.spawn_tile()
        else:
            reward = -1  # Penalty for invalid move
        done = not self.can_move()
        self.score += reward
        return self.board.copy(), reward, done, {}
    
    def move(self, direction):
        board = np.copy(self.board)
        reward = 0
        moved = False

        # Rotate board so all moves are left-moves
        for _ in range(direction):
            board = np.rot90(board)
            
        for i in range(self.grid_size):
            tiles = board[i][board[i] != 0]  # Extract non-zero
            merged = []
            j = 0
            while j < len(tiles):
                if j + 1 < len(tiles) and tiles[j] == tiles[j + 1]:
                    merged_val = tiles[j] * 2
                    reward += merged_val
                    merged.append(merged_val)
                    j += 2  # Skip next
                    moved = True
                else:
                    merged.append(tiles[j])
                    j += 1
            # Pad with zeros to the right
            merged += [0] * (self.grid_size - len(merged))
            # Detect if move or merge happened
            if not np.array_equal(board[i], merged):
                moved = True
            board[i] = merged

        # Restore original orientation
        for _ in range((4 - direction) % 4):
            board = np.rot90(board)
            
        if moved:
            self.board = board

        return moved, reward

    
    def can_move(self):
        for direction in range(4):
            temp_board = self.board.copy()
            moved, _ = self.move(direction)
            self.board = temp_board  # Restore original
            if moved:
                return True
        return False

# game = Game2048Env()
# state = game.reset()
# done = False

def print_board(board):
    for x in board:
        print("\t".join(f"{v:4}" for v in x))
    print("-" * 20)

# print_board(state)

# for _ in range(3):  # Play 10 random moves

#     action = Direction(np.random.randint(4))  # Random action for demonstration
#     state, reward, done, _ = game.step(action)

#     print(f"Action: {action.name} | Score: {game.score}")
#     print(f"Reward: {reward} | Done: {done}")
    
#     print_board(state)
    
class SimpleNeuralNetwork:
    """Simple feedforward netural network"""

    def __init__(self, input_size: int = 16, hidden_layers: List[int] = [8, 8], output_size: int = 4):
        self.layers = [input_size] + hidden_layers + [output_size]
        self.weights = []
        self.biases = []

        # Initialize weights and biases
        for i in range(len(self.layers) - 1):
            weight_matrix = np.random.randn(self.layers[i], self.layers[i+1]) * np.sqrt(2. / self.layers[i])
            bias_vector = np.zeros((self.layers[i+1],))
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network"""  
        a = x
        for i in range(len(self.weights) - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self.tanh(z)

        # Output layer with linear activation
        z = np.dot(a, self.weights[-1]) + self.biases[-1]
        return z
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.5):
        """Mutate the network's weights and biases"""
        for i in range(len(self.weights)):
            if random.random() < mutation_rate:
                mutation = np.random.randn(*self.weights[i].shape) * mutation_strength
                self.weights[i] += mutation
                
            if random.random() < mutation_rate:
                mutation = np.random.randn(*self.biases[i].shape) * mutation_strength
                self.biases[i] += mutation    

@dataclass
class GameResult:
    score: int
    max_tile: int
    moves: int

class Player:
    def __init__(self, network: SimpleNeuralNetwork):
        self.network = network

    def play(self, env: Game2048Env, max_steps: int = 100) -> int:
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = self.next_move(state)

            state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1

        return GameResult(score=total_reward, max_tile=np.max(state), moves=steps)
    
    def next_move(self, state: np.ndarray) -> Direction:
        flat_state = state.flatten() / 2048.0  # Normalize input
        q_values = self.network.forward(flat_state)
        action = Direction(np.argmax(q_values))  # Choose action with highest Q-value
        return action                

class EvolutionaryOptimizer:
    def __init__(
            self, 
            population_size: int = 50, 
            elite_size: int = 10,
            mutation_rate: float = 0.1, 
            mutation_strength: float = 0.5
        ):
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.population = [SimpleNeuralNetwork(hidden_layers=[32, 64, 32, 8]) for _ in range(population_size)]

    def evaluate(self, env: Game2048Env, games_per_player: int = 5) -> List[Tuple[SimpleNeuralNetwork, float]]:
        results = []
        for network in self.population:
            player = Player(network)
            total_score = 0
            for _ in range(games_per_player):
                game_result = player.play(env)
                total_score += game_result.score
            avg_score = total_score / games_per_player
            results.append((network, avg_score))
        return results

    def select_and_breed(self, evaluated: List[Tuple[SimpleNeuralNetwork, float]]) -> None:
        # Sort by score descending
        evaluated.sort(key=lambda x: x[1], reverse=True)
        elite = evaluated[:self.elite_size] 

        new_population = []
        new_population.extend([net for net, _ in elite])
        while len(new_population) < self.population_size:
            network = random.choice(elite)[0]
            new_population.append(network)  # Keep the best
            child = SimpleNeuralNetwork()
            child.weights = [np.copy(w) for w in network.weights]
            child.biases = [np.copy(b) for b in network.biases]
            child.mutate(self.mutation_rate, self.mutation_strength)
            new_population.append(child)

        # If population size is odd, add one more random network
        while len(new_population) < self.population_size:
            new_population.append(SimpleNeuralNetwork())

        self.population = new_population[:self.population_size]

    def run_generation(self, env: Game2048Env, games_per_player: int = 5) -> float:
        evaluated = self.evaluate(env, games_per_player)
        avg_score = sum(score for _, score in evaluated) / len(evaluated)
        self.select_and_breed(evaluated)
        return avg_score        

best_network = None
best_score = 0

def main():
    env = Game2048Env()
    optimizer = EvolutionaryOptimizer(population_size=100, elite_size=10, mutation_rate=0.1, mutation_strength=0.5)
    generations = 20
    games_per_player = 5

    avg_scores = []

    for gen in range(generations):
        avg_score = optimizer.run_generation(env, games_per_player)
        avg_scores.append(avg_score)
        print(f"Generation {gen+1}/{generations} - Average Score: {avg_score}")

    global best_network, best_score
    evaluated = optimizer.evaluate(env, games_per_player)
    gen_best_network, gen_best_score = max(evaluated, key=lambda x: x[1])
    if gen_best_score > best_score:
        best_score = gen_best_score
        best_network = gen_best_network    

    # Plot average scores over generations
    plt.plot(range(1, generations + 1), avg_scores)
    plt.xlabel('Generation')
    plt.ylabel('Average Score')
    plt.title('Evolution of Average Score over Generations')
    plt.show()

    env = Game2048Env()
    player = Player(best_network)

    while True:
        board = env.board
        print_board(board)
        action = player.next_move(board)
        print(f"Action: {action.name}")
        state, reward, done, _ = env.step(action)
        if done:
            print("Game Over")
            print_board(state)
            break

        time.sleep(1)  # Pause for a second to visualize    

if __name__ == "__main__":
    main()        

