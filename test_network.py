from manim import *
from typing import List
from dataclasses import dataclass
from enum import Enum

import torch

config.media_embed = True

import numpy as np


class Direction(Enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3


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
            # Stop if invalid move
            return self.board.copy(), reward, True, {}
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
                    reward += 10
                    merged.append(merged_val)
                    j += 2  # Skip next
                    moved = True
                else:
                    merged.append(tiles[j])
                    reward += 1
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


import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Determine the best available device
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# DEVICE = get_device()
DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")


class SimpleNeuralNetwork(nn.Module):
    """Simple feedforward neural network using PyTorch"""

    def __init__(
        self,
        input_size: int = 16,
        hidden_layers: List[int] = [256],
        output_size: int = 4,
        empty: bool = False,
    ):
        super().__init__()

        if empty:
            return

        # Build layers using PyTorch modules
        layers = []
        prev_size = input_size

        # Add hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            # layers.append(nn.Tanh())
            layers.append(nn.ReLU())
            prev_size = hidden_size

        # Add output layer (no activation)
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

        # Initialize weights using He initialization
        self._initialize_weights()

        # Move to device
        self.to(DEVICE)

    def _initialize_weights(self):
        """Initialize weights using He initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="tanh")
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """Forward pass through the network"""
        # Convert numpy array to tensor if needed and move to device
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(DEVICE)
        elif isinstance(x, torch.Tensor):
            x = x.to(DEVICE)

        return self.network(x)

    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.5):
        """Mutate the network's weights and biases"""
        with torch.no_grad():
            for param in self.parameters():
                if torch.rand(1).item() < mutation_rate:
                    mutation = torch.randn_like(param) * mutation_strength
                    param.add_(mutation)


import pickle


def save_network(network: SimpleNeuralNetwork, filename: str):
    torch.save(network.state_dict(), filename)


def load_network(filename: str, hidden_layers: List[int]) -> SimpleNeuralNetwork:
    network = SimpleNeuralNetwork(hidden_layers=hidden_layers)
    network.load_state_dict(torch.load(filename, map_location=DEVICE))
    network.to(DEVICE)
    return network


def save_population(population: List[SimpleNeuralNetwork], filename: str):
    with open(filename, "wb") as f:
        pickle.dump(population, f)


def load_population(filename: str) -> List[SimpleNeuralNetwork]:
    with open(filename, "rb") as f:
        population = pickle.load(f)
    return population


@dataclass
class GameResult:
    score: int
    max_tile: int
    moves: int


class Player:
    def __init__(self, network: SimpleNeuralNetwork):
        self.network = network

    def play(self, env: Game2048Env, max_steps: int = 100) -> GameResult:
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
        self.network.eval()  # Set to evaluation mode
        with torch.no_grad():
            flat_state = state.flatten() / 2048.0  # Normalize input
            q_values = self.network.forward(flat_state)
            # Move back to CPU for numpy conversion
            q_values_cpu = q_values.cpu()
            action = Direction(q_values_cpu.numpy().argmax())
            return action


import pickle


def save_network(network: SimpleNeuralNetwork, filename: str):
    torch.save(network.state_dict(), filename)


def load_network(filename: str, hidden_layers: List[int]) -> SimpleNeuralNetwork:
    network = SimpleNeuralNetwork(hidden_layers=hidden_layers)
    network.load_state_dict(torch.load(filename, map_location=DEVICE))
    network.to(DEVICE)
    return network


def save_population(population: List[SimpleNeuralNetwork], filename: str):
    with open(filename, "wb") as f:
        pickle.dump(population, f)


def load_population(filename: str) -> List[SimpleNeuralNetwork]:
    with open(filename, "rb") as f:
        population = pickle.load(f)
    return population


def find_latest_network(folder: str = "./networks/") -> str:
    import os

    files = [
        f
        for f in os.listdir(folder)
        if f.startswith("population_gen_") and f.endswith(".pkl")
    ]
    if not files:
        return None
    latest_file = max(files, key=lambda x: int(x.split("_")[2].split(".")[0]))
    return os.path.join(folder, latest_file)


hidden_layers = [1024, 1024, 1024]
layers_str = "_".join(map(str, hidden_layers))
latest_checkpoint = find_latest_network(f"./networks/{layers_str}")
# latest_checkpoint = './networks/2048_2048/population_gen_1.pkl'
print(latest_checkpoint)
population = load_population(latest_checkpoint)

best_block = 0
best_score = 0
best_network = None

games_per_player = 10

for i, net in enumerate(population):
    player = Player(net.to(DEVICE))

    for _ in range(games_per_player):
        env = Game2048Env()
        while True:
            board = env.board
            # print_board(board)
            action = player.next_move(board)
            # prev_state = state.copy()
            state, reward, done, _ = env.step(action)
            # print(f"Action: {action.name}, Best: {np.max(state)}")
            # if (prev_state == state).all():
            #     # print("nop")
            #     # print_board(prev_state)
            #     # print_board(state)
            #     break
            if done:
                # print("Game Over")
                # print_board(state)
                break

        best_block_this_game = np.max(state)
        if best_block_this_game > best_block:
            best_network = net
            best_block = best_block_this_game
            print(f"New Best Network Found! Tile: {best_block} | Score: {env.score}")

        best_score = max(best_score, env.score)
        # print(f"Player {i+1}/{len(population)} - Best Tile: {best_block} | Score: {env.score} | Max Score: {best_score}")
        # print("===================================")

player = Player(best_network)

del population
