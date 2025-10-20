from dataclasses import dataclass
from SimpleNeuralNetwork import SimpleNeuralNetwork
from Game import Game2048Env, Direction
import torch
import numpy as np
from typing import List


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
            actions = self.next_move(state)

            state, reward, done, _ = env.step(actions)
            total_reward += reward
            steps += 1

        return GameResult(score=total_reward, max_tile=np.max(state), moves=steps)

    def next_move(self, state: np.ndarray) -> List[Direction]:
        self.network.eval()  # Set to evaluation mode
        with torch.no_grad():
            flat_state = state.flatten() / 2048.0  # Normalize input
            q_values = self.network.forward(flat_state)
            # Move back to CPU for numpy conversion
            q_values_cpu = q_values.cpu()
            # Sort actions by Q-values in descending order
            actions = [Direction(i) for i in np.argsort(-q_values_cpu.numpy())]
            return actions
