import numpy as np
from enum import Enum
from typing import List


class Direction(Enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3


class Game2048Env:
    def __init__(self, random_seed: int = None):
        self.grid_size = 4
        if random_seed is not None:
            self.rng = np.random.default_rng(random_seed)
        else:
            self.rng = np.random.default_rng()
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
            x, y = empty[self.rng.integers(low=0, high=len(empty), size=1)[0]]
            self.board[x, y] = 2 if self.rng.random() < 0.9 else 4

    def step(self, actions: List[Direction]):
        direction = None
        # Limit to best action
        actions = actions[:1]
        for action in actions:
            moved, reward = self.move(action.value)
            direction = action
            if moved:
                self.spawn_tile()
                done = not self.can_move()
                self.score += reward
                break
            else:
                # Stop if invalid move
                done = True
                break
        
        return self.board.copy(), reward, done, {"direction": direction}

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
                    reward += 0
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
