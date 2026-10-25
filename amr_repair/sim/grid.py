"""Grid and map representation for MovingAI benchmark maps."""

from typing import Tuple, List, Set
import numpy as np
import networkx as nx


class GridMap:
    def __init__(self, blocked: np.ndarray, width: int, height: int):
        self.blocked: np.ndarray = blocked  # True = blocked (obstacle), False = free
        self.width: int = width
        self.height: int = height
        self.graph: nx.Graph = self._build_graph()

    def _build_graph(self) -> nx.Graph:
        g = nx.Graph()
        for y in range(self.height):
            for x in range(self.width):
                if not self.blocked[y, x]:
                    g.add_node((x, y))
                    # 4-connected neighbors
                    for dx, dy in [(1, 0), (0, 1)]:
                        nx_, ny_ = x + dx, y + dy
                        if nx_ < self.width and ny_ < self.height and not self.blocked[ny_, nx_]:
                            g.add_edge((x, y), (nx_, ny_))
        return g

    def is_free(self, cell: Tuple[int, int]) -> bool:
        x, y = cell
        if 0 <= x < self.width and 0 <= y < self.height:
            return not bool(self.blocked[y, x])
        return False

    def get_free_cells(self) -> List[Tuple[int, int]]:
        free_indices = np.argwhere(~self.blocked)
        # argwhere returns [y, x]
        return [(int(x), int(y)) for y, x in free_indices]

    def get_obstacle_cells(self) -> List[Tuple[int, int]]:
        obs_indices = np.argwhere(self.blocked)
        return [(int(x), int(y)) for y, x in obs_indices]

    @classmethod
    def from_movingai(cls, file_path: str) -> "GridMap":
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.rstrip() for line in f if line.strip()]

        height = int(lines[1].split()[1])
        width = int(lines[2].split()[1])
        grid_lines = lines[4:4 + height]

        blocked = np.zeros((height, width), dtype=bool)
        for y, row in enumerate(grid_lines):
            for x, char in enumerate(row):
                if char in ("@", "T", "O"):
                    blocked[y, x] = True

        return cls(blocked, width, height)


def load_map(file_path: str) -> GridMap:
    """Helper function to load a MovingAI .map file."""
    return GridMap.from_movingai(file_path)

