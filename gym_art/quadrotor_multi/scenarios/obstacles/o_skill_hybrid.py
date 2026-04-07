import copy
from collections import deque
import numpy as np

from gym_art.quadrotor_multi.obstacles.utils import get_cell_centers
from gym_art.quadrotor_multi.scenarios.obstacles.o_base import Scenario_o_base


class Scenario_o_skill_hybrid(Scenario_o_base):
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)
        self.approch_goal_metric = 1.0
        self.center_lane_x = None

    def _cell_index(self, x_idx, y_idx, x_cells):
        return x_idx * self._y_cells + y_idx

    def _center_from_cell(self, cell_centers, x_idx, y_idx, x_cells, z_value):
        pos_x, pos_y = cell_centers[self._cell_index(x_idx, y_idx, x_cells)]
        return np.array([pos_x, pos_y, z_value])

    def _sample_center_lane(self, x_cells):
        center = x_cells // 2
        candidates = [center]
        if center - 1 >= 1:
            candidates.append(center - 1)
        if center + 1 <= x_cells - 2:
            candidates.append(center + 1)
        return int(np.random.choice(candidates))

    def _sample_gate_openings(self, x_cells, opening_width, gate_count):
        max_open_start = max(0, x_cells - opening_width)
        if max_open_start == 0:
            return [0 for _ in range(gate_count)]

        margin = 1 if x_cells >= 6 else 0
        left_start = min(max_open_start, margin)
        left_end = max(left_start, (x_cells // 2) - opening_width)
        right_start = min(max_open_start, x_cells // 2)
        right_end = max(right_start, max_open_start - margin)

        left_candidates = list(range(left_start, left_end + 1))
        right_candidates = list(range(right_start, right_end + 1))

        if not left_candidates:
            left_candidates = [0]
        if not right_candidates:
            right_candidates = [max_open_start]

        start_left = bool(np.random.randint(0, 2))
        openings = []
        for gate_idx in range(gate_count):
            use_left = start_left if gate_idx % 2 == 0 else not start_left
            candidates = left_candidates if use_left else right_candidates
            openings.append(int(np.random.choice(candidates)))

        return openings

    def _grid_row(self, y_cells, travel_row):
        return y_cells - 1 - travel_row

    def _segment_layout(self, y_cells):
        nav_end = max(4, y_cells // 3)
        gate_spacing = 2

        gate_first = nav_end + 3
        max_last_gate = y_cells - 4
        desired_gate_count = 3 if y_cells >= 12 else 2
        max_gate_count = max(1, 1 + max(0, max_last_gate - gate_first) // gate_spacing)
        gate_count = min(desired_gate_count, max_gate_count)
        gate_rows = [gate_first + gate_idx * gate_spacing for gate_idx in range(gate_count)]

        recovery_start = min(y_cells - 2, gate_rows[-1] + 2)
        return nav_end, gate_rows, recovery_start

    def _add_navigation_obstacles(self, obst_map, nav_end, x_cells, y_cells):
        if nav_end < 1:
            return

        candidate_rows = sorted(
            set([1, max(2, nav_end // 3 + 1), max(2, (2 * nav_end) // 3), max(2, nav_end - 1)])
        )
        pattern_offsets = [-1, 1, 0, -2]
        for idx, travel_row in enumerate(candidate_rows):
            grid_row = self._grid_row(y_cells, travel_row)
            obstacle_col = int(np.clip(self.center_lane_x + pattern_offsets[idx % len(pattern_offsets)], 1, x_cells - 2))
            obst_map[obstacle_col, grid_row] = 1

    def _has_path(self, obst_map, start_x, goal_x):
        x_cells, y_cells = obst_map.shape
        start = (start_x, y_cells - 1)
        goal = (goal_x, 0)
        if obst_map[start] == 1 or obst_map[goal] == 1:
            return False

        q = deque([start])
        visited = {start}
        while q:
            x, y = q.popleft()
            if (x, y) == goal:
                return True

            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < x_cells and 0 <= ny < y_cells and obst_map[nx, ny] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append((nx, ny))

        return False

    def generate_obstacles(self, obst_spawn_area):
        x_cells = int(obst_spawn_area[0])
        y_cells = int(obst_spawn_area[1])
        self._y_cells = y_cells
        cell_centers = get_cell_centers(obst_area_length=x_cells, obst_area_width=y_cells, grid_size=1.0)
        opening_width = 2 if x_cells >= 6 else 1
        nav_end, gate_travel_rows, recovery_start = self._segment_layout(y_cells)

        max_attempts = 64
        obst_map = None
        for _ in range(max_attempts):
            self.center_lane_x = self._sample_center_lane(x_cells)
            obst_map = np.zeros((x_cells, y_cells))
            openings = self._sample_gate_openings(
                x_cells=x_cells, opening_width=opening_width, gate_count=len(gate_travel_rows)
            )

            # Segment 1: sparse navigation obstacles only.
            self._add_navigation_obstacles(obst_map=obst_map, nav_end=nav_end, x_cells=x_cells, y_cells=y_cells)

            # Segment 2: only offset gates, no extra clutter mixed in.
            for gate_travel_row, opening_start in zip(gate_travel_rows, openings):
                gate_row = self._grid_row(y_cells, gate_travel_row)
                obst_map[:, gate_row] = 1
                obst_map[opening_start:opening_start + opening_width, gate_row] = 0

            # Segment 3: recovery zone stays open on purpose.
            for travel_row in range(recovery_start, y_cells - 1):
                obst_map[:, self._grid_row(y_cells, travel_row)] = np.minimum(
                    obst_map[:, self._grid_row(y_cells, travel_row)], 0
                )

            if self._has_path(obst_map=obst_map, start_x=self.center_lane_x, goal_x=self.center_lane_x):
                break
        else:
            raise RuntimeError("Failed to generate a traversable o_skill_hybrid map")

        obst_pos_arr = []
        for x_idx in range(x_cells):
            for y_idx in range(y_cells):
                if obst_map[x_idx, y_idx] != 1:
                    continue
                pos_x, pos_y = cell_centers[self._cell_index(x_idx, y_idx, x_cells)]
                obst_pos_arr.append([pos_x, pos_y, self.room_dims[2] / 2.0])

        return obst_map, obst_pos_arr, cell_centers

    def step(self):
        return

    def reset(self, obst_map=None, cell_centers=None):
        self.obstacle_map = obst_map
        self.cell_centers = cell_centers
        if obst_map is None or cell_centers is None:
            raise NotImplementedError

        x_cells, y_cells = self.obstacle_map.shape
        z_value = min(2.0, self.room_dims[2] - 1.0)
        center_lane_x = self.center_lane_x if self.center_lane_x is not None else self._sample_center_lane(x_cells)

        self.start_point = self._center_from_cell(
            cell_centers=self.cell_centers, x_idx=center_lane_x, y_idx=y_cells - 1, x_cells=x_cells, z_value=z_value
        )
        self.end_point = self._center_from_cell(
            cell_centers=self.cell_centers, x_idx=center_lane_x, y_idx=0, x_cells=x_cells, z_value=z_value
        )

        self.update_formation_and_relate_param()
        self.spawn_points = np.array([copy.deepcopy(self.start_point) for _ in range(self.num_agents)])
        self.goals = np.array([copy.deepcopy(self.end_point) for _ in range(self.num_agents)])
