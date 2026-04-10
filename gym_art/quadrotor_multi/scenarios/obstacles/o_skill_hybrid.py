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
        self.start_lane_x = None
        self.goal_lane_x = None
        self.gate_openings = None
        self.gate_travel_rows = None
        self.guidance_type = "none"
        self.guidance_distance_field = None
        self.free_cell_centers = None
        self.free_cell_guidance_distances = None
        self.free_cells = None
        self.local_guidance_window_cells = 3
        self._x_cells = None
        self._y_cells = None

    def set_guidance_type(self, guidance_type):
        self.guidance_type = guidance_type

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

    def _lane_candidates(self, x_cells):
        margin = 1 if x_cells >= 6 else 0
        start = margin
        end = x_cells - 1 - margin
        if end < start:
            return [x_cells // 2]
        return list(range(start, end + 1))

    def _sample_start_goal_lanes(self, x_cells):
        candidates = self._lane_candidates(x_cells)
        start_lane = int(np.random.choice(candidates))
        goal_lane = int(np.random.choice(candidates))
        if len(candidates) > 1:
            min_separation = 1
            for _ in range(16):
                if abs(goal_lane - start_lane) >= min_separation:
                    break
                goal_lane = int(np.random.choice(candidates))
        return start_lane, goal_lane

    def _opening_center(self, opening_start, opening_width):
        return float(opening_start) + 0.5 * float(opening_width - 1)

    def _sample_gate_openings(self, x_cells, opening_width, gate_count, start_lane_x, goal_lane_x):
        max_open_start = max(0, x_cells - opening_width)
        if max_open_start == 0:
            return [0 for _ in range(gate_count)]

        margin = 1 if x_cells >= 6 else 0
        candidate_min = margin
        candidate_max = max_open_start - margin
        if candidate_max < candidate_min:
            candidates = list(range(max_open_start + 1))
        else:
            candidates = list(range(candidate_min, candidate_max + 1))

        if not candidates:
            candidates = [0]

        start_center = float(start_lane_x)
        goal_center = float(goal_lane_x)
        zigzag_amp = 1.0 if x_cells >= 8 else 0.0
        zigzag_dir = float(np.random.choice([-1.0, 1.0])) if gate_count > 1 else 0.0
        openings = []
        prev_center = start_center
        for gate_idx in range(gate_count):
            interp = float(gate_idx + 1) / float(gate_count + 1)
            anchor = (1.0 - interp) * start_center + interp * goal_center
            if zigzag_amp > 0.0:
                anchor += zigzag_dir * zigzag_amp * (1.0 if gate_idx % 2 == 0 else -1.0)
            anchor += float(np.random.randint(-1, 2))

            ranked_candidates = sorted(
                candidates,
                key=lambda start: (
                    abs(self._opening_center(start, opening_width) - anchor),
                    np.random.random(),
                ),
            )
            chosen = None
            min_delta = 1.5 if x_cells >= 8 else 0.5
            for opening_start in ranked_candidates:
                opening_center = self._opening_center(opening_start, opening_width)
                if gate_idx > 0 and len(candidates) > 1 and abs(opening_center - prev_center) < min_delta:
                    continue
                chosen = int(opening_start)
                prev_center = opening_center
                break

            if chosen is None:
                chosen = int(np.random.choice(candidates))
                prev_center = self._opening_center(chosen, opening_width)

            openings.append(chosen)

        return openings

    def _gate_route_has_diversity(self, x_cells, opening_width, start_lane_x, goal_lane_x, openings):
        if not openings:
            return False

        mid_x = 0.5 * float(x_cells - 1)
        opening_centers = [self._opening_center(opening_start, opening_width) for opening_start in openings]
        opening_sides = [np.sign(center - mid_x) for center in opening_centers]
        nonzero_sides = [side for side in opening_sides if side != 0]
        uses_both_sides = len(set(nonzero_sides)) >= 2

        route_nodes = [float(start_lane_x)] + opening_centers + [float(goal_lane_x)]
        route_deltas = np.diff(route_nodes)
        positive_moves = np.any(route_deltas > 0.5)
        negative_moves = np.any(route_deltas < -0.5)
        has_direction_change = bool(positive_moves and negative_moves)

        horizontal_span = float(max(route_nodes) - min(route_nodes))
        min_span = 3.0 if x_cells >= 10 else 2.0
        has_large_cross_track = horizontal_span >= min_span
        return uses_both_sides and has_direction_change and has_large_cross_track

    def _grid_row(self, y_cells, travel_row):
        return y_cells - 1 - travel_row

    def _segment_layout(self, y_cells):
        nav_end = int(np.random.randint(max(4, y_cells // 4), max(5, y_cells // 3) + 1))
        gate_spacing = int(np.random.choice([2, 3]))
        gate_first = nav_end + int(np.random.choice([2, 3, 4]))
        max_last_gate = y_cells - 4
        desired_gate_count = 3 if y_cells >= 12 else 2
        max_gate_count = max(1, 1 + max(0, max_last_gate - gate_first) // gate_spacing)
        gate_count = min(desired_gate_count, max_gate_count)
        gate_rows = [gate_first + gate_idx * gate_spacing for gate_idx in range(gate_count)]

        recovery_start = min(y_cells - 2, gate_rows[-1] + 2)
        return nav_end, gate_rows, recovery_start

    def _protected_route_cells(self, x_cells, y_cells, start_lane_x, goal_lane_x, gate_travel_rows, openings,
                               opening_width):
        route_nodes = [(float(start_lane_x), float(y_cells - 1))]
        for gate_travel_row, opening_start in zip(gate_travel_rows, openings):
            route_nodes.append((self._opening_center(opening_start, opening_width), float(self._grid_row(y_cells, gate_travel_row))))
        route_nodes.append((float(goal_lane_x), 0.0))

        protected = set()
        for (x0, y0), (x1, y1) in zip(route_nodes[:-1], route_nodes[1:]):
            y_start = int(min(y0, y1))
            y_end = int(max(y0, y1))
            if y_end == y_start:
                y_values = [y_start]
            else:
                y_values = range(y_start, y_end + 1)

            for y_idx in y_values:
                if abs(y1 - y0) < 1e-6:
                    interp = 0.0
                else:
                    interp = float(y_idx - y0) / float(y1 - y0)
                x_interp = x0 + interp * (x1 - x0)
                center_col = int(round(x_interp))
                for offset in (-1, 0, 1):
                    x_idx = int(np.clip(center_col + offset, 0, x_cells - 1))
                    protected.add((x_idx, y_idx))

        for gate_travel_row, opening_start in zip(gate_travel_rows, openings):
            gate_row = self._grid_row(y_cells, gate_travel_row)
            for x_idx in range(opening_start, opening_start + opening_width):
                protected.add((int(x_idx), int(gate_row)))

        protected.add((int(start_lane_x), y_cells - 1))
        protected.add((int(goal_lane_x), 0))
        return protected

    def _add_random_cylinder_obstacles(self, obst_map, x_cells, y_cells, gate_travel_rows, protected_cells):
        gate_grid_rows = {self._grid_row(y_cells, gate_travel_row) for gate_travel_row in gate_travel_rows}
        candidate_cells = []
        for x_idx in range(0, x_cells):
            for y_idx in range(1, y_cells - 1):
                if y_idx in gate_grid_rows:
                    continue
                if (x_idx, y_idx) in protected_cells:
                    continue
                candidate_cells.append((x_idx, y_idx))

        if not candidate_cells:
            return

        base_count = max(6, int(0.09 * x_cells * y_cells))
        max_count = min(len(candidate_cells), max(base_count, int(0.16 * x_cells * y_cells)))
        num_obstacles = int(np.random.randint(base_count, max_count + 1)) if max_count > base_count else max_count
        if num_obstacles <= 0:
            return

        sampled_indices = np.random.choice(len(candidate_cells), size=num_obstacles, replace=False)
        for idx in sampled_indices:
            x_idx, y_idx = candidate_cells[int(idx)]
            obst_map[x_idx, y_idx] = 1

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

    def _compute_guidance_distance_field(self, obst_map, goal_x_idx, goal_y_idx):
        x_cells, y_cells = obst_map.shape
        guidance_distance_field = np.full((x_cells, y_cells), np.inf, dtype=np.float32)
        if obst_map[goal_x_idx, goal_y_idx] != 0:
            raise RuntimeError("Goal cell must remain traversable for guidance reward")

        q = deque([(goal_x_idx, goal_y_idx)])
        guidance_distance_field[goal_x_idx, goal_y_idx] = 0.0

        while q:
            x_idx, y_idx = q.popleft()
            base_distance = guidance_distance_field[x_idx, y_idx] + 1.0
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x_idx + dx, y_idx + dy
                if 0 <= nx < x_cells and 0 <= ny < y_cells and obst_map[nx, ny] == 0:
                    if base_distance < guidance_distance_field[nx, ny]:
                        guidance_distance_field[nx, ny] = base_distance
                        q.append((nx, ny))

        return guidance_distance_field

    def _position_to_grid_cell(self, pos):
        if self._x_cells is None or self._y_cells is None:
            raise RuntimeError("Grid dimensions must be initialized before querying guidance distance")

        x_idx = int(np.clip(np.round(pos[0] + self._x_cells / 2.0 - 0.5), 0, self._x_cells - 1))
        y_idx = int(np.clip(np.round(self._y_cells / 2.0 - 0.5 - pos[1]), 0, self._y_cells - 1))
        return x_idx, y_idx

    def _cell_center_xy(self, x_idx, y_idx):
        return self.cell_centers[self._cell_index(int(x_idx), int(y_idx), self._x_cells)]

    def _nearest_valid_guidance_cell(self, pos):
        x_idx, y_idx = self._position_to_grid_cell(pos)
        if self.obstacle_map[x_idx, y_idx] == 0 and np.isfinite(self.guidance_distance_field[x_idx, y_idx]):
            return int(x_idx), int(y_idx)

        best_cell = None
        best_distance = np.inf
        max_radius = max(self._x_cells, self._y_cells)
        for radius in range(1, max_radius + 1):
            x_min = max(0, x_idx - radius)
            x_max = min(self._x_cells - 1, x_idx + radius)
            y_min = max(0, y_idx - radius)
            y_max = min(self._y_cells - 1, y_idx + radius)

            for cand_x in range(x_min, x_max + 1):
                for cand_y in range(y_min, y_max + 1):
                    if self.obstacle_map[cand_x, cand_y] != 0:
                        continue
                    if not np.isfinite(self.guidance_distance_field[cand_x, cand_y]):
                        continue
                    center_xy = self._cell_center_xy(cand_x, cand_y)
                    distance = float(np.linalg.norm(center_xy - pos[:2]))
                    if distance < best_distance:
                        best_distance = distance
                        best_cell = (int(cand_x), int(cand_y))

            if best_cell is not None:
                return best_cell

        raise RuntimeError("Failed to locate a valid guidance anchor cell")

    def _continuous_guidance_distance(self, pos, distance_field, x_min=0, y_min=0, candidate_radius=1):
        anchor_x, anchor_y = self._nearest_valid_guidance_cell(pos)
        guidance_xy_distance = np.inf

        for dx in range(-candidate_radius, candidate_radius + 1):
            for dy in range(-candidate_radius, candidate_radius + 1):
                cand_x = anchor_x + dx
                cand_y = anchor_y + dy
                if not (0 <= cand_x < self._x_cells and 0 <= cand_y < self._y_cells):
                    continue
                if self.obstacle_map[cand_x, cand_y] != 0:
                    continue

                local_x = cand_x - x_min
                local_y = cand_y - y_min
                if not (0 <= local_x < distance_field.shape[0] and 0 <= local_y < distance_field.shape[1]):
                    continue

                bfs_distance = distance_field[local_x, local_y]
                if not np.isfinite(bfs_distance):
                    continue

                center_xy = self._cell_center_xy(cand_x, cand_y)
                candidate_distance = float(bfs_distance + np.linalg.norm(center_xy - pos[:2]))
                if candidate_distance < guidance_xy_distance:
                    guidance_xy_distance = candidate_distance

        if not np.isfinite(guidance_xy_distance):
            center_xy = self._cell_center_xy(anchor_x, anchor_y)
            guidance_xy_distance = float(
                self.guidance_distance_field[anchor_x, anchor_y] + np.linalg.norm(center_xy - pos[:2])
            )

        guidance_z_distance = float(abs(self.end_point[2] - pos[2]))
        return guidance_xy_distance + guidance_z_distance

    def _compute_local_guidance_distance(self, pos):
        if self.free_cell_centers is None or self.free_cells is None or len(self.free_cells) == 0:
            return float(np.linalg.norm(self.end_point[:2] - pos[:2]) + abs(self.end_point[2] - pos[2]))

        curr_x_idx, curr_y_idx = self._nearest_valid_guidance_cell(pos)

        x_cells, y_cells = self.obstacle_map.shape
        radius = self.local_guidance_window_cells
        x_min = max(0, int(curr_x_idx) - radius)
        x_max = min(x_cells - 1, int(curr_x_idx) + radius)
        y_min = max(0, int(curr_y_idx) - radius)
        y_max = min(y_cells - 1, int(curr_y_idx) + radius)

        goal_dir = self.end_point[:2] - pos[:2]
        goal_norm = np.linalg.norm(goal_dir)
        if goal_norm < 1e-6:
            return float(abs(self.end_point[2] - pos[2]))
        goal_dir = goal_dir / goal_norm

        candidate_indices = []
        candidate_scores = []
        for idx, (x_idx, y_idx) in enumerate(self.free_cells):
            if not (x_min <= x_idx <= x_max and y_min <= y_idx <= y_max):
                continue
            disp = self.free_cell_centers[idx] - pos[:2]
            progress = float(np.dot(disp, goal_dir))
            if progress <= 0.0:
                continue
            boundary_bonus = 1.0 if (
                x_idx == x_min or x_idx == x_max or y_idx == y_min or y_idx == y_max
            ) else 0.0
            candidate_indices.append(idx)
            candidate_scores.append(progress + 0.25 * boundary_bonus)

        if not candidate_indices:
            return self.get_global_guidance_distance(pos)

        target_idx = candidate_indices[int(np.argmax(candidate_scores))]
        target_x_idx, target_y_idx = self.free_cells[target_idx]

        local_obst_map = self.obstacle_map[x_min:x_max + 1, y_min:y_max + 1]
        local_field = self._compute_guidance_distance_field(
            obst_map=local_obst_map,
            goal_x_idx=int(target_x_idx) - x_min,
            goal_y_idx=int(target_y_idx) - y_min,
        )

        return self._continuous_guidance_distance(
            pos=pos, distance_field=local_field, x_min=x_min, y_min=y_min, candidate_radius=1
        )

    def _build_gate_box_items(self, obst_map, cell_centers, x_cells, gate_travel_rows, grid_size=1.0):
        gate_items = []
        for gate_travel_row in gate_travel_rows:
            gate_row = self._grid_row(y_cells=self._y_cells, travel_row=gate_travel_row)
            occupied_cols = np.where(obst_map[:, gate_row] == 1)[0]
            if len(occupied_cols) == 0:
                continue

            run_start = occupied_cols[0]
            run_end = occupied_cols[0]
            for col_idx in occupied_cols[1:]:
                if col_idx == run_end + 1:
                    run_end = col_idx
                    continue

                gate_items.append(self._make_box_item(cell_centers, x_cells, gate_row, run_start, run_end, grid_size))
                run_start = col_idx
                run_end = col_idx

            gate_items.append(self._make_box_item(cell_centers, x_cells, gate_row, run_start, run_end, grid_size))

        return gate_items

    def _make_box_item(self, cell_centers, x_cells, gate_row, run_start, run_end, grid_size):
        left_center = self._center_from_cell(
            cell_centers=cell_centers,
            x_idx=run_start,
            y_idx=gate_row,
            x_cells=x_cells,
            z_value=self.room_dims[2] / 2.0,
        )
        right_center = self._center_from_cell(
            cell_centers=cell_centers,
            x_idx=run_end,
            y_idx=gate_row,
            x_cells=x_cells,
            z_value=self.room_dims[2] / 2.0,
        )
        center = np.array(
            [
                0.5 * (left_center[0] + right_center[0]),
                left_center[1],
                self.room_dims[2] / 2.0,
            ],
            dtype=np.float32,
        )
        size = np.array(
            [
                (run_end - run_start + 1) * grid_size,
                grid_size,
                self.room_dims[2],
            ],
            dtype=np.float32,
        )
        return {"type": "box", "center": center, "size": size}

    def generate_obstacles(self, obst_spawn_area):
        x_cells = int(obst_spawn_area[0])
        y_cells = int(obst_spawn_area[1])
        self._x_cells = x_cells
        self._y_cells = y_cells
        cell_centers = get_cell_centers(obst_area_length=x_cells, obst_area_width=y_cells, grid_size=1.0)
        opening_width = 2 if x_cells >= 6 else 1
        max_attempts = 64
        obst_map = None
        for _ in range(max_attempts):
            self.start_lane_x, self.goal_lane_x = self._sample_start_goal_lanes(x_cells)
            self.center_lane_x = self.start_lane_x
            nav_end, gate_travel_rows, recovery_start = self._segment_layout(y_cells)
            obst_map = np.zeros((x_cells, y_cells))
            openings = self._sample_gate_openings(
                x_cells=x_cells,
                opening_width=opening_width,
                gate_count=len(gate_travel_rows),
                start_lane_x=self.start_lane_x,
                goal_lane_x=self.goal_lane_x,
            )

            protected_cells = self._protected_route_cells(
                x_cells=x_cells,
                y_cells=y_cells,
                start_lane_x=self.start_lane_x,
                goal_lane_x=self.goal_lane_x,
                gate_travel_rows=gate_travel_rows,
                openings=openings,
                opening_width=opening_width,
            )

            self._add_random_cylinder_obstacles(
                obst_map=obst_map,
                x_cells=x_cells,
                y_cells=y_cells,
                gate_travel_rows=gate_travel_rows,
                protected_cells=protected_cells,
            )

            # Offset gates stay continuous; cylinders are sampled over the remaining rows.
            for gate_travel_row, opening_start in zip(gate_travel_rows, openings):
                gate_row = self._grid_row(y_cells, gate_travel_row)
                obst_map[:, gate_row] = 1
                obst_map[opening_start:opening_start + opening_width, gate_row] = 0

            if self._has_path(obst_map=obst_map, start_x=self.start_lane_x, goal_x=self.goal_lane_x) and \
                    self._gate_route_has_diversity(
                        x_cells=x_cells,
                        opening_width=opening_width,
                        start_lane_x=self.start_lane_x,
                        goal_lane_x=self.goal_lane_x,
                        openings=openings,
                    ):
                self.gate_openings = openings
                self.gate_travel_rows = gate_travel_rows
                break
        else:
            raise RuntimeError("Failed to generate a traversable o_skill_hybrid map")

        gate_grid_rows = {self._grid_row(y_cells, gate_travel_row) for gate_travel_row in gate_travel_rows}
        obstacle_items = []
        for x_idx in range(x_cells):
            for y_idx in range(y_cells):
                if obst_map[x_idx, y_idx] != 1:
                    continue
                if y_idx in gate_grid_rows:
                    continue
                pos_x, pos_y = cell_centers[self._cell_index(x_idx, y_idx, x_cells)]
                obstacle_items.append([pos_x, pos_y, self.room_dims[2] / 2.0])

        obstacle_items.extend(
            self._build_gate_box_items(
                obst_map=obst_map,
                cell_centers=cell_centers,
                x_cells=x_cells,
                gate_travel_rows=gate_travel_rows,
                grid_size=1.0,
            )
        )

        return obst_map, obstacle_items, cell_centers

    def step(self):
        return

    def get_global_guidance_distance(self, pos):
        if self.free_cell_centers is None or self.free_cell_guidance_distances is None:
            return float(np.linalg.norm(self.end_point[:2] - pos[:2]) + abs(self.end_point[2] - pos[2]))
        return self._continuous_guidance_distance(pos=pos, distance_field=self.guidance_distance_field, candidate_radius=1)

    def get_guidance_distance(self, pos):
        if self.guidance_type == "global_bfs":
            return self.get_global_guidance_distance(pos)
        if self.guidance_type == "local_bfs":
            return self._compute_local_guidance_distance(pos)
        return float(np.linalg.norm(self.end_point - pos))

    def reset(self, obst_map=None, cell_centers=None):
        self.obstacle_map = obst_map
        self.cell_centers = cell_centers
        if obst_map is None or cell_centers is None:
            raise NotImplementedError

        x_cells, y_cells = self.obstacle_map.shape
        self._x_cells = x_cells
        self._y_cells = y_cells
        z_value = min(2.0, self.room_dims[2] - 1.0)
        start_lane_x = self.start_lane_x if self.start_lane_x is not None else self._sample_center_lane(x_cells)
        goal_lane_x = self.goal_lane_x if self.goal_lane_x is not None else start_lane_x

        self.start_point = self._center_from_cell(
            cell_centers=self.cell_centers, x_idx=start_lane_x, y_idx=y_cells - 1, x_cells=x_cells, z_value=z_value
        )
        self.end_point = self._center_from_cell(
            cell_centers=self.cell_centers, x_idx=goal_lane_x, y_idx=0, x_cells=x_cells, z_value=z_value
        )
        free_cells = np.argwhere(self.obstacle_map == 0)
        self.free_cells = free_cells.astype(np.int32)
        self.free_cell_centers = np.zeros((len(free_cells), 2), dtype=np.float32)
        self.free_cell_guidance_distances = np.zeros(len(free_cells), dtype=np.float32)
        self.guidance_distance_field = self._compute_guidance_distance_field(
            obst_map=self.obstacle_map, goal_x_idx=goal_lane_x, goal_y_idx=0
        )
        for idx, (x_idx, y_idx) in enumerate(free_cells):
            cell_center = self._center_from_cell(
                cell_centers=self.cell_centers, x_idx=int(x_idx), y_idx=int(y_idx), x_cells=x_cells, z_value=z_value
            )
            self.free_cell_centers[idx] = cell_center[:2]
            self.free_cell_guidance_distances[idx] = self.guidance_distance_field[int(x_idx), int(y_idx)]

        self.update_formation_and_relate_param()
        self.spawn_points = np.array([copy.deepcopy(self.start_point) for _ in range(self.num_agents)])
        self.goals = np.array([copy.deepcopy(self.end_point) for _ in range(self.num_agents)])
