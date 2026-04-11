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
        self.local_target_hold_steps = 6
        self.local_target_reached_cells = 1
        self.local_targets = None
        self.local_target_ages = None
        self.local_target_fields = None
        self.local_target_windows = None
        self.local_last_anchor_cells = None
        self.local_last_positions = None
        self.local_last_guidance_values = None
        self._x_cells = None
        self._y_cells = None
        self.nearest_valid_guidance_cells = None
        self.agent_goal_fields = None

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

    def _build_multi_agent_spawn_points(self, x_cells, y_cells, z_value):
        if self.num_agents == 1:
            return np.array([copy.deepcopy(self.start_point)], dtype=np.float32)

        anchor = np.array([int(self.start_lane_x), y_cells - 1], dtype=np.int32)
        free_cells = np.argwhere(self.obstacle_map == 0)
        if len(free_cells) < self.num_agents:
            raise RuntimeError("Not enough free cells to place all agents")

        def _candidate_key(cell):
            x_idx, y_idx = int(cell[0]), int(cell[1])
            manhattan = abs(x_idx - anchor[0]) + abs(y_idx - anchor[1])
            lane_bias = abs(x_idx - int(self.start_lane_x))
            depth_bias = y_cells - 1 - y_idx
            return manhattan, lane_bias, depth_bias, np.random.random()

        ranked_cells = sorted(free_cells.tolist(), key=_candidate_key)
        selected_cells = []
        for cell in ranked_cells:
            x_idx, y_idx = int(cell[0]), int(cell[1])
            keep = True
            for sel_x, sel_y in selected_cells:
                if abs(sel_x - x_idx) + abs(sel_y - y_idx) < 2:
                    keep = False
                    break
            if keep:
                selected_cells.append((x_idx, y_idx))
            if len(selected_cells) >= self.num_agents:
                break

        if len(selected_cells) < self.num_agents:
            selected_cells = [(int(cell[0]), int(cell[1])) for cell in ranked_cells[:self.num_agents]]

        spawn_points = []
        for x_idx, y_idx in selected_cells:
            spawn_points.append(
                self._center_from_cell(
                    cell_centers=self.cell_centers,
                    x_idx=x_idx,
                    y_idx=y_idx,
                    x_cells=x_cells,
                    z_value=z_value,
                )
            )

        return np.asarray(spawn_points, dtype=np.float32)

    def _build_multi_agent_goals(self, x_cells, y_cells, z_value):
        if self.num_agents == 1:
            return np.array([copy.deepcopy(self.end_point)], dtype=np.float32)

        anchor = np.array([int(self.goal_lane_x), 0], dtype=np.int32)
        free_cells = np.argwhere(self.obstacle_map == 0)
        if len(free_cells) < self.num_agents:
            raise RuntimeError("Not enough free cells to place all goal points")

        def _candidate_key(cell):
            x_idx, y_idx = int(cell[0]), int(cell[1])
            manhattan = abs(x_idx - anchor[0]) + abs(y_idx - anchor[1])
            lane_bias = abs(x_idx - int(self.goal_lane_x))
            depth_bias = y_idx
            return manhattan, lane_bias, depth_bias, np.random.random()

        ranked_cells = sorted(free_cells.tolist(), key=_candidate_key)
        selected_cells = []
        for cell in ranked_cells:
            x_idx, y_idx = int(cell[0]), int(cell[1])
            keep = True
            for sel_x, sel_y in selected_cells:
                if abs(sel_x - x_idx) + abs(sel_y - y_idx) < 2:
                    keep = False
                    break
            if keep:
                selected_cells.append((x_idx, y_idx))
            if len(selected_cells) >= self.num_agents:
                break

        if len(selected_cells) < self.num_agents:
            selected_cells = [(int(cell[0]), int(cell[1])) for cell in ranked_cells[:self.num_agents]]

        goals = []
        for x_idx, y_idx in selected_cells:
            goals.append(
                self._center_from_cell(
                    cell_centers=self.cell_centers,
                    x_idx=x_idx,
                    y_idx=y_idx,
                    x_cells=x_cells,
                    z_value=z_value,
                )
            )

        return np.asarray(goals, dtype=np.float32)

    def _use_shared_goal(self):
        return True

    def _build_goals_and_guidance_fields(self, x_cells, y_cells, z_value):
        if self._use_shared_goal():
            goals = np.asarray([copy.deepcopy(self.end_point) for _ in range(self.num_agents)], dtype=np.float32)
            return goals, None

        goals = self._build_multi_agent_goals(x_cells=x_cells, y_cells=y_cells, z_value=z_value)
        agent_goal_fields = []
        for goal in goals:
            goal_x_idx, goal_y_idx = self._position_to_grid_cell(goal)
            agent_goal_fields.append(
                self._compute_guidance_distance_field(
                    obst_map=self.obstacle_map,
                    goal_x_idx=goal_x_idx,
                    goal_y_idx=goal_y_idx,
                )
            )
        return goals, agent_goal_fields

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
        if self.nearest_valid_guidance_cells is None:
            raise RuntimeError("Nearest valid guidance lookup table is not initialized")
        anchor_x, anchor_y = self.nearest_valid_guidance_cells[x_idx, y_idx]
        if anchor_x < 0 or anchor_y < 0:
            raise RuntimeError("Failed to locate a valid guidance anchor cell")
        return int(anchor_x), int(anchor_y)

    def _build_nearest_valid_guidance_cells(self):
        nearest = np.full((self._x_cells, self._y_cells, 2), -1, dtype=np.int32)
        valid_cells = []
        for x_idx in range(self._x_cells):
            for y_idx in range(self._y_cells):
                if self.obstacle_map[x_idx, y_idx] == 0 and np.isfinite(self.guidance_distance_field[x_idx, y_idx]):
                    nearest[x_idx, y_idx] = (x_idx, y_idx)
                    valid_cells.append((x_idx, y_idx))

        if not valid_cells:
            raise RuntimeError("No valid guidance cells available to initialize lookup table")

        for x_idx in range(self._x_cells):
            for y_idx in range(self._y_cells):
                if nearest[x_idx, y_idx, 0] >= 0:
                    continue

                best_cell = None
                best_distance = np.inf
                query_xy = self._cell_center_xy(x_idx, y_idx)
                for cand_x, cand_y in valid_cells:
                    center_xy = self._cell_center_xy(cand_x, cand_y)
                    distance = float(np.linalg.norm(center_xy - query_xy))
                    if distance < best_distance:
                        best_distance = distance
                        best_cell = (cand_x, cand_y)

                nearest[x_idx, y_idx] = best_cell

        self.nearest_valid_guidance_cells = nearest

    def _continuous_guidance_distance(
        self,
        pos,
        distance_field,
        x_min=0,
        y_min=0,
        candidate_radius=1,
        anchor_cell=None,
        goal_z=None,
    ):
        if anchor_cell is None:
            anchor_x, anchor_y = self._nearest_valid_guidance_cell(pos)
        else:
            anchor_x, anchor_y = int(anchor_cell[0]), int(anchor_cell[1])
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
                distance_field[anchor_x - x_min, anchor_y - y_min] + np.linalg.norm(center_xy - pos[:2])
            )

        if goal_z is None:
            goal_z = self.end_point[2]
        guidance_z_distance = float(abs(goal_z - pos[2]))
        return guidance_xy_distance + guidance_z_distance

    def _select_local_target_cell(self, local_reach_field, x_min, x_max, y_min, y_max, goal_xy, goal_lane_x, guidance_field):
        best_boundary_candidate = None
        best_inner_candidate = None
        for x_idx in range(x_min, x_max + 1):
            for y_idx in range(y_min, y_max + 1):
                if self.obstacle_map[x_idx, y_idx] != 0:
                    continue

                local_reach_distance = local_reach_field[x_idx - x_min, y_idx - y_min]
                if not np.isfinite(local_reach_distance):
                    continue

                global_distance = guidance_field[x_idx, y_idx]
                if not np.isfinite(global_distance):
                    continue

                center_xy = self._cell_center_xy(x_idx, y_idx)
                heuristic_distance = float(np.linalg.norm(goal_xy - center_xy))
                tie_breaker = abs(int(x_idx) - int(goal_lane_x))
                candidate = (float(global_distance), float(local_reach_distance), heuristic_distance, tie_breaker, x_idx, y_idx)

                on_boundary = (x_idx == x_min or x_idx == x_max or y_idx == y_min or y_idx == y_max)
                if on_boundary:
                    if best_boundary_candidate is None or candidate < best_boundary_candidate:
                        best_boundary_candidate = candidate
                else:
                    if best_inner_candidate is None or candidate < best_inner_candidate:
                        best_inner_candidate = candidate

        if best_boundary_candidate is not None:
            _, _, _, _, target_x_idx, target_y_idx = best_boundary_candidate
            return int(target_x_idx), int(target_y_idx)

        if best_inner_candidate is not None:
            _, _, _, _, target_x_idx, target_y_idx = best_inner_candidate
            return int(target_x_idx), int(target_y_idx)

        return None

    def _invalidate_local_target_cache(self, agent_idx):
        if self.local_targets is not None:
            self.local_targets[agent_idx] = None
        if self.local_target_ages is not None:
            self.local_target_ages[agent_idx] = 0
        if self.local_target_fields is not None:
            self.local_target_fields[agent_idx] = None
        if self.local_target_windows is not None:
            self.local_target_windows[agent_idx] = None
        if self.local_last_anchor_cells is not None:
            self.local_last_anchor_cells[agent_idx] = None
        if self.local_last_positions is not None:
            self.local_last_positions[agent_idx] = None
        if self.local_last_guidance_values is not None:
            self.local_last_guidance_values[agent_idx] = None

    def _is_local_target_valid(self, agent_idx, curr_x_idx, curr_y_idx, x_min, x_max, y_min, y_max):
        if self.local_targets is None or self.local_target_ages is None:
            return False
        target = self.local_targets[agent_idx]
        if target is None:
            return False
        if self.local_target_ages[agent_idx] >= self.local_target_hold_steps:
            return False

        target_x_idx, target_y_idx = int(target[0]), int(target[1])
        if not (0 <= target_x_idx < self._x_cells and 0 <= target_y_idx < self._y_cells):
            return False
        if self.obstacle_map[target_x_idx, target_y_idx] != 0:
            return False
        if not (x_min <= target_x_idx <= x_max and y_min <= target_y_idx <= y_max):
            return False

        if abs(target_x_idx - curr_x_idx) + abs(target_y_idx - curr_y_idx) <= self.local_target_reached_cells:
            return False

        return True

    def _refresh_local_target(self, agent_idx, x_min, x_max, y_min, y_max, local_reach_field, goal_xy, goal_lane_x, guidance_field):
        target = self._select_local_target_cell(
            local_reach_field=local_reach_field,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            goal_xy=goal_xy,
            goal_lane_x=goal_lane_x,
            guidance_field=guidance_field,
        )
        if target is None:
            self._invalidate_local_target_cache(agent_idx)
            return None
        self.local_targets[agent_idx] = target
        self.local_target_ages[agent_idx] = 0
        self.local_target_fields[agent_idx] = None
        self.local_target_windows[agent_idx] = None
        if self.local_last_anchor_cells is not None:
            self.local_last_anchor_cells[agent_idx] = None
        if self.local_last_positions is not None:
            self.local_last_positions[agent_idx] = None
        if self.local_last_guidance_values is not None:
            self.local_last_guidance_values[agent_idx] = None
        return target

    def _compute_local_guidance_distance(self, pos, agent_idx=0):
        if self.free_cell_centers is None or self.free_cells is None or len(self.free_cells) == 0:
            goal = self.goals[agent_idx] if self.goals is not None else self.end_point
            return float(np.linalg.norm(goal[:2] - pos[:2]) + abs(goal[2] - pos[2]))

        curr_x_idx, curr_y_idx = self._nearest_valid_guidance_cell(pos)
        if self.local_last_anchor_cells is not None:
            last_anchor = self.local_last_anchor_cells[agent_idx]
            last_pos = self.local_last_positions[agent_idx]
            last_value = self.local_last_guidance_values[agent_idx]
            if (
                last_value is not None
                and last_pos is not None
                and np.allclose(last_pos, pos, atol=1e-6)
                and last_anchor is not None
                and int(last_anchor[0]) == int(curr_x_idx)
                and int(last_anchor[1]) == int(curr_y_idx)
            ):
                return float(last_value)

        x_cells, y_cells = self.obstacle_map.shape
        radius = self.local_guidance_window_cells
        x_min = max(0, int(curr_x_idx) - radius)
        x_max = min(x_cells - 1, int(curr_x_idx) + radius)
        y_min = max(0, int(curr_y_idx) - radius)
        y_max = min(y_cells - 1, int(curr_y_idx) + radius)
        local_window = (x_min, x_max, y_min, y_max)

        goal = self.goals[agent_idx] if self.goals is not None else self.end_point
        goal_norm = np.linalg.norm(goal[:2] - pos[:2])
        if goal_norm < 1e-6:
            return float(abs(goal[2] - pos[2]))

        if self.agent_goal_fields is not None:
            guidance_field = self.agent_goal_fields[agent_idx]
        else:
            guidance_field = self.guidance_distance_field
        goal_xy = goal[:2]
        goal_lane_x = self._position_to_grid_cell(goal)[0]

        local_obst_map = self.obstacle_map[x_min:x_max + 1, y_min:y_max + 1]
        local_target_valid = self._is_local_target_valid(
            agent_idx=agent_idx,
            curr_x_idx=curr_x_idx,
            curr_y_idx=curr_y_idx,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )

        if not local_target_valid:
            local_reach_field = self._compute_guidance_distance_field(
                obst_map=local_obst_map,
                goal_x_idx=curr_x_idx - x_min,
                goal_y_idx=curr_y_idx - y_min,
            )
            target_cell = self._refresh_local_target(
                agent_idx=agent_idx,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                local_reach_field=local_reach_field,
                goal_xy=goal_xy,
                goal_lane_x=goal_lane_x,
                guidance_field=guidance_field,
            )
        else:
            target_cell = self.local_targets[agent_idx]

        if target_cell is None:
            return self.get_global_guidance_distance(pos, agent_idx=agent_idx)

        target_x_idx, target_y_idx = int(target_cell[0]), int(target_cell[1])
        local_field = self.local_target_fields[agent_idx]
        cached_window = self.local_target_windows[agent_idx]
        if local_field is None or cached_window != local_window:
            local_field = self._compute_guidance_distance_field(
                obst_map=local_obst_map,
                goal_x_idx=target_x_idx - x_min,
                goal_y_idx=target_y_idx - y_min,
            )
            self.local_target_fields[agent_idx] = local_field
            self.local_target_windows[agent_idx] = local_window
        anchor_local_x = curr_x_idx - x_min
        anchor_local_y = curr_y_idx - y_min
        if not np.isfinite(local_field[anchor_local_x, anchor_local_y]):
            local_reach_field = self._compute_guidance_distance_field(
                obst_map=local_obst_map,
                goal_x_idx=curr_x_idx - x_min,
                goal_y_idx=curr_y_idx - y_min,
            )
            target_cell = self._refresh_local_target(
                agent_idx=agent_idx,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                local_reach_field=local_reach_field,
                goal_xy=goal_xy,
                goal_lane_x=goal_lane_x,
                guidance_field=guidance_field,
            )
            if target_cell is None:
                return self.get_global_guidance_distance(pos, agent_idx=agent_idx)
            target_x_idx, target_y_idx = int(target_cell[0]), int(target_cell[1])
            local_field = self._compute_guidance_distance_field(
                obst_map=local_obst_map,
                goal_x_idx=target_x_idx - x_min,
                goal_y_idx=target_y_idx - y_min,
            )
            self.local_target_fields[agent_idx] = local_field
            self.local_target_windows[agent_idx] = local_window

        local_distance = self._continuous_guidance_distance(
            pos=pos,
            distance_field=local_field,
            x_min=x_min,
            y_min=y_min,
            candidate_radius=1,
            anchor_cell=(curr_x_idx, curr_y_idx),
            goal_z=goal[2],
        )
        target_suffix = float(guidance_field[target_x_idx, target_y_idx])
        guidance_value = float(local_distance + target_suffix)
        if self.local_last_anchor_cells is not None:
            self.local_last_anchor_cells[agent_idx] = (int(curr_x_idx), int(curr_y_idx))
            self.local_last_positions[agent_idx] = np.array(pos, copy=True)
            self.local_last_guidance_values[agent_idx] = guidance_value
        return guidance_value


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
            raise RuntimeError("Failed to generate a traversable skill hybrid map")

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
        if self.guidance_type == "local_bfs" and self.local_target_ages is not None and self.local_targets is not None:
            for idx, target in enumerate(self.local_targets):
                if target is not None:
                    self.local_target_ages[idx] += 1
                if self.local_last_positions is not None:
                    self.local_last_positions[idx] = None
                    self.local_last_guidance_values[idx] = None
        return

    def get_global_guidance_distance(self, pos, agent_idx=0):
        goal = self.goals[agent_idx] if self.goals is not None else self.end_point
        if self.free_cell_centers is None or self.free_cell_guidance_distances is None:
            return float(np.linalg.norm(goal[:2] - pos[:2]) + abs(goal[2] - pos[2]))
        if self.agent_goal_fields is not None:
            guidance_field = self.agent_goal_fields[agent_idx]
        else:
            guidance_field = self.guidance_distance_field
        return self._continuous_guidance_distance(
            pos=pos,
            distance_field=guidance_field,
            candidate_radius=1,
            goal_z=goal[2],
        )

    def get_guidance_distance(self, pos, agent_idx=0):
        if self.guidance_type == "global_bfs":
            return self.get_global_guidance_distance(pos, agent_idx=agent_idx)
        if self.guidance_type == "local_bfs":
            return self._compute_local_guidance_distance(pos, agent_idx=agent_idx)
        goal = self.goals[agent_idx] if self.goals is not None else self.end_point
        return float(np.linalg.norm(goal - pos))

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
        self._build_nearest_valid_guidance_cells()
        self.local_targets = [None for _ in range(self.num_agents)]
        self.local_target_ages = np.zeros(self.num_agents, dtype=np.int32)
        self.local_target_fields = [None for _ in range(self.num_agents)]
        self.local_target_windows = [None for _ in range(self.num_agents)]
        self.local_last_anchor_cells = [None for _ in range(self.num_agents)]
        self.local_last_positions = [None for _ in range(self.num_agents)]
        self.local_last_guidance_values = [None for _ in range(self.num_agents)]
        for idx, (x_idx, y_idx) in enumerate(free_cells):
            cell_center = self._center_from_cell(
                cell_centers=self.cell_centers, x_idx=int(x_idx), y_idx=int(y_idx), x_cells=x_cells, z_value=z_value
            )
            self.free_cell_centers[idx] = cell_center[:2]
            self.free_cell_guidance_distances[idx] = self.guidance_distance_field[int(x_idx), int(y_idx)]

        self.update_formation_and_relate_param()
        self.spawn_points = self._build_multi_agent_spawn_points(x_cells=x_cells, y_cells=y_cells, z_value=z_value)
        self.goals, self.agent_goal_fields = self._build_goals_and_guidance_fields(
            x_cells=x_cells,
            y_cells=y_cells,
            z_value=z_value,
        )


class Scenario_o_skill_hybrid_same_goal(Scenario_o_skill_hybrid):
    pass


class Scenario_o_skill_hybrid_diff_goal(Scenario_o_skill_hybrid):
    def _use_shared_goal(self):
        return False
