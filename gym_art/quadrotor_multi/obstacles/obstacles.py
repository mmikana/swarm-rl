import copy
import numpy as np

from gym_art.quadrotor_multi.collisions.obstacles import (
    perform_collision_with_box_obstacle,
    perform_collision_with_obstacle,
)
from gym_art.quadrotor_multi.obstacles.utils import (
    OBSTACLE_BOX,
    OBSTACLE_CYLINDER,
    collision_detection,
    get_surround_sdfs,
)


class MultiObstacles:
    def __init__(self, obstacle_size=1.0, quad_radius=0.046):
        self.size = obstacle_size
        self.obstacle_radius = obstacle_size / 2.0
        self.quad_radius = quad_radius
        self.pos_arr = []
        self.obstacle_types = np.zeros(0, dtype=np.int32)
        self.obstacle_size_xy = np.zeros((0, 2), dtype=np.float32)
        self.obstacle_size_xyz = np.zeros((0, 3), dtype=np.float32)
        self.resolution = 0.1

    def _parse_obstacle_items(self, obstacle_items):
        positions = []
        obstacle_types = []
        obstacle_size_xy = []
        obstacle_size_xyz = []

        for item in obstacle_items:
            if isinstance(item, dict):
                obstacle_type = item.get("type", "cylinder")
                center = np.asarray(item["center"], dtype=np.float32)
                size = np.asarray(item["size"], dtype=np.float32)
                if obstacle_type == "box":
                    obstacle_types.append(OBSTACLE_BOX)
                else:
                    obstacle_types.append(OBSTACLE_CYLINDER)
            else:
                center = np.asarray(item, dtype=np.float32)
                size = np.asarray([self.size, self.size, 0.0], dtype=np.float32)
                obstacle_types.append(OBSTACLE_CYLINDER)

            positions.append(center)
            obstacle_size_xyz.append(size)
            obstacle_size_xy.append(size[:2])

        if positions:
            pos_arr = np.asarray(positions, dtype=np.float32)
            type_arr = np.asarray(obstacle_types, dtype=np.int32)
            size_xy_arr = np.asarray(obstacle_size_xy, dtype=np.float32)
            size_xyz_arr = np.asarray(obstacle_size_xyz, dtype=np.float32)
        else:
            pos_arr = np.zeros((0, 3), dtype=np.float32)
            type_arr = np.zeros(0, dtype=np.int32)
            size_xy_arr = np.zeros((0, 2), dtype=np.float32)
            size_xyz_arr = np.zeros((0, 3), dtype=np.float32)

        return pos_arr, type_arr, size_xy_arr, size_xyz_arr

    def reset(self, obs, quads_pos, pos_arr):
        self.pos_arr, self.obstacle_types, self.obstacle_size_xy, self.obstacle_size_xyz = \
            self._parse_obstacle_items(pos_arr)

        quads_sdf_obs = 100 * np.ones((len(quads_pos), 9))
        quads_sdf_obs = get_surround_sdfs(
            quad_poses=quads_pos[:, :2],
            obst_poses=self.pos_arr[:, :2],
            obstacle_types=self.obstacle_types,
            obstacle_size_xy=self.obstacle_size_xy,
            quads_sdf_obs=quads_sdf_obs,
            resolution=self.resolution,
        )

        obs = np.concatenate((obs, quads_sdf_obs), axis=1)

        return obs

    def step(self, obs, quads_pos):
        quads_sdf_obs = 100 * np.ones((len(quads_pos), 9))
        quads_sdf_obs = get_surround_sdfs(
            quad_poses=quads_pos[:, :2],
            obst_poses=self.pos_arr[:, :2],
            obstacle_types=self.obstacle_types,
            obstacle_size_xy=self.obstacle_size_xy,
            quads_sdf_obs=quads_sdf_obs,
            resolution=self.resolution,
        )

        obs = np.concatenate((obs, quads_sdf_obs), axis=1)

        return obs

    def collision_detection(self, pos_quads):
        quad_collisions = collision_detection(
            quad_poses=pos_quads[:, :2],
            obst_poses=self.pos_arr[:, :2],
            obstacle_types=self.obstacle_types,
            obstacle_size_xy=self.obstacle_size_xy,
            quad_radius=self.quad_radius,
        )

        collided_quads_id = np.where(quad_collisions > -1)[0]
        collided_obstacles_id = quad_collisions[collided_quads_id]
        quad_obst_pair = {}
        for i, key in enumerate(collided_quads_id):
            quad_obst_pair[key] = int(collided_obstacles_id[i])

        return collided_quads_id, quad_obst_pair

    def perform_collision_response(self, drone_dyn, obstacle_id):
        if self.obstacle_types[obstacle_id] == OBSTACLE_BOX:
            perform_collision_with_box_obstacle(
                drone_dyn=drone_dyn,
                obstacle_pos=self.pos_arr[obstacle_id],
                obstacle_size_xy=self.obstacle_size_xy[obstacle_id],
            )
        else:
            perform_collision_with_obstacle(
                drone_dyn=drone_dyn,
                obstacle_pos=self.pos_arr[obstacle_id],
                obstacle_size=float(self.obstacle_size_xy[obstacle_id][0]),
            )
