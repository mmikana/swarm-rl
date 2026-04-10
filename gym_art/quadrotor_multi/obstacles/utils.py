import numpy as np
from numba import njit

OBSTACLE_CYLINDER = 0
OBSTACLE_BOX = 1


@njit
def obstacle_signed_distance(point_xy, obstacle_center_xy, obstacle_type, obstacle_size_xy):
    rel_x = point_xy[0] - obstacle_center_xy[0]
    rel_y = point_xy[1] - obstacle_center_xy[1]

    if obstacle_type == OBSTACLE_CYLINDER:
        radius = obstacle_size_xy[0] / 2.0
        return np.sqrt(rel_x * rel_x + rel_y * rel_y) - radius

    half_x = obstacle_size_xy[0] / 2.0
    half_y = obstacle_size_xy[1] / 2.0
    qx = abs(rel_x) - half_x
    qy = abs(rel_y) - half_y
    ox = max(qx, 0.0)
    oy = max(qy, 0.0)
    outside = np.sqrt(ox * ox + oy * oy)
    inside = min(max(qx, qy), 0.0)
    return outside + inside


@njit
def get_surround_sdfs(quad_poses, obst_poses, obstacle_types, obstacle_size_xy, quads_sdf_obs, resolution=0.1):
    # Shape of quads_sdf_obs: (quad_num, 9)

    sdf_map = np.array([-1., -1., -1., 0., 0., 0., 1., 1., 1.])
    sdf_map *= resolution

    for i, q_pos in enumerate(quad_poses):
        q_pos_x, q_pos_y = q_pos[0], q_pos[1]

        for g_i, g_x in enumerate([q_pos_x - resolution, q_pos_x, q_pos_x + resolution]):
            for g_j, g_y in enumerate([q_pos_y - resolution, q_pos_y, q_pos_y + resolution]):
                grid_pos = np.array([g_x, g_y])

                min_dist = 100.0
                for obst_idx, o_pos in enumerate(obst_poses):
                    dist = obstacle_signed_distance(
                        point_xy=grid_pos,
                        obstacle_center_xy=o_pos,
                        obstacle_type=obstacle_types[obst_idx],
                        obstacle_size_xy=obstacle_size_xy[obst_idx],
                    )
                    if dist < min_dist:
                        min_dist = dist

                g_id = g_i * 3 + g_j
                quads_sdf_obs[i, g_id] = min_dist

    return quads_sdf_obs


@njit
def collision_detection(quad_poses, obst_poses, obstacle_types, obstacle_size_xy, quad_radius):
    quad_num = len(quad_poses)
    # Get distance matrix b/w quad and obst
    quad_collisions = -1 * np.ones(quad_num)
    for i, q_pos in enumerate(quad_poses):
        for j, o_pos in enumerate(obst_poses):
            dist = obstacle_signed_distance(
                point_xy=q_pos,
                obstacle_center_xy=o_pos,
                obstacle_type=obstacle_types[j],
                obstacle_size_xy=obstacle_size_xy[j],
            )
            if dist <= quad_radius:
                quad_collisions[i] = j
                break

    return quad_collisions


@njit
def _segment_circle_collision(p0, p1, center, radius):
    d = p1 - p0
    a = np.dot(d, d)
    if a < 1e-12:
        rel = p0 - center
        return np.dot(rel, rel) <= radius * radius

    t = -np.dot(p0 - center, d) / a
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    closest = p0 + t * d
    rel = closest - center
    return np.dot(rel, rel) <= radius * radius


@njit
def _segment_box_collision(p0, p1, center, half_x, half_y):
    min_x = center[0] - half_x
    max_x = center[0] + half_x
    min_y = center[1] - half_y
    max_y = center[1] + half_y

    if min_x <= p0[0] <= max_x and min_y <= p0[1] <= max_y:
        return True
    if min_x <= p1[0] <= max_x and min_y <= p1[1] <= max_y:
        return True

    d = p1 - p0
    t_min = 0.0
    t_max = 1.0

    if abs(d[0]) < 1e-12:
        if p0[0] < min_x or p0[0] > max_x:
            return False
    else:
        inv_dx = 1.0 / d[0]
        tx1 = (min_x - p0[0]) * inv_dx
        tx2 = (max_x - p0[0]) * inv_dx
        if tx1 > tx2:
            tx1, tx2 = tx2, tx1
        if tx1 > t_min:
            t_min = tx1
        if tx2 < t_max:
            t_max = tx2
        if t_min > t_max:
            return False

    if abs(d[1]) < 1e-12:
        if p0[1] < min_y or p0[1] > max_y:
            return False
    else:
        inv_dy = 1.0 / d[1]
        ty1 = (min_y - p0[1]) * inv_dy
        ty2 = (max_y - p0[1]) * inv_dy
        if ty1 > ty2:
            ty1, ty2 = ty2, ty1
        if ty1 > t_min:
            t_min = ty1
        if ty2 < t_max:
            t_max = ty2
        if t_min > t_max:
            return False

    return True


@njit
def collision_detection_swept(prev_quad_poses, quad_poses, obst_poses, obstacle_types, obstacle_size_xy, quad_radius):
    quad_num = len(quad_poses)
    quad_collisions = -1 * np.ones(quad_num)
    for i in range(quad_num):
        p0 = prev_quad_poses[i]
        p1 = quad_poses[i]
        for j, o_pos in enumerate(obst_poses):
            dist_curr = obstacle_signed_distance(
                point_xy=p1,
                obstacle_center_xy=o_pos,
                obstacle_type=obstacle_types[j],
                obstacle_size_xy=obstacle_size_xy[j],
            )
            if dist_curr <= quad_radius:
                quad_collisions[i] = j
                break

            if obstacle_types[j] == OBSTACLE_CYLINDER:
                radius = obstacle_size_xy[j][0] / 2.0 + quad_radius
                if _segment_circle_collision(p0=p0, p1=p1, center=o_pos, radius=radius):
                    quad_collisions[i] = j
                    break
            else:
                half_x = obstacle_size_xy[j][0] / 2.0 + quad_radius
                half_y = obstacle_size_xy[j][1] / 2.0 + quad_radius
                if _segment_box_collision(p0=p0, p1=p1, center=o_pos, half_x=half_x, half_y=half_y):
                    quad_collisions[i] = j
                    break

    return quad_collisions


@njit
def get_cell_centers(obst_area_length, obst_area_width, grid_size=1.):
    count = 0
    i_len = obst_area_length / grid_size
    j_len = obst_area_width / grid_size
    cell_centers = np.zeros((int(i_len * j_len), 2))
    for i in np.arange(0, obst_area_length, grid_size):
        for j in np.arange(obst_area_width - grid_size, -grid_size, -grid_size):
            cell_centers[count][0] = i + (grid_size / 2) - obst_area_length // 2
            cell_centers[count][1] = j + (grid_size / 2) - obst_area_width // 2
            count += 1

    return cell_centers


if __name__ == "__main__":
    from gym_art.quadrotor_multi.obstacles.test.unit_test import unit_test
    from gym_art.quadrotor_multi.obstacles.test.speed_test import speed_test

    # Unit Test
    unit_test()
    speed_test()
