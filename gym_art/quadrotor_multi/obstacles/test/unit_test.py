import numpy as np

from gym_art.quadrotor_multi.obstacles.utils import (
    OBSTACLE_BOX,
    OBSTACLE_CYLINDER,
    collision_detection,
    get_cell_centers,
    get_surround_sdfs,
)


def test_get_surround_sdfs():
    quad_poses = np.array([[0., 0.]])
    obst_poses = np.array([[0.2, 0.]])
    obstacle_types = np.array([OBSTACLE_CYLINDER])
    obstacle_size_xy = np.array([[0.6, 0.6]])
    quads_sdf_obs = 100 * np.ones((len(quad_poses), 9))

    # get_surround_sdfs
    dist = []
    for i, x in enumerate([-0.1, 0, 0.1]):
        for j, y in enumerate([-0.1, 0, 0.1]):
            tmp = np.linalg.norm([x - obst_poses[0][0], y - obst_poses[0][1]]) - 0.3
            dist.append(tmp)

    test_res = get_surround_sdfs(
        quad_poses, obst_poses, obstacle_types, obstacle_size_xy, quads_sdf_obs, resolution=0.1
    )
    true_res = np.array(dist)
    assert test_res.all() == true_res.all()
    return


def test_collision_detection():
    quad_poses = np.array([[0., 0.]])
    obst_poses = np.array([[0.2, 0.]])
    obstacle_types = np.array([OBSTACLE_CYLINDER])
    obstacle_size_xy = np.array([[0.6, 0.6]])
    # collision_detection
    quad_collisions = collision_detection(
        quad_poses, obst_poses, obstacle_types, obstacle_size_xy, quad_radius=0.046
    )
    test_res = np.where(quad_collisions > -1)[0]
    true_res = np.array([0])
    assert test_res.all() == true_res.all()
    return


def test_box_collision_detection():
    quad_poses = np.array([[0.49, 0.0]])
    obst_poses = np.array([[0.0, 0.0]])
    obstacle_types = np.array([OBSTACLE_BOX])
    obstacle_size_xy = np.array([[1.0, 1.0]])
    quad_collisions = collision_detection(
        quad_poses, obst_poses, obstacle_types, obstacle_size_xy, quad_radius=0.046
    )
    test_res = np.where(quad_collisions > -1)[0]
    true_res = np.array([0])
    assert test_res.all() == true_res.all()
    return


def test_get_cell_centers():
    obst_area_length = 8.0
    obst_area_width = 8.0
    grid_size = 1.0
    test_res = get_cell_centers(obst_area_length=obst_area_length, obst_area_width=obst_area_width, grid_size=grid_size)

    true_res = np.array([
        (i + (grid_size / 2) - obst_area_length // 2, j + (grid_size / 2) - obst_area_width // 2)
        for i in np.arange(0, obst_area_length, grid_size)
        for j in np.arange(obst_area_width - grid_size, -grid_size, -grid_size)])

    assert test_res.all() == true_res.all()
    return


def unit_test():
    test_get_surround_sdfs()
    test_collision_detection()
    test_box_collision_detection()
    test_get_cell_centers()
    print('Pass unit test!')
    return


if __name__ == "__main__":
    unit_test()
