from swarm_rl.runs.obstacles.quad_obstacle_baseline import QUAD_BASELINE_CLI_8


ROOM_CLI = "--quads_room_dims 10 16 10 --quads_obst_spawn_area 10 16 "
COMMON_CLI = (
    "--quads_neighbor_obs_type=pos_vel --quads_encoder_type=attention "
    "--with_wandb=False --wandb_project=Quad-Swarm-RL --wandb_user=multi-drones "
    "--wandb_group=bfs "
    "--quads_use_adaptive_skill=False "
    "--quads_use_diversity_loss=False "
    "--quads_action_type=omegathrust"
)


def build_bfs_cli(guidance_type, quads_mode, neighbor_visible_num, gpu_profile="safe"):
    cli = QUAD_BASELINE_CLI_8

    if gpu_profile == "safe":
        cli = cli.replace("--device=cpu", "--device=gpu --serial_mode=True")
    elif gpu_profile == "optimized":
        cli = (
            cli.replace("--device=cpu", "--device=gpu")
            .replace("--num_envs_per_worker=2", "--num_envs_per_worker=4")
            .replace("--batch_size=1024", "--batch_size=3072 --num_batches_per_epoch=2 --num_epochs=1")
        )
    elif gpu_profile != "cpu":
        raise ValueError(f"Unknown gpu_profile: {gpu_profile}")

    return cli + (
        f"--quads_mode={quads_mode} {ROOM_CLI}"
        f"--quads_guidance_type={guidance_type} "
        f"--quads_neighbor_visible_num={neighbor_visible_num} "
        f"{COMMON_CLI}"
    )
