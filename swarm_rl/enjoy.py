import sys
import json
from pathlib import Path

from sample_factory.enjoy import enjoy

from swarm_rl.train import parse_swarm_cfg, register_swarm_components


def main():
    """Script entry point."""
    # Parse config first to check if CBF was used during training
    cfg = parse_swarm_cfg(evaluation=True)

    # Manually load config.json from checkpoint to detect CBF
    # Sample Factory's parse_swarm_cfg may not load all custom params correctly
    use_cbf = False
    if hasattr(cfg, 'train_dir') and hasattr(cfg, 'experiment'):
        config_path = Path(cfg.train_dir) / cfg.experiment / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                checkpoint_cfg = json.load(f)
                use_cbf = checkpoint_cfg.get('quads_use_cbf', False)
                print(f"[INFO] Loaded config from {config_path}")
                print(f"[INFO] quads_use_cbf={use_cbf}")
        else:
            # Fallback to cfg attribute
            use_cbf = getattr(cfg, 'quads_use_cbf', False)
            print(f"[INFO] Config file not found, using cfg.quads_use_cbf={use_cbf}")
    else:
        use_cbf = getattr(cfg, 'quads_use_cbf', False)

    # IMPORTANT: CBF model inherits from ActorCriticSharedWeights
    # Must force actor_critic_share_weights=True for checkpoint compatibility
    if use_cbf:
        print(f"[INFO] CBF detected, forcing actor_critic_share_weights=True")
        print(f"[INFO] Before: actor_critic_share_weights={cfg.actor_critic_share_weights}")
        cfg.actor_critic_share_weights = True
        print(f"[INFO] After: actor_critic_share_weights={cfg.actor_critic_share_weights}")

    register_swarm_components(use_cbf=use_cbf)

    status = enjoy(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
