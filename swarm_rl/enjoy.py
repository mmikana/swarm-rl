import sys

from sample_factory.enjoy import enjoy

from swarm_rl.train import parse_swarm_cfg, register_swarm_components


def main():
    """Script entry point."""
    # Parse config first to check if CBF was used during training
    cfg = parse_swarm_cfg(evaluation=True)

    # Register components with CBF support if needed
    use_cbf = getattr(cfg, 'quads_use_cbf', False)
    register_swarm_components(use_cbf=use_cbf)

    status = enjoy(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
