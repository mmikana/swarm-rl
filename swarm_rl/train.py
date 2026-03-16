"""
Main script for training a swarm of quadrotors with SampleFactory

"""

import sys

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl

from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env
from swarm_rl.env_wrappers.quadrotor_params import add_quadrotors_env_args, quadrotors_override_defaults
from swarm_rl.models.quad_multi_model import register_models

# Global reference to CBF model class (for pickling)
_CBF_MODEL_CLASS = None


def make_actor_critic_with_cbf(cfg, obs_space, action_space):
    """
    Custom Actor-Critic factory that uses CBF.
    This is a module-level function to support pickling for multiprocessing.
    """
    from sample_factory.algo.utils.context import global_model_factory
    from swarm_rl.models.quad_multi_model_rcbf import QuadActorCriticWithCBF
    
    model_factory = global_model_factory()
    return QuadActorCriticWithCBF(model_factory, obs_space, action_space, cfg)


def register_swarm_components(use_cbf=False):
    """
    Register swarm environment and models

    Args:
        use_cbf: If True, register CBF-enabled Actor-Critic model
    """
    register_env("quadrotor_multi", make_quadrotor_env)
    register_models()

    # Register custom Actor-Critic if using CBF
    if use_cbf:
        from sample_factory.algo.utils.context import global_model_factory
        
        # Store reference for pickling
        global _CBF_MODEL_CLASS
        
        # Override the default Actor-Critic factory
        global_model_factory().make_actor_critic_func = make_actor_critic_with_cbf


def parse_swarm_cfg(argv=None, evaluation=False):
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_quadrotors_env_args(partial_cfg.env, parser)
    quadrotors_override_defaults(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def main():
    """Script entry point."""
    # Parse config first to check if CBF is enabled
    cfg = parse_swarm_cfg(evaluation=False)

    # Register components (with or without CBF)
    use_cbf = getattr(cfg, 'quads_use_cbf', False)
    register_swarm_components(use_cbf=use_cbf)

    # Run training
    status = run_rl(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
