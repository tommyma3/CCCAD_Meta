from .darkroom import sample_darkroom, sample_darkroom_permuted, Darkroom, DarkroomPermuted, map_dark_states, map_dark_states_inverse
import metaworld
from gymnasium.wrappers import TimeLimit


ENVIRONMENT = {
    'darkroom': Darkroom,
    'darkroompermuted': DarkroomPermuted,
}


SAMPLE_ENVIRONMENT = {
    'darkroom': sample_darkroom,
    'darkroompermuted': sample_darkroom_permuted,
}


def make_env(config, env_cls=None, task=None, **kwargs):
    """
    Create environment factory function.
    
    For metaworld: requires env_cls and task parameters
    For darkroom: uses config['env'] and kwargs (e.g., goal)
    """
    if config['env'] == 'metaworld':
        def _init():
            env = env_cls()
            env.set_task(task)
            return TimeLimit(env, max_episode_steps=config['horizon'])
        return _init
    else:
        def _init():
            return ENVIRONMENT[config['env']](config, **kwargs)
        return _init