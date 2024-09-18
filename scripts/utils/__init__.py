# utils/__init__.py

# Keep the __all__ definition
__all__ = [
    "handle_image",
    "benchmark_run",
    "flush",
    "init_models",
    "prompts",
    "warmup_model",
    "activate_inductor_opts",
    "set_random_seed",
]

# Import functions from other modules, but delay 'prompts'
from .background import handle_image
from .benchmark import benchmark_run
from .flush_memory import flush
from .inductor_opts import activate_inductor_opts
from .init_pipe_models import init_models
from .random_seed_set import set_random_seed
from .warmup import warmup_model


# Use lazy import for 'prompts'
def get_prompts():
    from .prompt_list import prompts

    return prompts
