# utils/__init__.py

# Keep the __all__ definition
__all__ = ["handle_image", "benchmark_run", "flush", "init_models", "prompts", "warmup_model"]

# Import functions from other modules, but delay 'prompts'
from .background import handle_image
from .benchmark import benchmark_run
from .flush_memory import flush
from .init_pipe_models import init_models
from .warmup import warmup_model

# Use lazy import for 'prompts'
def get_prompts():
    from .prompt_list import prompts
    return prompts