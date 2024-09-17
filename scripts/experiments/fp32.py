import os

import torch
from huggingface_hub import login
from models import StableT2I3D
from utils import (
    activate_inductor_opts,
    benchmark_run,
    flush,
    get_prompts,
    init_models,
    set_random_seed,
    warmup_model,
)

login(token=os.getenv("HF_TOKEN_PYTORCH"))

config = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.float32,
}

flush()
set_random_seed(42)
activate_inductor_opts()

models_dict = init_models(config)

model = StableT2I3D(
    t2i_model=models_dict["t2i_model"],
    i_3d_model=models_dict["i_3d_model"],
    dtype=config["dtype"],
    device=config["device"],
)

model = warmup_model(model=model, warmup_iter=3, warmup_prompt="Warm-up model")

benchmark_run(
    model=model,
    prompt_list=get_prompts(),
    run_name="FP32",
    config=config,
    save_file=True,
)
