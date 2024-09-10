import os

import torch
from huggingface_hub import login
from models import StableT2I3D
from utils import benchmark, flush, init_pipe_models, prompts, warmup

login(token=os.getenv("HF_TOKEN_PYTORCH"))

config = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.float32,
}

flush()

models_dict = init_pipe_models(config)

model = StableT2I3D(
    t2i_model=models_dict["t2i_model"],
    i_3d_model=models_dict["i_3d_model"],
    dtype=config["dtype"],
    device=config["device"],
)

model = warmup(model=model, warmup_iter=10, warmup_prompt="Warm-up model")

benchmark(
    model=model, prompt_list=prompts, run_name="FP32", config=config, save_file=True
)
