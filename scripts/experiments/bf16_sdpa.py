import os

import torch
from diffusers.models.attention_processor import AttnProcessor2_0
from huggingface_hub import login
from models import StableT2I3D
from utils import benchmark_run, flush, init_models, get_prompts, warmup_model

login(token=os.getenv("HF_TOKEN_PYTORCH"))

config = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.bfloat16,
}

flush()

models_dict = init_models(config)

models_dict["t2i_model"].unet.set_attn_processor(AttnProcessor2_0())

model = StableT2I3D(
    t2i_model=models_dict["t2i_model"],
    i_3d_model=models_dict["i_3d_model"],
    dtype=config["dtype"],
    device=config["device"],
)

model = warmup_model(model=model, warmup_iter=10, warmup_prompt="Warm-up model")

benchmark_run(
    model=model,
    prompt_list=get_prompts(),
    run_name="BF16-SDPA",
    config=config,
    save_file=True,
)
