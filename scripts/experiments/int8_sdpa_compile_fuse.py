import os

import torch
from diffusers.models.attention_processor import AttnProcessor2_0
from huggingface_hub import login
from models import StableT2I3DModel
from torchao.quantization import int8_weight_only, quantize_
from utils import benchmark, flush, init_pipe_models, prompts, warmup

login(token=os.getenv("HF_TOKEN_PYTORCH"))

config = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.bfloat16,
}

flush()

models_dict = init_pipe_models(config)

# SDPA
models_dict["t2i_model"].unet.set_attn_processor(AttnProcessor2_0())

# Fuse-QKV
models_dict["t2i_model"].fuse_qkv_projections()

# Quantize
models_dict["t2i_model"].unet = quantize_(
    models_dict["t2i_model"].unet, int8_weight_only(), device="cuda"
)
models_dict["t2i_model"].vae = quantize_(
    models_dict["t2i_model"].vae, int8_weight_only(), device="cuda"
)
models_dict["i_3d_model"] = quantize_(
    models_dict["i_3d_model"], int8_weight_only(), device="cuda"
)

# Compile
models_dict["t2i_model"].unet = torch.compile(
    models_dict["t2i_model"].unet, mode="max-autotune", backend="inductor"
)
models_dict["t2i_model"].vae = torch.compile(
    models_dict["t2i_model"].vae, mode="max-autotune", backend="inductor"
)
models_dict["i_3d_model"] = torch.compile(
    models_dict["i_3d_model"], mode="max-autotune", backend="inductor"
)

model = StableT2I3DModel(
    t2i_model=models_dict["t2i_model"],
    i_3d_model=models_dict["i_3d_model"],
    dtype=config["dtype"],
    device=config["device"],
)

model = warmup(model=model, warmup_iter=10, warmup_prompt="Warm-up model")

benchmark(
    model=model,
    prompt_list=prompts,
    run_name="int8-AutoQuant-SDPA-Compile-Fuse",
    config=config,
    save_file=True,
)
