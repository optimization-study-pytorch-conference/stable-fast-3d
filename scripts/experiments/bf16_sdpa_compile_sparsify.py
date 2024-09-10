import os

import torch
from diffusers.models.attention_processor import AttnProcessor2_0
from huggingface_hub import login
from models import StableT2I3DModel
from torchao.quantization import int8_dynamic_activation_int8_semi_sparse_weight
from torchao.sparsity import sparsify_
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

# Compile and Sparsify
models_dict["t2i_model"].unet = sparsify_(
    torch.compile(
        models_dict["t2i_model"].unet, mode="max-autotune", backend="inductor"
    ),
    int8_dynamic_activation_int8_semi_sparse_weight(),
)
models_dict["t2i_model"].vae = sparsify_(
    torch.compile(
        models_dict["t2i_model"].vae, mode="max-autotune", backend="inductor"
    ),
    int8_dynamic_activation_int8_semi_sparse_weight(),
)
models_dict["i_3d_model"] = sparsify_(
    torch.compile(models_dict["i_3d_model"], mode="max-autotune", backend="inductor"),
    int8_dynamic_activation_int8_semi_sparse_weight(),
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
    run_name="BF16-SDPA-Compile-Sparsify",
    config=config,
    save_file=True,
)
