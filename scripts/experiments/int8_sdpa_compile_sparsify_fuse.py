import os

import torch
from models import StableT2I3DModel
from torchao.quantization import (
    int8_dynamic_activation_int8_semi_sparse_weight,
    int8_weight_only,
    quantize_,
)
from torchao.sparsity import sparsify_
from utils import benchmark_run, flush, init_models, get_prompts, warmup_model

config = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.bfloat16,
}

from huggingface_hub import login

login(token=os.getenv("HF_TOKEN_PYTORCH"))

flush()

models_dict = init_models(config)

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

# Fuse-QKV
models_dict["t2i_model"].fuse_qkv_projections()

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

model = warmup_model(model=model, warmup_iter=10, warmup_prompt="Warm-up model")

benchmark_run(
    model=model,
    prompt_list=get_prompts(),
    run_name="int8-SDPA-Compile-Sparsify-FuseQKV",
    config=config,
    save_file=True,
)
