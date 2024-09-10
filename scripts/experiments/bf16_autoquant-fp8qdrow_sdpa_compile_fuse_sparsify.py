import os

import torch
from diffusers.models.attention_processor import AttnProcessor2_0
from huggingface_hub import login
from models import StableT2I3DModel
from torchao.quantization import (
    float8_dynamic_activation_float8_weight,
    int8_dynamic_activation_int8_semi_sparse_weight,
    quantize_,
)
from torchao.quantization.quant_api import PerRow
from torchao.sparsity import sparsify_
from utils import benchmark, flush, init_pipe_models, prompts, warmup

login(token=os.getenv("HF_TOKEN_PYTORCH"))

config = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.bfloat16,
}

flush()

models_dict = init_pipe_models(config)

# Quantize
models_dict["t2i_model"].unet = quantize_(
    models_dict["t2i_model"].unet,
    float8_dynamic_activation_float8_weight(granularity=PerRow()),
)
models_dict["t2i_model"].vae = quantize_(
    models_dict["t2i_model"].vae,
    float8_dynamic_activation_float8_weight(granularity=PerRow()),
)
models_dict["i_3d_model"] = quantize_(
    models_dict["i_3d_model"],
    float8_dynamic_activation_float8_weight(granularity=PerRow()),
)

# SDPA
models_dict["t2i_model"].unet.set_attn_processor(AttnProcessor2_0())

# Fuse QKV
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

model = warmup(model=model, warmup_iter=10, warmup_prompt="Warm-up model")

benchmark(
    model=model,
    prompt_list=prompts,
    run_name="BF16-AutoQuant-FP8QDRow-SDPA-Compile-Fuse-Sparsify",
    config=config,
    save_file=True,
)
