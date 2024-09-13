import os

import torch
from diffusers.models.attention_processor import AttnProcessor2_0
from huggingface_hub import login
from models import StableT2I3DModel
from torchao import autoquant
from torchao.quantization import int8_dynamic_activation_int8_semi_sparse_weight
from torchao.sparsity import sparsify_
from utils import benchmark_run, flush, init_models, get_prompts, warmup_model

login(token=os.getenv("HF_TOKEN_PYTORCH"))

config = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.bfloat16,
}

flush()

models_dict = init_models(config)

# Quantize
models_dict["t2i_model"].unet = autoquant(
    torch.compile(
        models_dict["t2i_model"].unet, mode="max-autotune", backend="inductor"
    )
)
models_dict["t2i_model"].vae = autoquant(
    torch.compile(models_dict["t2i_model"].vae, mode="max-autotune", backend="inductor")
)
models_dict["i_3d_model"] = autoquant(
    torch.compile(models_dict["i_3d_model"], mode="max-autotune", backend="inductor")
)

# SDPA
models_dict["t2i_model"].unet.set_attn_processor(AttnProcessor2_0())

# Fuse QKV
models_dict["t2i_model"].fuse_qkv_projections()

# Compile and Sparsify
models_dict["t2i_model"].unet = sparsify_(
    models_dict["t2i_model"].unet, int8_dynamic_activation_int8_semi_sparse_weight()
)
models_dict["t2i_model"].vae = sparsify_(
    models_dict["t2i_model"].vae, int8_dynamic_activation_int8_semi_sparse_weight()
)
models_dict["i_3d_model"] = sparsify_(
    models_dict["i_3d_model"], int8_dynamic_activation_int8_semi_sparse_weight()
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
    run_name="BF16-AutoQuant-Int4-SDPA-Compile-Fuse-Sparsify",
    config=config,
    save_file=True,
)
