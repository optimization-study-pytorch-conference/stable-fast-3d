import os

import torch
from diffusers.models.attention_processor import AttnProcessor2_0
from huggingface_hub import login
from models import StableT2I3D
from torchao.quantization import (
    int8_dynamic_activation_int8_semi_sparse_weight,
    int8_weight_only,
    quantize_,
)
from torchao.sparsity import sparsify_
from utils import benchmark_run, flush, init_models, get_prompts, warmup_model

login(token=os.getenv("HF_TOKEN_PYTORCH"))

config = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.bfloat16,
}

flush()

models_dict = init_models(config)

torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

# Quantize
models_dict["t2i_model"].transformer = quantize_(
    models_dict["t2i_model"].transformer, int8_weight_only(), device="cuda"
)
models_dict["t2i_model"].vae.decode = quantize_(
    models_dict["t2i_model"].vae.decode, int8_weight_only(), device="cuda"
)
models_dict["i_3d_model"] = quantize_(
    models_dict["i_3d_model"], int8_weight_only(), device="cuda"
)

# SDPA
models_dict["t2i_model"].transformer.set_attn_processor(AttnProcessor2_0())

# Compile and Sparsify
models_dict["t2i_model"].transformer = sparsify_(
    torch.compile(
        models_dict["t2i_model"].transformer, mode="max-autotune", backend="inductor", fullgraph=True
    ),
    int8_dynamic_activation_int8_semi_sparse_weight(),
)
models_dict["t2i_model"].vae.decode = sparsify_(
    torch.compile(
        models_dict["t2i_model"].vae.decode, mode="max-autotune", backend="inductor", fullgraph=True
    ),
    int8_dynamic_activation_int8_semi_sparse_weight(),
)
models_dict["i_3d_model"] = sparsify_(
    torch.compile(models_dict["i_3d_model"], mode="max-autotune", backend="inductor", fullgraph=True),
    int8_dynamic_activation_int8_semi_sparse_weight(),
)

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
    run_name="int8-SDPA-Compile-Sparsify",
    config=config,
    save_file=True,
)
