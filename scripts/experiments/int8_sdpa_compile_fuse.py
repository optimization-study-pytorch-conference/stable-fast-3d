import os

import torch
from huggingface_hub import login
from models import StableT2I3D
from torchao.quantization import int8_weight_only, quantize_
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

# # Fuse-QKV
# models_dict["t2i_model"].fuse_qkv_projections()

# Quantize
models_dict["t2i_model"].transformer = quantize_(
    models_dict["t2i_model"].transformer, int8_weight_only(), device="cuda"
)
models_dict["t2i_model"].vae = quantize_(
    models_dict["t2i_model"].vae, int8_weight_only(), device="cuda"
)
models_dict["i_3d_model"] = quantize_(
    models_dict["i_3d_model"], int8_weight_only(), device="cuda"
)

# Compile
models_dict["t2i_model"].transformer = torch.compile(
    models_dict["t2i_model"].transformer, mode="max-autotune", backend="inductor", fullgraph=True
)
models_dict["t2i_model"].vae.decode = torch.compile(
    models_dict["t2i_model"].vae.decode, mode="max-autotune", backend="inductor", fullgraph=True
)
models_dict["i_3d_model"] = torch.compile(
    models_dict["i_3d_model"], mode="max-autotune", backend="inductor", fullgraph=True
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
    run_name="int8-AutoQuant-SDPA-Compile-Fuse",
    config=config,
    save_file=True,
)
