import os

import torch
from huggingface_hub import login
from models import StableT2I3D
from torchao.quantization import int8_weight_only, quantize_
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
    "dtype": torch.bfloat16,
}

flush()
set_random_seed(42)
activate_inductor_opts()

models_dict = init_models(config)

models_dict["t2i_model"].transformer.enable_forward_chunking()
models_dict["t2i_model"].transformer.fuse_qkv_projections()

quantize_(
    torch.compile(
        models_dict["t2i_model"].transformer,
        mode="max-autotune",
        backend="inductor",
        fullgraph=True,
    ),
    int8_weight_only(),
    device="cuda",
)
quantize_(
    torch.compile(
        models_dict["t2i_model"].vae,
        mode="max-autotune",
        backend="inductor",
        fullgraph=True,
    ),
    int8_weight_only(),
    device="cuda",
)
quantize_(
    torch.compile(
        models_dict["i_3d_model"],
        mode="max-autotune",
        backend="inductor",
        fullgraph=True,
    ),
    int8_weight_only(),
    device="cuda",
)

model = StableT2I3D(
    t2i_model=models_dict["t2i_model"],
    i_3d_model=models_dict["i_3d_model"],
    dtype=config["dtype"],
    device=config["device"],
)

print(dir(model))
print(type(model))
print(dir(model.t2i_pipe))
print(type(model.t2i_pipe))
print(dir(model.t2i_pipe.transformer))
print(type(model.t2i_pipe.transformer))

model = warmup_model(model=model, warmup_iter=3, warmup_prompt="Warm-up model")

benchmark_run(
    model=model,
    prompt_list=get_prompts(),
    run_name="BF16-FChunk-Compile-Fuse-Quantize",
    config=config,
    save_file=True,
)
