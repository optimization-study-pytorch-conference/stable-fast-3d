import os

import torch
from huggingface_hub import login
from models import StableT2I3D
from utils import benchmark_run, flush, init_models, get_prompts, warmup_model, activate_inductor_opts, set_random_seed
from torchao import autoquant

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

model = StableT2I3D(
    t2i_model=models_dict["t2i_model"],
    i_3d_model=models_dict["i_3d_model"],
    dtype=config["dtype"],
    device=config["device"],
)

model.t2i_pipe.transformer = autoquant(
    torch.compile(
        model.t2i_pipe.transformer, mode="max-autotune", backend="inductor", fullgraph=True
    )
)
model.t2i_pipe.vae = autoquant(
    torch.compile(model.t2i_pipe.vae, mode="max-autotune", backend="inductor", fullgraph=True)
)
model.i_3d_model = autoquant(
    torch.compile(model.i_3d_model, mode="max-autotune", backend="inductor", fullgraph=True)
)



model = warmup_model(model=model, warmup_iter=3, warmup_prompt="Warm-up model")

benchmark_run(
    model=model,
    prompt_list=get_prompts(),
    run_name="BF16-FChunk-Compile-Fuse-AutoQuant",
    config=config,
    save_file=True,
)
