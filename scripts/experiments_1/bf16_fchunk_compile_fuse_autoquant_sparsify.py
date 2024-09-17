import os

import torch
from huggingface_hub import login
from models import StableT2I3D
from utils import benchmark_run, flush, init_models, get_prompts, warmup_model, activate_inductor_opts, set_random_seed
from torchao import autoquant
from torchao.sparsity import sparsify_, int8_dynamic_activation_int8_semi_sparse_weight

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

models_dict["t2i_model"].transformer = autoquant(
    torch.compile(
        models_dict["t2i_model"].transformer, mode="max-autotune", backend="inductor", fullgraph=True
    ), error_on_unseen=False
)
models_dict["t2i_model"].vae = autoquant(
    torch.compile(models_dict["t2i_model"].vae, mode="max-autotune", backend="inductor", fullgraph=True), error_on_unseen=False
)

# Compile and Sparsify
models_dict["t2i_model"].transformer = sparsify_(
    models_dict["t2i_model"].transformer,
    int8_dynamic_activation_int8_semi_sparse_weight(),
)
models_dict["t2i_model"].vae = sparsify_(
    models_dict["t2i_model"].vae,
    int8_dynamic_activation_int8_semi_sparse_weight(),
)
models_dict["i_3d_model"] = sparsify_(
    models_dict["i_3d_model"],
    int8_dynamic_activation_int8_semi_sparse_weight(),
)

model = StableT2I3D(
    t2i_model=models_dict["t2i_model"],
    i_3d_model=models_dict["i_3d_model"],
    dtype=config["dtype"],
    device=config["device"],
)

model = warmup_model(model=model, warmup_iter=3, warmup_prompt="Warm-up model")

benchmark_run(
    model=model,
    prompt_list=get_prompts(),
    run_name="BF16-SDPA-Compile-Fuse-AutoQuant-Sparsify",
    config=config,
    save_file=True,
)
