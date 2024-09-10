from diffusers import StableDiffusion3Pipeline

from sf3d.system import SF3D


def init_models(device, dtype):
    t2i_model = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=dtype
    ).to(device)

    i_3d_model = SF3D.from_pretrained(
        "stabilityai/stable-fast-3d",
        config_name="config.yaml",
        weight_name="model.safetensors",
    ).to(device)

    return {"t2i_model": t2i_model, "i_3d_model": i_3d_model}
