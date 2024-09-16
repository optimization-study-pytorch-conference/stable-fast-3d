# from diffusers import PixArtSigmaPipeline
from diffusers import StableDiffusion3Pipeline

from sf3d.system import SF3D


def init_models(config):
    t2i_model = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=config["dtype"]
    ).to(config["device"])

    # t2i_model = PixArtSigmaPipeline.from_pretrained(
    #     "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", torch_dtype=config["dtype"]
    # ).to(config["device"])

    i_3d_model = SF3D.from_pretrained(
        "stabilityai/stable-fast-3d",
        config_name="config.yaml",
        weight_name="model.safetensors",
    ).to(config["device"])

    return {"t2i_model": t2i_model, "i_3d_model": i_3d_model}
