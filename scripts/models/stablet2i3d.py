import os

import rembg
import torch
from utils import handle_image


class StableT2I3D(torch.nn.Module):
    def __init__(
        self,
        t2i_model,
        i_3d_model,
        dtype=torch.float32,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        **kwargs
    ):
        super(StableT2I3D, self).__init__(**kwargs)
        self.t2i_pipe = t2i_model
        self.i_3d_model = i_3d_model
        self.dtype = dtype
        self.device = device
        self.rembg_session = rembg.new_session()

        self.t2i_pipe.transformer.to(memory_format=torch.channels_last)
        self.t2i_pipe.vae.to(memory_format=torch.channels_last)

    def generate(self, prompt):
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=self.dtype):

                start_t2i = torch.cuda.Event(enable_timing=True)
                end_t2i = torch.cuda.Event(enable_timing=True)

                start_t2i.record()
                image = self.t2i_pipe(
                    prompt=prompt, num_inference_steps=28, height=1024, width=1024
                ).images[0]
                end_t2i.record()

                torch.cuda.synchronize()

                t2i_generation_time = start_t2i.elapsed_time(end_t2i)

                start_bg_remove = torch.cuda.Event(enable_timing=True)
                end_bg_remove = torch.cuda.Event(enable_timing=True)

                start_bg_remove.record()
                bg_removed_image = handle_image(image, self.rembg_session)
                end_bg_remove.record()

                torch.cuda.synchronize()

                bg_remove_time = start_bg_remove.elapsed_time(end_bg_remove)

                start_3d_generation = torch.cuda.Event(enable_timing=True)
                end_3d_generation = torch.cuda.Event(enable_timing=True)

                start_3d_generation.record()
                mesh, glob_dict = self.i_3d_model.run_image(
                    bg_removed_image, bake_resolution=1024, remesh="none"
                )
                end_3d_generation.record()

                torch.cuda.synchronize()

                generation_time = start_3d_generation.elapsed_time(end_3d_generation)

                total_time = t2i_generation_time + bg_remove_time + generation_time

        return (
            mesh,
            glob_dict,
            image,
            bg_removed_image,
            t2i_generation_time,
            bg_remove_time,
            generation_time,
            total_time,
        )

    def save_result(self, mesh, out_mesh_path=os.path.join(os.getcwd(), "mesh.glb")):
        mesh.export(out_mesh_path, include_normals=True)
        return True
