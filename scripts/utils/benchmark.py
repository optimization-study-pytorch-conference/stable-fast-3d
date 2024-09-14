import os

import torch
import wandb
from tqdm.autonotebook import tqdm


def benchmark_run(model, prompts_list, config, save_file=True):
    run = wandb.init(
        project="PyTorch-Conference-3D-Optimization",
        entity="suvadityamuk",
        name=config["run_name"],
        save_code=True,
        config=config,
    )

    result_table = wandb.Table(
        columns=[
            "Prompt",
            "Generated-Image",
            "Background-Removed-Image",
            "T2I-Generation-Time",
            "BG-Remove-Time",
            "3D-Generation-Time",
            "Total-Time",
        ]
    )

    for idx, prompt in tqdm(enumerate(prompts_list), total=len(prompts_list)):

        torch.cuda.synchronize()

        (
            mesh,
            glob_dict,
            image,
            bg_removed_image,
            t2i_generation_time,
            bg_remove_time,
            generation_time,
            total_time,
        ) = model.generate(prompt)

        if save_file:
            save_path_dir = os.path.join(os.getcwd(), str(idx))
            if not os.path.exists(save_path_dir):
                os.makedirs(save_path_dir)
            save_path = os.path.join(save_path_dir, "mesh.glb")
            model.save_result(mesh, glob_dict, save_path)

        result_table.add_data(
            prompt,
            wandb.Image(image, caption=prompt),
            wandb.Image(bg_removed_image, caption=prompt),
            t2i_generation_time,
            bg_remove_time,
            generation_time,
            total_time,
        )

        run.log(
            {
                "prompt": prompt,
                "generated_image": wandb.Image(image, caption=prompt),
                "background_removed_image": wandb.Image(
                    bg_removed_image, caption=prompt
                ),
                "3d_generation_sample": wandb.Object3D(open(save_path)),
                "t2i-generation-time": t2i_generation_time,
                "bg_remove_time": bg_remove_time,
                "3d_generation_time": generation_time,
                "total_time": total_time,
            }
        )

    run.log({"result_table": result_table})

    run.finish()
