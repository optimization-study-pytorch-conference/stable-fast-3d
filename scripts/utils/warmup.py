from tqdm.autonotebook import tqdm


def warmup_model(model, warmup_iter, warmup_prompt):
    warmup_iter = 10
    print("Warm-up started")

    for _ in tqdm(range(warmup_iter)):

        _ = model.generate(warmup_prompt)

    print("Warm-up complete")
