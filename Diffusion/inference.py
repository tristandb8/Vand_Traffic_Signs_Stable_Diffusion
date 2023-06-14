
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import os
from os import path
import uuid
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--lora-weights')
    parser.add_argument('--count', default=1, type=int)
    args = parser.parse_args()

    print(vars(args))

    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if args.lora_weights is not None:
        pipe.unet.load_attn_procs(args.lora_weights)
    pipe.to("cuda")

    iterations = args.count
    batch_size = 1

    out_dir = args.out_dir
    if not path.exists(out_dir):
        os.mkdir(out_dir)

    for _ in range(iterations):
        batch = [args.caption] * batch_size
        images = pipe(batch, num_inference_steps=25).images
        for image in images:
            if not image.getbbox(): continue # if nsfw filter blocked content
            image.save(path.join(out_dir, f'vandalized_sign_{uuid.uuid4()}.png'))