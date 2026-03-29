!pip install diffusers transformers accelerate torch --quiet

from huggingface_hub import login
login()

import torch
from diffusers import StableDiffusionPipeline

model_id="runwayml/stable-diffusion-v1-5"
pipe=StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)

pipe=pipe.to("cuda")

prompt = "dogs"
image = pipe(prompt).images[0]
image

image.save("generated_image.png")
