import modal
import os
import subprocess
import requests

#initialize
app = modal.App("sdxl-lora")
volume = modal.Volume.from_name("sdxl-volume")
LOCAL_DATA_DIR = "/Users/timothyniko/Desktop/lora_package_data"

#set image
def setup_modal_image():
    image = (
        modal.Image.debian_slim()
        .pip_install([
            "accelerate",
            "transformers",
            "datasets",
            "torch",
            "torchvision",
            "numpy",
            "huggingface-hub",
            "scipy",
            "safetensors",
            "diffusers",
            "peft",
            "xformers",
            "wandb",
            "ftfy",
            "regex",
            "bitsandbytes",
        ])
    )
    #mount data inside container
    image = image.add_local_dir(LOCAL_DATA_DIR, "/root/data")
    return image

#train environment params set
@app.function(
    image=setup_modal_image(),
    gpu=modal.gpu.A100(count=1),
    volumes={"/training": volume},
    timeout=86400
)
#training function
def train_lora():
    
    #official HF training script
    script_url = (
        "https://raw.githubusercontent.com/huggingface/"
        "diffusers/v0.25.0/examples/text_to_image/train_text_to_image_lora_sdxl.py"
    )
    response = requests.get(script_url)
    with open("train_text_to_image_lora_sdxl.py", "w") as f:
        f.write(response.text)
    
    #training params
    training_args = [
        "python", "train_text_to_image_lora_sdxl.py",
        "--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0",
        "--train_data_dir=/root/data",
        "--output_dir=/training/lora_output",
        "--resolution=1024",
        "--train_batch_size=2",
        "--gradient_accumulation_steps=4",
        "--learning_rate=5e-5", 
        "--rank=32",
        "--max_train_steps=4500",
        "--validation_epochs=500",
        "--checkpointing_steps=500",
        "--seed=42",
        "--mixed_precision=no",  
        "--enable_xformers_memory_efficient_attention",
        "--image_column=image",
        "--caption_column=text",
        "--train_text_encoder"  es
    ]
    
    subprocess.run(training_args)

#entrypoint
@app.local_entrypoint()
def main():
    train_lora.remote()