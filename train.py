import torch
from datasets import load_dataset
from torchvision import transforms
from config.ddpm_ffhq import FFHQTrainingConfig
from model.ddpm_ffhq import init_model
from model.ddpm_scheduler import DDPMScheduler
from PIL import Image
from diffusers import DDPMPipeline
import math
import os
from accelerate import Accelerator
from huggingface_hub import HfFolder, Repository, whoami
from tqdm.auto import tqdm
from pathlib import Path
from trainer import Trainer

config = FFHQTrainingConfig()
dataset = load_dataset(config.dataset_name, data_dir = config.data_dir, drop_labels=False, split="train")

#Set up the dataloader
preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}

dataset.set_transform(transform)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

#Setup model
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
model = init_model(config)
trainer = Trainer(config)
trainer.train(model, train_dataloader, noise_scheduler)


