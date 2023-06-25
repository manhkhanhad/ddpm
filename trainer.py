from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
from huggingface_hub import HfFolder, Repository, whoami
from tqdm.auto import tqdm
from pathlib import Path
import os

class Trainer():
    def __init__(self, config):
        self.accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="wandb",
            project_dir=os.path.join(config.output_dir, "logs"),
        )
        
        self.accelerator.init_trackers(
            project_name="ddpm",
            config={"learning_rate": config.learning_rate}
            # init_kwargs={"wandb": {"entity": "my-wandb-team"}}
            )
        
        if self.accelerator.is_main_process:
            if config.push_to_hub:
                repo_name = self.get_full_repo_name(Path(config.output_dir).name)
                repo = Repository(config.output_dir, clone_from=repo_name)
            elif config.output_dir is not None:
                os.makedirs(config.output_dir, exist_ok=True)
            self.accelerator.init_trackers("train_example")
            
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer = optimizer,
            num_warmup_steps = config.lr_warmup_steps,
            num_training_steps = config.num_epochs * len(train_dataloader),
        )

    def get_full_repo_name(model_id: str, organization: str = None, token: str = None):
        if token is None:
            token = HfFolder.get_token()
        if organization is None:
            username = whoami(token)["name"]
            return f"{username}/{model_id}"
        else:
            return f"{organization}/{model_id}"

    def make_grid(images, rows, cols):
        w, h = images[0].size
        grid = Image.new("RGB", size=(cols * w, rows * h))
        for i, image in enumerate(images):
            grid.paste(image, box=(i % cols * w, i // cols * h))
        return grid

    def evaluate(config, epoch, pipeline):
        # Sample some images from random noise (this is the backward diffusion process).
        # The default pipeline output type is `List[PIL.Image]`
        images = pipeline(
            batch_size=config.eval_batch_size,
            generator=torch.manual_seed(config.seed),
        ).images

        # Make a grid out of the images
        image_grid = make_grid(images, rows=4, cols=4)

        # Save the images
        test_dir = os.path.join(config.output_dir, "samples")
        os.makedirs(test_dir, exist_ok=True)
        image_grid.save(f"{test_dir}/{epoch:04d}.png")

    def train(model, noise_scheduler, train_dataloader):
        model, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
            model, self.optimizer, train_dataloader, self.lr_scheduler)
        global_step = 0
        
        for epoch in range(config.num_epochs):
            progress_bar = tqdm(total=len(train_dataloader), disable=not self.accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")
            
            for step, batch in enumerate(train_dataloader):
                clean_images = batch["images"]
                noise = torch.randn(clean_images.shape).to(clean_images.device)
                bs = clean_images.shape[0]
                timesteps = torch.randint(0, noise_scheduler.num_timesteps, size=(bs,), device=clean_images.device)
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                
                with accelerator.accumulate(model):
                    noise_pred = model(noisy_images, timesteps, return_dict=False)
                    loss = F.mse_loss(noise_pred, noise)
                    accelerator.backward(loss)
                    
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                progress_bar.update(1)
                logs = {"loss": loss.detach().items(), 
                        "lr": optimizer.param_groups[0]["lr"],
                        "step": global_step}
                
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                gloacal_step += 1
        
            if self.accelerator.is_main_process:
                pipeline = DDPMPipeline(unet = accelerator.unwrap_model(model), scheduler = noise_scheduler)
                
                if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                    evaluate(config, epoch, pipeline)
                
                if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                    if config.push_to_hub:
                        repo.push_to_hub(commit_message=f"Saving weights and logs of epoch {epoch}")
                    else:
                        pipeline.save_pretrained(config.output_dir)
                    