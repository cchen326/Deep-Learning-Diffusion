import logging
import math
from pathlib import Path
from logging import getLogger as get_logger
from tqdm import tqdm

import torch

from models import UNet, VAE, ClassEmbedder
from schedulers import DDPMScheduler, DDIMScheduler
from pipelines import DDPMPipeline
from utils import seed_everything, load_checkpoint
from torchvision import datasets,transforms
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from train import parse_args

logger = get_logger(__name__)


def save_intermediate_denoising_steps(pipeline, args, device, save_dir, num_samples=4, save_interval=50):
    """
    Generate images and save 2x2 grids at intermediate denoising steps.

    Args:
        pipeline: The diffusion pipeline
        args: Arguments containing generation parameters
        device: Device to run on
        save_dir: Directory to save intermediate grids
        num_samples: Number of samples to generate (should be 4 for 2x2 grid)
        save_interval: Save a grid every N timesteps
    """
    from PIL import Image
    import torch
    from utils import randn_tensor

    logger.info(f"Generating {num_samples} samples with intermediate steps saved every {save_interval} timesteps")

    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Setup
    image_shape = (num_samples, pipeline.unet.input_ch, pipeline.unet.input_size, pipeline.unet.input_size)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    # Get class embeddings if using CFG
    if args.use_cfg:
        classes = torch.randint(0, args.num_classes, (num_samples,), device=device)
        class_embeds = pipeline.class_embedder(classes)
        uncond_classes = torch.full_like(classes, pipeline.class_embedder.num_classes)
        uncond_embeds = pipeline.class_embedder(uncond_classes)
    else:
        class_embeds = None
        uncond_embeds = None

    # Start with random noise
    image = randn_tensor(image_shape, generator=generator, device=device)

    # Set timesteps
    pipeline.scheduler.set_timesteps(num_inference_steps=args.num_inference_steps, device=device)
    timesteps = pipeline.scheduler.timesteps

    logger.info(f"Denoising from t={timesteps[0].item()} to t={timesteps[-1].item()}")

    def save_2x2_grid(images_tensor, timestep, save_path):
        """Save 4 images as a 2x2 grid"""
        cols, rows = 2, 2
        img_size = args.image_size
        grid_image = Image.new('RGB', (cols * img_size, rows * img_size), color=(255, 255, 255))

        # Convert tensor to PIL images
        images_rescaled = (images_tensor + 1) / 2  # [-1, 1] -> [0, 1]
        images_rescaled = images_rescaled.clamp(0, 1)

        for idx in range(min(num_samples, cols * rows)):
            img_tensor = images_rescaled[idx]
            img_np = (img_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype('uint8')
            pil_img = Image.fromarray(img_np)

            x = (idx % cols) * img_size
            y = (idx // cols) * img_size
            grid_image.paste(pil_img, (x, y))

        grid_image.save(save_path)
        logger.info(f"Saved grid at t={timestep} to {save_path}")

    # Denoising loop with intermediate saves
    step_count = 0
    for i, t in enumerate(tqdm(timesteps, desc="Denoising")):
        # Save at intervals
        if i % save_interval == 0 or i == len(timesteps) - 1:
            save_path = save_dir / f"denoising_step_{i:04d}_t{t.item():04d}.png"
            save_2x2_grid(image.clone(), t.item(), save_path)

        # CFG
        if args.use_cfg and class_embeds is not None:
            model_input = torch.cat([image, image], dim=0)
            c = torch.cat([uncond_embeds, class_embeds], dim=0)
        else:
            model_input = image
            c = class_embeds

        # Predict noise
        model_output = pipeline.unet(model_input, t, c)

        # Apply CFG
        if args.use_cfg and class_embeds is not None:
            uncond_model_output, cond_model_output = model_output.chunk(2)
            model_output = uncond_model_output + args.cfg_guidance_scale * (cond_model_output - uncond_model_output)

        # Denoise step
        image = pipeline.scheduler.step(model_output, t, image, generator)
        step_count += 1

    logger.info(f"Saved {step_count // save_interval + 1} intermediate grids to {save_dir}")

    return image


def main():
    # parse arguments
    args = parse_args()
    
    # seed everything
    seed_everything(args.seed)
    
    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    
    # setup model
    logger.info("Creating model")
    # unet
    unet = UNet(input_size=args.unet_in_size, input_ch=args.unet_in_ch, T=args.num_train_timesteps, ch=args.unet_ch, ch_mult=args.unet_ch_mult, attn=args.unet_attn, num_res_blocks=args.unet_num_res_blocks, dropout=args.unet_dropout, conditional=args.use_cfg, c_dim=args.unet_ch)
    # preint number of parameters
    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params / 10 ** 6:.2f}M")
    
    # TODO: ddpm shceduler
    #scheduler = DDPMScheduler(None)
    # vae 
    vae = None
    if args.latent_ddpm:        
        vae = VAE()
        vae.init_from_ckpt('pretrained/model.ckpt')
        vae.eval()
    # cfg
    class_embedder = None
    if args.use_cfg:
        # TODO: class embeder
        class_embedder = ClassEmbedder(embed_dim=args.unet_ch, n_classes=args.num_classes, cond_drop_rate=0.0)
        
    # send to device
    unet = unet.to(device)
    #scheduler = scheduler.to(device)
    if vae:
        vae = vae.to(device)
    if class_embedder:
        class_embedder = class_embedder.to(device)
        
    # scheduler
    if args.use_ddim:
        scheduler_class = DDIMScheduler
    else:
        scheduler_class = DDPMScheduler
    # TOOD: scheduler
    scheduler = scheduler_class(num_train_timesteps=args.num_train_timesteps,
                              num_inference_steps=args.num_inference_steps,
                              beta_start=args.beta_start,
                              beta_end=args.beta_end,
                              beta_schedule=args.beta_schedule,
                              variance_type=args.variance_type,
                              prediction_type=args.prediction_type,
                              clip_sample=args.clip_sample,
                              clip_sample_range=args.clip_sample_range)

    scheduler = scheduler.to(device)
    # load checkpoint
    load_checkpoint(unet, scheduler, vae=vae, class_embedder=class_embedder, checkpoint_path=args.ckpt)
    
    # TODO: pipeline
    pipeline = DDPMPipeline(unet, scheduler, vae, class_embedder)

    # Save intermediate denoising steps (2x2 grids at different timesteps)
    if args.save_intermediate_steps:
        ckpt_path = Path(args.ckpt).resolve()
        intermediate_dir = ckpt_path.parent / "intermediate_denoising"
        logger.info("***** Saving Intermediate Denoising Steps *****")
        save_intermediate_denoising_steps(
            pipeline, args, device,
            save_dir=intermediate_dir,
            num_samples=4,  # 2x2 grid
            save_interval=args.intermediate_save_interval
        )

    logger.info("***** Running Infrence *****")

    # TODO: load validation images as reference batch
    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    val_dataset = datasets.CIFAR10(root="data",train=False,download=True,transform=val_transform)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size=args.batch_size,shuffle=False)

    # TODO: using torchmetrics for evaluation, check the documents of torchmetrics
    import torchmetrics
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore

    # Initialize metrics early
    logger.info("Initializing FID and IS metrics")
    fid_metric = FrechetInceptionDistance(feature=2048, reset_real_features=True, normalize=False, input_img_size=(3, 299, 299), feature_extractor_weights_path=None, antialias=True).to(device)
    is_metric = InceptionScore(feature='logits_unbiased', splits=10, normalize=False).to(device)

    def to_uint8(batch: torch.Tensor) -> torch.Tensor:
        batch = batch.clamp(0.0, 1.0)
        return (batch * 255.0).round().to(torch.uint8)

    # Process real images in chunks
    logger.info("Processing real images for FID...")
    eval_batch_size = args.batch_size
    for images, _ in tqdm(val_dataloader, desc="Processing real images"):
        batch_uint8 = to_uint8(images).to(device)
        fid_metric.update(batch_uint8, real=True)
        del batch_uint8
        torch.cuda.empty_cache()

    # TODO: we run inference to generation 5000 images
    # TODO: with cfg, we generate 50 images per class
    # Process generated images in chunks - update metrics immediately
    preview_images = []  # Keep only first 16 for preview grid
    to_tensor = transforms.ToTensor()
    batch_size = args.batch_size
    total_generated = 0

    if args.use_cfg:
        samples_per_class = max(1, args.samples_per_class)
        for class_idx in tqdm(range(args.num_classes), desc="Generating images"):
            logger.info(f"Generating {samples_per_class} images for class {class_idx}")
            remaining = samples_per_class
            while remaining > 0:
                current_batch_size = min(batch_size, remaining)
                classes = torch.full(
                    (current_batch_size,), class_idx, dtype=torch.long, device=device
                )
                gen_images = pipeline(
                    batch_size=current_batch_size,
                    num_inference_steps=args.num_inference_steps,
                    classes=classes.tolist(),
                    guidance_scale=args.cfg_guidance_scale,
                    generator=generator,
                    device=device,
                )
                gen_images_tensor = torch.stack([to_tensor(img) for img in gen_images], dim=0)

                # Keep first 16 images for preview
                if len(preview_images) < 16:
                    preview_images.append(gen_images_tensor[:min(16 - len(preview_images), current_batch_size)])

                # Update metrics immediately
                batch_uint8 = to_uint8(gen_images_tensor).to(device)
                fid_metric.update(batch_uint8, real=False)
                is_metric.update(batch_uint8)

                total_generated += current_batch_size
                remaining -= current_batch_size

                # Free memory immediately
                del gen_images, gen_images_tensor, batch_uint8
                torch.cuda.empty_cache()
    else:
        total_samples = max(1, args.num_inference_samples)
        remaining = total_samples
        for _ in tqdm(range(math.ceil(total_samples / batch_size)), desc="Generating images"):
            current_batch_size = min(batch_size, remaining)
            if current_batch_size <= 0:
                break
            gen_images = pipeline(
                batch_size=current_batch_size,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                device=device,
            )
            gen_images_tensor = torch.stack([to_tensor(img) for img in gen_images], dim=0)

            # Keep first 16 images for preview
            if len(preview_images) < 16:
                preview_images.append(gen_images_tensor[:min(16 - len(preview_images), current_batch_size)])

            # Update metrics immediately
            batch_uint8 = to_uint8(gen_images_tensor).to(device)
            fid_metric.update(batch_uint8, real=False)
            is_metric.update(batch_uint8)

            total_generated += current_batch_size
            remaining -= current_batch_size

            # Free memory immediately
            del gen_images, gen_images_tensor, batch_uint8
            torch.cuda.empty_cache()

    if total_generated == 0:
        raise ValueError("No images were generated; please check batch size and sample configuration.")

    logger.info(f"Generated {total_generated} images total")

    # Save preview grid from first 16 images
    if preview_images:
        preview_tensor = torch.cat(preview_images, dim=0)[:16]
        grid = make_grid(
            preview_tensor,
            nrow=min(4, preview_tensor.shape[0]),
            padding=2,
            pad_value=1.0,
        )
        preview_image = to_pil_image(grid)
        ckpt_path = Path(args.ckpt).resolve()
        samples_dir = ckpt_path.parent
        samples_dir.mkdir(parents=True, exist_ok=True)
        preview_path = samples_dir / f"inference_samples_preview.png"
        preview_image.save(preview_path)
        logger.info(f"Saved inference preview grid to {preview_path}")
        del preview_tensor, grid, preview_image
        torch.cuda.empty_cache()

    # Compute final metrics
    logger.info("Computing FID and IS scores...")
    fid_score = fid_metric.compute().item()
    is_mean, is_std = is_metric.compute()
    logger.info(f"FID: {fid_score:.3f}, IS: {is_mean:.3f} ± {is_std:.3f}")

    
        
    


if __name__ == '__main__':
    main()
