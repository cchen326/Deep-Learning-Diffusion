import logging
from logging import getLogger as get_logger
from tqdm import tqdm

import torch

from models import UNet, VAE, ClassEmbedder
from schedulers import DDPMScheduler, DDIMScheduler
from pipelines import DDPMPipeline
from utils import seed_everything, load_checkpoint
from torchvision import datasets,transforms
from train import parse_args

logger = get_logger(__name__)


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

    
    logger.info("***** Running Infrence *****")
    
    # TODO: we run inference to generation 5000 images
    # TODO: with cfg, we generate 50 images per class 
    all_images = []
    to_tensor = transforms.ToTensor()
    if args.use_cfg:
        # generate 50 images per class
        for i in tqdm(range(args.num_classes)):
            logger.info(f"Generating 50 images for class {i}")
            batch_size = 50
            classes = torch.full((batch_size,), i, dtype=torch.long, device=device)
            gen_images = pipeline(
                batch_size=batch_size,
                num_inference_steps=args.num_inference_steps,
                classes=classes.tolist(),
                guidance_scale=args.cfg_guidance_scale,
                generator=generator,
                device=device,
            )
            gen_images = torch.stack([to_tensor(img) for img in gen_images], dim=0)
            all_images.append(gen_images)
    else:
        # generate 5000 images
        batch_size = 50
        for _ in tqdm(range(0, 5000, batch_size)):
            gen_images = pipeline(
                batch_size=batch_size,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                device=device,
            )
            gen_images = torch.stack([to_tensor(img) for img in gen_images], dim=0)
            all_images.append(gen_images)
            
    generated_images = torch.cat(all_images, dim=0)
    
    # TODO: load validation images as reference batch
    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    val_dataset = datasets.CIFAR10(root="data",train=False,download=True,transform=val_transform)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size=args.batch_size,shuffle=False)
    val_images = []
    for images, _ in val_dataloader:
        val_images.append(images)
    val_images = torch.cat(val_images,dim=0)

    # TODO: using torchmetrics for evaluation, check the documents of torchmetrics
    import torchmetrics 
    
    from torchmetrics.image.fid import FrechetInceptionDistance, InceptionScore
    
    # TODO: compute FID and IS
    fid_metric = FrechetInceptionDistance(feature=2048, reset_real_features=True, normalize=False, input_img_size=(3, 299, 299), feature_extractor_weights_path=None, antialias=True).to(device)
    is_metric = InceptionScore(feature='logits_unbiased', splits=10, normalize=False).to(device)

    for batch in val_images.split(batch_size):
        fid_metric.update(batch.to(device), real=True)

    for batch in generated_images.split(batch_size):
        batch = batch.to(device)
        fid_metric.update(batch, real=False)
        is_metric.update(batch)

    fid_score = fid_metric.compute().item()
    is_mean, is_std = is_metric.compute()
    logger.info(f"FID: {fid_score:.3f}, IS: {is_mean:.3f} ± {is_std:.3f}")

    
        
    


if __name__ == '__main__':
    main()
